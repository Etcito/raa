//! S17 integer-tick sampling: tick addressing, key clamping, source-ordered
//! f32 interpolation of prepared runtime keys, plus pose construction.

use crate::{AnimError, Channel, Clip, ClipTrack, Interp, LocalPose, Pose, Rig, RootMotion};
use raa_det::{bezier, lerp, normalize};

/// Address a clip tick. Looping clips wrap by `lengthTicks`; sampling then
/// clamps before the first key and after the last key of each track.
pub fn address_tick(tick: u32, length_ticks: u32, looping: bool) -> u32 {
    if looping && length_ticks != 0 {
        tick % length_ticks
    } else {
        tick
    }
}

fn sample_error(path: &str, message: impl Into<String>) -> AnimError {
    AnimError::new("clip.not_finite", path, message)
}

/// Rotation samples are normalized with sqrt; other channels pass through.
fn finish(channel: Channel, value: [f32; 4], path: &str) -> Result<[f32; 4], AnimError> {
    if channel != Channel::Rotation {
        return Ok(value);
    }
    normalize(value).map_err(|e| sample_error(path, e.to_string()))
}

fn sample_values(
    track: &ClipTrack,
    addressed: u32,
    path: &str,
) -> Result<Option<[f32; 4]>, AnimError> {
    let width = track.channel.width();
    let keys = &track.runtime;
    let Some(first) = keys.first() else {
        return Ok(None);
    };
    if addressed <= first.tick {
        return Ok(Some(finish(track.channel, first.v, path)?));
    }
    let last = keys.last().expect("keys is non-empty");
    if addressed >= last.tick {
        return Ok(Some(finish(track.channel, last.v, path)?));
    }
    let mut i = 0;
    while keys[i + 1].tick <= addressed {
        i += 1;
    }
    let a = &keys[i];
    let b = &keys[i + 1];
    let u = (addressed - a.tick) as f32 / (b.tick - a.tick) as f32;
    let mut out = [0.0f32; 4];
    match a.interp {
        Interp::Constant => out = a.v,
        Interp::Linear => {
            for c in 0..width {
                out[c] = lerp(a.v[c], b.v[c], u).map_err(|e| sample_error(path, e.to_string()))?;
            }
        }
        Interp::Bezier => {
            for c in 0..width {
                out[c] = bezier(a.v[c], a.v[c] + a.hr[c], b.v[c] + b.hl[c], b.v[c], u)
                    .map_err(|e| sample_error(path, e.to_string()))?;
            }
        }
    }
    Ok(Some(finish(track.channel, out, path)?))
}

impl Clip {
    /// Sample track `index` at `tick`, returning its channel components as f32.
    /// An empty track yields an empty vector; `local_pose_at` then keeps the
    /// rig rest value for that bone and channel.
    pub fn sample_track(&self, index: usize, tick: u32) -> Result<Vec<f32>, AnimError> {
        let track = self
            .tracks
            .get(index)
            .ok_or_else(|| AnimError::new("clip.schema", "/tracks", "track index out of range"))?;
        let addressed = address_tick(tick, self.length_ticks, self.looping);
        let path = format!("/tracks/{index}");
        Ok(match sample_values(track, addressed, &path)? {
            Some(values) => values[..track.channel.width()].to_vec(),
            None => Vec::new(),
        })
    }

    /// Per-bone local TRS at `tick`: tracked channels are sampled, everything
    /// else keeps the rig rest value (a bone without a track is unchanged).
    pub fn local_pose_at(&self, rig: &Rig, tick: u32) -> Result<Vec<LocalPose>, AnimError> {
        let mut local = rig.local_rest_pose()?;
        let addressed = address_tick(tick, self.length_ticks, self.looping);
        for (i, track) in self.tracks.iter().enumerate() {
            let Some(bone) = rig.bones.iter().position(|bone| bone.name == track.bone) else {
                return Err(AnimError::new(
                    "clip.bone_unknown",
                    format!("/tracks/{i}/bone"),
                    "bone is not in the rig",
                ));
            };
            let path = format!("/tracks/{i}");
            let Some(values) = sample_values(track, addressed, &path)? else {
                continue;
            };
            match track.channel {
                Channel::Translation => local[bone].t = [values[0], values[1], values[2]],
                Channel::Rotation => local[bone].r = values,
                Channel::Scale => local[bone].s = [values[0], values[1], values[2]],
            }
        }
        if self.root_motion != RootMotion::None {
            if let Some(root) = rig.bones.iter().position(|bone| bone.parent.is_none()) {
                for axis in 0..3 {
                    if self.root_motion == RootMotion::Xyz || axis != 1 {
                        local[root].t[axis] = rig.bones[root].rest.t[axis] as f32;
                    }
                }
            }
        }
        Ok(local)
    }

    /// Stateless adjacent-tick translation delta. Loop seams use the unwrapped
    /// endpoint; selected axes are removed by `local_pose_at` and `pose_at`.
    pub fn root_delta_at(&self, rig: &Rig, tick: u32) -> Result<[f32; 3], AnimError> {
        let mut delta = [0.0; 3];
        if self.root_motion == RootMotion::None || tick == 0 || self.length_ticks == 0 {
            return Ok(delta);
        }
        let Some(root) = rig.bones.iter().find(|bone| bone.parent.is_none()) else {
            return Ok(delta);
        };
        let Some((index, track)) =
            self.tracks.iter().enumerate().find(|(_, track)| {
                track.bone == root.name && track.channel == Channel::Translation
            })
        else {
            return Ok(delta);
        };
        let end = if self.looping {
            let r = tick % self.length_ticks;
            if r == 0 { self.length_ticks } else { r }
        } else {
            tick.min(self.length_ticks)
        };
        let previous = if self.looping {
            end - 1
        } else {
            (tick - 1).min(self.length_ticks)
        };
        let path = format!("/tracks/{index}");
        if let (Some(now), Some(before)) = (
            sample_values(track, end, &path)?,
            sample_values(track, previous, &path)?,
        ) {
            for axis in 0..3 {
                if self.root_motion == RootMotion::Xyz || axis != 1 {
                    delta[axis] = now[axis] - before[axis];
                    if !delta[axis].is_finite() {
                        return Err(sample_error(&path, "root delta is not finite"));
                    }
                }
            }
        }
        Ok(delta)
    }

    /// S16 pose (local TRS, model and skin matrices) at integer `tick`.
    pub fn pose_at(&self, rig: &Rig, tick: u32) -> Result<Pose, AnimError> {
        rig.pose(&self.local_pose_at(rig, tick)?)
    }
}
