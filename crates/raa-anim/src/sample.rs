//! S17 integer-tick sampling: tick addressing, key clamping and source-ordered
//! f32 interpolation, plus pose construction through the S16 matrices.

use crate::{AnimError, Channel, Clip, ClipTrack, Interp, JsonValue, LocalPose, Pose, Rig};
use raa_det::{bezier, lerp, normalize, sign_align};

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

fn numbers<const N: usize>(
    value: &JsonValue,
    width: usize,
    path: &str,
) -> Result<[f32; N], AnimError> {
    let JsonValue::Array(items) = value else {
        return Err(AnimError::new("clip.schema", path, "expected array"));
    };
    if width > N || items.len() != width {
        return Err(AnimError::new("clip.schema", path, "wrong channel width"));
    }
    let mut out = [0.0f32; N];
    for (i, item) in items.iter().enumerate() {
        let JsonValue::Number(n) = item else {
            return Err(AnimError::new("clip.schema", path, "expected number"));
        };
        let value = *n as f32;
        if !value.is_finite() {
            return Err(sample_error(path, "value is not finite in f32"));
        }
        out[i] = value;
    }
    Ok(out)
}

/// A handle as value-space offsets: the document dt offset is ignored.
fn handle(value: Option<&JsonValue>, width: usize, path: &str) -> Result<[f32; 4], AnimError> {
    let Some(value) = value else {
        return Ok([0.0; 4]);
    };
    let all: [f32; 5] = numbers(value, width + 1, path)?;
    Ok([all[1], all[2], all[3], all[4]])
}

fn sample_values(
    track: &ClipTrack,
    addressed: u32,
    path: &str,
) -> Result<Option<[f32; 4]>, AnimError> {
    let width = track.channel.width();
    let Some(first) = track.keys.first() else {
        return Ok(None);
    };
    if addressed <= first.tick {
        return Ok(Some(numbers(&first.v, width, path)?));
    }
    let last = track.keys.last().expect("keys is non-empty");
    if addressed >= last.tick {
        return Ok(Some(numbers(&last.v, width, path)?));
    }
    let mut i = 0;
    while track.keys[i + 1].tick <= addressed {
        i += 1;
    }
    let a = &track.keys[i];
    let b = &track.keys[i + 1];
    let av = numbers(&a.v, width, path)?;
    let mut bv = numbers(&b.v, width, path)?;
    let mut flip = false;
    if track.channel == Channel::Rotation {
        let aligned = sign_align(av, bv).map_err(|e| sample_error(path, e.to_string()))?;
        flip = aligned != bv;
        bv = aligned;
    }
    let u = (addressed - a.tick) as f32 / (b.tick - a.tick) as f32;
    let mut out = [0.0f32; 4];
    match a.interp {
        Interp::Constant => out = av,
        Interp::Linear => {
            for c in 0..width {
                out[c] = lerp(av[c], bv[c], u).map_err(|e| sample_error(path, e.to_string()))?;
            }
        }
        Interp::Bezier => {
            let hr = handle(a.hr.as_ref(), width, path)?;
            let hl = handle(b.hl.as_ref(), width, path)?;
            for c in 0..width {
                let p2 = if flip { bv[c] - hl[c] } else { bv[c] + hl[c] };
                out[c] = bezier(av[c], av[c] + hr[c], p2, bv[c], u)
                    .map_err(|e| sample_error(path, e.to_string()))?;
            }
        }
    }
    if track.channel == Channel::Rotation {
        out = normalize(out).map_err(|e| sample_error(path, e.to_string()))?;
    }
    Ok(Some(out))
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
        Ok(local)
    }

    /// S16 pose (local TRS, model and skin matrices) at integer `tick`.
    pub fn pose_at(&self, rig: &Rig, tick: u32) -> Result<Pose, AnimError> {
        rig.pose(&self.local_pose_at(rig, tick)?)
    }
}
