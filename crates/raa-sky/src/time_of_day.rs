//! Solar/lunar positions and a celestial basis, in right-handed scene coordinates (+Y up).
//! Astronomy and defaults adapted from Solstice's `crates/solstice-render/src/time_of_day.rs`
//! at revision 7e465b599c13da165948ebaa44df824c36c3ae07.
/*
MIT License

Copyright (c) 2026 SamG-Coder

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

use raa_core::RaaError;
use raa_math::{Quat, Vec3};
use std::f32::consts::{FRAC_PI_2, TAU};

const LUNAR_MONTH: f32 = 29.530_59;
const A_NEW_MOON: f32 = 6.0;
const OBLIQUITY: f32 = 23.44;

/// Observer's latitude and scene orientation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Place {
    /// Degrees north, in [-90, 90].
    pub latitude: f32,
    /// Finite radians about +Y; zero puts north at -Z and east at +X.
    pub north: f32,
}

impl Default for Place {
    fn default() -> Self {
        Self {
            latitude: 38.0,
            north: 0.0,
        }
    }
}

impl Place {
    pub fn validate(&self) -> Result<(), RaaError> {
        if !self.latitude.is_finite() || !(-90.0..=90.0).contains(&self.latitude) {
            return Err(invalid("latitude must be finite degrees in [-90, 90]"));
        }
        if !self.north.is_finite() {
            return Err(invalid("north must be finite radians"));
        }
        Ok(())
    }
}

/// A moment in a repeating 365-day year, using local solar time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimeOfDay {
    /// One-based day in 1..=365.
    pub day: u16,
    /// Finite hours in [0, 24); noon is 12.
    pub hours: f64,
}

impl Default for TimeOfDay {
    fn default() -> Self {
        Self {
            day: 172,
            hours: 9.0,
        }
    }
}

impl TimeOfDay {
    pub fn validate(&self) -> Result<(), RaaError> {
        if !(1..=365).contains(&self.day) {
            return Err(invalid("day must be in 1..=365"));
        }
        if !self.hours.is_finite() || !(0.0..24.0).contains(&self.hours) {
            return Err(invalid("hours must be finite and in [0, 24)"));
        }
        Ok(())
    }

    /// Advances by signed finite hours, wrapping both day and year; refusals are atomic.
    pub fn advance(&mut self, hours: f64) -> Result<(), RaaError> {
        self.validate()?;
        if !hours.is_finite() {
            return Err(invalid("advance.hours must be finite"));
        }
        const YEAR_HOURS: f64 = 365.0 * 24.0;
        // Reduce before adding so even f64::MAX cannot overflow. The second remainder
        // handles rem_euclid rounding a tiny negative value up to YEAR_HOURS itself.
        let cursor = (f64::from(self.day - 1) * 24.0 + self.hours + hours % YEAR_HOURS)
            .rem_euclid(YEAR_HOURS)
            % YEAR_HOURS;
        self.day = (cursor / 24.0).floor() as u16 + 1;
        self.hours = cursor % 24.0;
        Ok(())
    }

    /// Evaluates validated time/place data. Directions point toward bodies; negate them
    /// when constructing RAA directional lights, which store light travel direction.
    pub fn sky(&self, place: &Place) -> CelestialSky {
        let latitude = place.latitude.to_radians();
        let day = f32::from(self.day);
        let hours = self.hours as f32;
        let year = TAU * (day + 10.0) / 365.0;
        let sun_declination = -OBLIQUITY.to_radians() * year.cos();
        let sun_hour_angle = TAU * (hours - 12.0) / 24.0;
        let (sun_alt, sun_az) = altitude_azimuth(sun_declination, sun_hour_angle, latitude);
        let age = ((day + hours / 24.0 - A_NEW_MOON) / LUNAR_MONTH).rem_euclid(1.0);
        let moon_lit = 0.5 - 0.5 * (TAU * age).cos();
        let moon_hour_angle = sun_hour_angle - TAU * age;
        let moon_declination = sun_declination * (TAU * age).cos();
        let (moon_alt, moon_az) = altitude_azimuth(moon_declination, moon_hour_angle, latitude);
        let sun_longitude = year - FRAC_PI_2;
        let sun_right_ascension =
            (OBLIQUITY.to_radians().cos() * sun_longitude.sin()).atan2(sun_longitude.cos());
        let sidereal_time = sun_hour_angle + sun_right_ascension;
        let (equinox_alt, equinox_az) = altitude_azimuth(0.0, sidereal_time, latitude);
        CelestialSky {
            sun: horizontal(sun_alt, sun_az, place),
            moon: horizontal(moon_alt, moon_az, place),
            moon_lit,
            pole: horizontal(latitude, 0.0, place),
            equinox: horizontal(equinox_alt, equinox_az, place),
        }
    }
}

/// Unit directions toward bodies and an orthonormal celestial basis in scene space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CelestialSky {
    pub sun: Vec3,
    pub moon: Vec3,
    /// Illuminated fraction in [0, 1], from new to full moon.
    pub moon_lit: f32,
    pub pole: Vec3,
    pub equinox: Vec3,
}

impl CelestialSky {
    /// Direction of a star at finite right ascension and declination, in radians.
    pub fn celestial(&self, right_ascension: f32, declination: f32) -> Vec3 {
        let (sin_dec, cos_dec) = declination.sin_cos();
        let (sin_ra, cos_ra) = right_ascension.sin_cos();
        let towards_ra_6h = self.pole.cross(self.equinox);
        self.equinox * cos_dec * cos_ra + towards_ra_6h * cos_dec * sin_ra + self.pole * sin_dec
    }
}

/// Authored settings; playback uses a separate runtime cursor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimeOfDaySettings {
    pub time: TimeOfDay,
    pub place: Place,
    pub enabled: bool,
    pub playing: bool,
    /// Nonnegative simulated seconds per real second.
    pub speed: f64,
}

impl Default for TimeOfDaySettings {
    fn default() -> Self {
        Self {
            time: TimeOfDay::default(),
            place: Place::default(),
            enabled: false,
            playing: false,
            speed: 1.0,
        }
    }
}

impl TimeOfDaySettings {
    pub fn validate(&self) -> Result<(), RaaError> {
        self.time.validate()?;
        self.place.validate()?;
        if !self.speed.is_finite() || self.speed < 0.0 {
            return Err(invalid("speed must be finite and nonnegative"));
        }
        Ok(())
    }
}

/// Source of the accepted directional light.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DominantLight {
    #[default]
    Manual,
    Sun,
    Moon,
}

/// Authored settings and their current evaluated snapshot.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimeOfDayState {
    pub settings: TimeOfDaySettings,
    pub current: TimeOfDay,
    pub sky: CelestialSky,
    pub dominant: DominantLight,
}

impl Default for TimeOfDayState {
    fn default() -> Self {
        let settings = TimeOfDaySettings::default();
        Self {
            settings,
            current: settings.time,
            sky: settings.time.sky(&settings.place),
            dominant: DominantLight::Manual,
        }
    }
}

fn horizontal(altitude: f32, azimuth: f32, place: &Place) -> Vec3 {
    let (sin_alt, cos_alt) = altitude.sin_cos();
    let (sin_az, cos_az) = azimuth.sin_cos();
    let local = Vec3::new(cos_alt * sin_az, sin_alt, -cos_alt * cos_az);
    Quat::from_rotation_y(place.north).rotate(local)
}

fn altitude_azimuth(declination: f32, hour_angle: f32, latitude: f32) -> (f32, f32) {
    let sin_alt =
        latitude.sin() * declination.sin() + latitude.cos() * declination.cos() * hour_angle.cos();
    let altitude = sin_alt.clamp(-1.0, 1.0).asin();
    let y = -hour_angle.sin() * declination.cos();
    let x =
        declination.sin() * latitude.cos() - declination.cos() * latitude.sin() * hour_angle.cos();
    (altitude, y.atan2(x))
}

fn invalid(message: &str) -> RaaError {
    RaaError::FrameGraph {
        code: "sky.invalid",
        message: message.into(),
    }
}
