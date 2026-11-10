//! Weather records and their authored text form.
//!
//! A state is written one setting per line; missing settings keep the fair-weather defaults,
//! `#` starts a comment and the last repeated setting wins. Distances are metres, wind speed is
//! metres per second, `wind_from` is a compass direction FROM north in degrees, and fractions
//! are in `0..1`. Parsing normalizes values; [`Weather::validate`] is the strict structured
//! boundary that refuses out-of-domain or nonfinite fields with `sky.invalid`.
//!
//! Text and values are adapted from Solstice's `crates/solstice-render/src/clouds/weather.rs`
//! and `weather/*.weather` at revision 7e465b599c13da165948ebaa44df824c36c3ae07.
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

/// What the sky is doing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Weather {
    /// Share of the sky covered by the main cloud layer, in [0, 1].
    pub coverage: f32,
    /// Cloud kind: 0 flat stratus, 0.5 stratocumulus, 1 tall cumulus.
    pub cloud_type: f32,
    /// Cloud-base height above the ground, in metres.
    pub base: f32,
    /// Cloud-layer depth in metres.
    pub thickness: f32,
    /// Cloud optical thickness relative to ordinary cloud.
    pub density: f32,
    /// Share of the sky covered by high, thin cirrus, in [0, 1].
    pub cirrus: f32,
    /// Wind speed at cloud height, in metres per second.
    pub wind_speed: f32,
    /// Compass direction the wind blows FROM, in degrees.
    pub wind_from: f32,
    /// Cloud growth and fade rate relative to ordinary weather.
    pub evolution: f32,
    /// Rain intensity in [0, 1].
    pub rain: f32,
}

/// The built-in weather states, in rank order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WeatherPreset {
    Clear,
    Fair,
    Broken,
    Overcast,
    Storm,
    Cirrus,
}

const BUILT_IN: [(WeatherPreset, &str); 6] = [
    (
        WeatherPreset::Clear,
        include_str!("../weather/clear.weather"),
    ),
    (WeatherPreset::Fair, include_str!("../weather/fair.weather")),
    (
        WeatherPreset::Broken,
        include_str!("../weather/broken.weather"),
    ),
    (
        WeatherPreset::Overcast,
        include_str!("../weather/overcast.weather"),
    ),
    (
        WeatherPreset::Storm,
        include_str!("../weather/storm.weather"),
    ),
    (
        WeatherPreset::Cirrus,
        include_str!("../weather/cirrus.weather"),
    ),
];

impl WeatherPreset {
    /// Exact lowercase name used by text, presets and records.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Clear => "clear",
            Self::Fair => "fair",
            Self::Broken => "broken",
            Self::Overcast => "overcast",
            Self::Storm => "storm",
            Self::Cirrus => "cirrus",
        }
    }

    /// Stable built-in rank: clear 0 through cirrus 5.
    pub const fn rank(self) -> u8 {
        match self {
            Self::Clear => 0,
            Self::Fair => 1,
            Self::Broken => 2,
            Self::Overcast => 3,
            Self::Storm => 4,
            Self::Cirrus => 5,
        }
    }

    /// The preset with that exact lowercase name, or `None`.
    pub fn by_name(name: &str) -> Option<Self> {
        match name {
            "clear" => Some(Self::Clear),
            "fair" => Some(Self::Fair),
            "broken" => Some(Self::Broken),
            "overcast" => Some(Self::Overcast),
            "storm" => Some(Self::Storm),
            "cirrus" => Some(Self::Cirrus),
            _ => None,
        }
    }

    fn text(self) -> &'static str {
        BUILT_IN
            .iter()
            .find(|(preset, _)| *preset == self)
            .map(|(_, text)| *text)
            .unwrap_or_default()
    }

    fn weather(self) -> Weather {
        Weather::parse(self.text()).unwrap_or_default()
    }
}

impl Default for Weather {
    /// Fair weather, the source and structured default target.
    fn default() -> Self {
        Self::base_state()
    }
}

impl Weather {
    /// Names of the built-in states, in rank order.
    pub fn names() -> impl Iterator<Item = &'static str> {
        BUILT_IN.iter().map(|(preset, _)| preset.as_str())
    }

    /// A built-in state by exact lowercase name, or `None`.
    pub fn by_name(name: &str) -> Option<Self> {
        WeatherPreset::by_name(name).map(WeatherPreset::weather)
    }

    /// Fair-weather defaults: the base every text file starts from.
    fn base_state() -> Self {
        Self {
            coverage: 0.3,
            cloud_type: 0.9,
            base: 1200.0,
            thickness: 1600.0,
            density: 1.0,
            cirrus: 0.15,
            wind_speed: 7.0,
            wind_from: 250.0,
            evolution: 1.0,
            rain: 0.0,
        }
    }
    /// Reads a state from its text form. Errors name the one-based line at fault.
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut weather = Self::base_state();
        for (index, raw) in text.lines().enumerate() {
            let words: Vec<&str> = raw
                .split('#')
                .next()
                .unwrap_or("")
                .split_whitespace()
                .collect();
            let Some((&keyword, rest)) = words.split_first() else {
                continue;
            };
            let at = |e: String| format!("line {}: {e}", index + 1);
            let single = || match rest {
                [value] => number(value).map_err(at),
                _ => Err(at(format!("{keyword} takes one number"))),
            };
            match keyword {
                "coverage" => weather.coverage = single()?.clamp(0.0, 1.0),
                "type" => weather.cloud_type = single()?.clamp(0.0, 1.0),
                "base" => weather.base = single()?.max(0.0),
                "thickness" => weather.thickness = single()?.max(100.0),
                "density" => weather.density = single()?.max(0.0),
                "cirrus" => weather.cirrus = single()?.clamp(0.0, 1.0),
                "evolution" => weather.evolution = single()?.max(0.0),
                "rain" => weather.rain = single()?.clamp(0.0, 1.0),
                "wind" => {
                    for word in rest {
                        let (key, value) = word
                            .split_once('=')
                            .ok_or_else(|| at(format!("expected key=value, found {word:?}")))?;
                        match key {
                            "speed" => weather.wind_speed = number(value).map_err(at)?.max(0.0),
                            "from" => weather.wind_from = number(value).map_err(at)?,
                            _ => return Err(at(format!("unknown wind setting {key}"))),
                        }
                    }
                }
                _ => return Err(at(format!("unknown setting {keyword:?}"))),
            }
        }
        Ok(weather)
    }

    /// Writes this state in the form [`Weather::parse`] reads: one setting per line in
    /// authored order, ending with a final LF.
    pub fn to_text(&self) -> String {
        format!(
            "coverage {}\ntype {}\nbase {}\nthickness {}\ndensity {}\ncirrus {}\nwind speed={} from={}\nevolution {}\nrain {}\n",
            self.coverage,
            self.cloud_type,
            self.base,
            self.thickness,
            self.density,
            self.cirrus,
            self.wind_speed,
            self.wind_from,
            self.evolution,
            self.rain
        )
    }

    /// Strict structured validation: refuses out-of-domain and nonfinite fields, including a
    /// nonfinite `base + thickness`, with `sky.invalid`.
    pub fn validate(&self) -> Result<(), RaaError> {
        fraction("coverage", self.coverage)?;
        fraction("cloud_type", self.cloud_type)?;
        nonnegative("base", self.base)?;
        if !self.thickness.is_finite() || self.thickness < 100.0 {
            return Err(invalid(
                "thickness must be finite metres and at least 100".into(),
            ));
        }
        nonnegative("density", self.density)?;
        fraction("cirrus", self.cirrus)?;
        nonnegative("wind_speed", self.wind_speed)?;
        if !self.wind_from.is_finite() {
            return Err(invalid("wind_from must be finite degrees".into()));
        }
        nonnegative("evolution", self.evolution)?;
        fraction("rain", self.rain)?;
        if !(self.base + self.thickness).is_finite() {
            return Err(invalid("base + thickness must be finite".into()));
        }
        Ok(())
    }
}

/// Authored weather settings. Cycle cursor, interpolation and observations are runtime state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeatherSettings {
    pub enabled: bool,
    pub shadows: bool,
    pub changing: bool,
    pub seed: u32,
    /// Easing time for authored target changes, in seconds.
    pub transition_seconds: f32,
    pub target: Weather,
}

impl Default for WeatherSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            shadows: true,
            changing: false,
            seed: 1,
            transition_seconds: 0.0,
            target: Weather::default(),
        }
    }
}

impl WeatherSettings {
    /// Validates the complete authored candidate before any mutation.
    pub fn validate(&self) -> Result<(), RaaError> {
        self.target.validate()?;
        if !self.transition_seconds.is_finite() || self.transition_seconds < 0.0 {
            return Err(invalid(
                "transition_seconds must be finite and nonnegative".into(),
            ));
        }
        Ok(())
    }
}

/// An accepted weather snapshot plus the active cycle's preset identity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResolvedWeatherState {
    pub settings: WeatherSettings,
    pub current: Weather,
    /// `None` whenever no cycle is active.
    pub cycle_preset: Option<WeatherPreset>,
}

/// A structured weather edit carrying explicit cycle intent.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeatherUpdate {
    pub settings: WeatherSettings,
    /// `None` preserves the stored start identity, initially fair; `Some` supplies it
    /// explicitly, even when target values have overrides.
    pub cycle_start: Option<WeatherPreset>,
    /// Explicit restart intent, not an inference from descriptor equality. It never enables
    /// `changing` implicitly.
    pub restart_cycle: bool,
}

impl WeatherUpdate {
    /// Validates the complete candidate before any mutation.
    pub fn validate(&self) -> Result<(), RaaError> {
        self.settings.validate()
    }
}

fn number(text: &str) -> Result<f32, String> {
    match text.parse::<f32>() {
        Ok(value) if value.is_finite() => Ok(value),
        _ => Err(format!("{text:?} is not a number")),
    }
}

fn fraction(field: &str, value: f32) -> Result<(), RaaError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(invalid(format!("{field} must be finite in [0, 1]")))
    }
}

fn nonnegative(field: &str, value: f32) -> Result<(), RaaError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(invalid(format!("{field} must be finite and nonnegative")))
    }
}

fn invalid(message: String) -> RaaError {
    RaaError::FrameGraph {
        code: "sky.invalid",
        message,
    }
}
