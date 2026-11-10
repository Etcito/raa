use raa_core::RaaError;
use raa_sky::{ResolvedWeatherState, Weather, WeatherPreset, WeatherSettings, WeatherUpdate};

fn fair() -> Weather {
    Weather {
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

fn refused(result: Result<(), RaaError>, field: &str) {
    let error = result.unwrap_err();
    assert_eq!(error.code(), "sky.invalid");
    assert!(error.message().contains(field), "{error}");
}

fn parse_refused(text: &str, line: usize, message: &str) {
    let error = Weather::parse(text).unwrap_err();
    assert!(error.contains(&format!("line {line}:")), "{error}");
    assert!(error.contains(message), "{error}");
}

#[test]
fn records_have_the_contracted_fields_and_defaults() {
    let fair = fair();
    assert_eq!(Weather::default(), fair);

    let settings = WeatherSettings::default();
    assert!(settings.enabled);
    assert!(settings.shadows);
    assert!(!settings.changing);
    assert_eq!(settings.seed, 1);
    assert_eq!(settings.transition_seconds, 0.0);
    assert_eq!(settings.target, fair);

    let state = ResolvedWeatherState {
        settings,
        current: fair,
        cycle_preset: None,
    };
    assert_eq!(state.settings, settings);
    assert_eq!(state.current, fair);
    assert_eq!(state.cycle_preset, None);

    let update = WeatherUpdate {
        settings,
        cycle_start: Some(WeatherPreset::Storm),
        restart_cycle: true,
    };
    assert_eq!(update.settings, settings);
    assert_eq!(update.cycle_start, Some(WeatherPreset::Storm));
    assert!(update.restart_cycle);
    let preserve = WeatherUpdate {
        cycle_start: None,
        restart_cycle: false,
        ..update
    };
    assert_eq!(preserve.cycle_start, None);
    assert!(!preserve.restart_cycle);
}

#[test]
fn preset_names_and_ranks_are_lossless() {
    let expected = ["clear", "fair", "broken", "overcast", "storm", "cirrus"];
    assert_eq!(Weather::names().collect::<Vec<_>>(), expected);
    for (rank, name) in expected.iter().enumerate() {
        let preset = WeatherPreset::by_name(name).unwrap();
        assert_eq!(preset.as_str(), *name);
        assert_eq!(usize::from(preset.rank()), rank);
        assert_eq!(WeatherPreset::by_name(preset.as_str()), Some(preset));
        assert!(Weather::by_name(name).is_some());
    }
    assert_eq!(WeatherPreset::by_name("Clear"), None);
    assert_eq!(WeatherPreset::by_name("rain"), None);
    assert_eq!(Weather::by_name("stormy"), None);
}

#[test]
fn built_in_states_match_their_authored_values_and_round_trip() {
    let cases = [
        (
            "clear",
            Weather {
                coverage: 0.0,
                cirrus: 0.05,
                wind_speed: 5.0,
                ..fair()
            },
        ),
        ("fair", fair()),
        (
            "broken",
            Weather {
                coverage: 0.62,
                cloud_type: 0.95,
                base: 1000.0,
                thickness: 2600.0,
                density: 1.15,
                cirrus: 0.3,
                wind_speed: 10.0,
                wind_from: 240.0,
                evolution: 1.3,
                rain: 0.0,
            },
        ),
        (
            "overcast",
            Weather {
                coverage: 0.97,
                cloud_type: 0.12,
                base: 600.0,
                thickness: 900.0,
                density: 1.3,
                cirrus: 0.0,
                wind_speed: 8.0,
                wind_from: 220.0,
                evolution: 0.6,
                rain: 0.0,
            },
        ),
        (
            "storm",
            Weather {
                coverage: 0.85,
                cloud_type: 1.0,
                base: 800.0,
                thickness: 7000.0,
                density: 2.0,
                cirrus: 0.6,
                wind_speed: 16.0,
                wind_from: 230.0,
                evolution: 2.0,
                rain: 1.0,
            },
        ),
        (
            "cirrus",
            Weather {
                coverage: 0.04,
                cloud_type: 0.9,
                cirrus: 0.75,
                wind_speed: 20.0,
                wind_from: 260.0,
                ..fair()
            },
        ),
    ];
    for (name, expected) in cases {
        let weather = Weather::by_name(name).unwrap();
        assert_eq!(weather, expected, "{name}");
        assert_eq!(Weather::parse(&weather.to_text()), Ok(expected), "{name}");
        weather.validate().unwrap();
    }
    assert_eq!(Weather::names().count(), 6);
}

#[test]
fn text_parsing_normalizes_comments_empties_repeats_and_partial_wind() {
    assert_eq!(Weather::parse(""), Ok(fair()));
    assert_eq!(Weather::parse("# only a comment\n   \n"), Ok(fair()));
    assert_eq!(
        Weather::parse("coverage 0.2 # half the sky\n"),
        Ok(Weather {
            coverage: 0.2,
            ..fair()
        })
    );
    assert_eq!(
        Weather::parse("coverage 0.1\ncoverage 0.4\n"),
        Ok(Weather {
            coverage: 0.4,
            ..fair()
        })
    );
    assert_eq!(
        Weather::parse("wind speed=9\n"),
        Ok(Weather {
            wind_speed: 9.0,
            ..fair()
        })
    );
    assert_eq!(
        Weather::parse("wind from=100\n"),
        Ok(Weather {
            wind_from: 100.0,
            ..fair()
        })
    );
    let clamped = Weather {
        coverage: 1.0,
        cloud_type: 0.0,
        base: 0.0,
        thickness: 100.0,
        density: 0.0,
        cirrus: 1.0,
        wind_speed: 0.0,
        wind_from: 725.0,
        evolution: 0.0,
        rain: 1.0,
    };
    assert_eq!(
        Weather::parse(
            "coverage 2\ntype -1\nbase -5\nthickness 1\ndensity -3\ncirrus 2\n\
             wind speed=-2 from=725\nevolution -1\nrain 3\n"
        ),
        Ok(clamped)
    );
    let text = Weather::default().to_text();
    assert_eq!(
        text,
        "coverage 0.3\ntype 0.9\nbase 1200\nthickness 1600\ndensity 1\ncirrus 0.15\n\
         wind speed=7 from=250\nevolution 1\nrain 0\n"
    );
    assert!(text.ends_with('\n'));
    assert_eq!(Weather::parse(&text), Ok(Weather::default()));
}

#[test]
fn text_parser_refuses_unknown_keys_and_malformed_numbers_with_one_based_lines() {
    parse_refused("coverage 0.2\nbogus 1\n", 2, "unknown setting");
    parse_refused("wind speed=1 gust=2\n", 1, "unknown wind setting");
    parse_refused("wind speed=1\ngust 2\n", 2, "unknown setting");
    parse_refused("wind speed\n", 1, "expected key=value");
    parse_refused("wind speed=1 speed\n", 1, "expected key=value");
    parse_refused("coverage nope\n", 1, "not a number");
    parse_refused("\ncoverage nan\n", 2, "not a number");
    parse_refused("coverage inf\n", 1, "not a number");
    parse_refused("coverage 1e999\n", 1, "not a number");
    parse_refused("coverage 0.1 0.2\n", 1, "takes one number");
    parse_refused("base\n", 1, "takes one number");
    parse_refused("rain 0\ntype 4 5\n", 2, "takes one number");
}

#[test]
fn structured_validation_refuses_out_of_domain_and_nonfinite_fields() {
    Weather::default().validate().unwrap();
    Weather {
        coverage: 1.0,
        cloud_type: 0.0,
        base: 0.0,
        thickness: 100.0,
        density: 0.0,
        cirrus: 1.0,
        wind_speed: 0.0,
        wind_from: -725.0,
        evolution: 0.0,
        rain: 1.0,
    }
    .validate()
    .unwrap();
    for (weather, field) in [
        (
            Weather {
                coverage: 1.1,
                ..fair()
            },
            "coverage",
        ),
        (
            Weather {
                coverage: f32::NAN,
                ..fair()
            },
            "coverage",
        ),
        (
            Weather {
                cloud_type: -0.1,
                ..fair()
            },
            "cloud_type",
        ),
        (
            Weather {
                base: -1.0,
                ..fair()
            },
            "base",
        ),
        (
            Weather {
                base: f32::INFINITY,
                ..fair()
            },
            "base",
        ),
        (
            Weather {
                thickness: 99.0,
                ..fair()
            },
            "thickness",
        ),
        (
            Weather {
                thickness: f32::NAN,
                ..fair()
            },
            "thickness",
        ),
        (
            Weather {
                density: -0.1,
                ..fair()
            },
            "density",
        ),
        (
            Weather {
                cirrus: 1.1,
                ..fair()
            },
            "cirrus",
        ),
        (
            Weather {
                wind_speed: -1.0,
                ..fair()
            },
            "wind_speed",
        ),
        (
            Weather {
                wind_from: f32::NAN,
                ..fair()
            },
            "wind_from",
        ),
        (
            Weather {
                evolution: -0.1,
                ..fair()
            },
            "evolution",
        ),
        (
            Weather {
                rain: 1.1,
                ..fair()
            },
            "rain",
        ),
    ] {
        refused(weather.validate(), field);
    }
}

#[test]
fn nonfinite_derived_totals_are_refused_after_finite_fields_pass() {
    let extreme = Weather {
        base: 3.0e38,
        thickness: 3.0e38,
        ..fair()
    };
    assert!(extreme.base.is_finite() && extreme.thickness.is_finite());
    refused(extreme.validate(), "base + thickness");
    let max = Weather {
        base: f32::MAX,
        thickness: f32::MAX,
        ..fair()
    };
    refused(max.validate(), "base + thickness");
}

#[test]
fn settings_and_update_validation_cover_transition_and_target() {
    WeatherSettings::default().validate().unwrap();
    refused(
        WeatherSettings {
            transition_seconds: -1.0,
            ..WeatherSettings::default()
        }
        .validate(),
        "transition_seconds",
    );
    refused(
        WeatherSettings {
            transition_seconds: f32::NAN,
            ..WeatherSettings::default()
        }
        .validate(),
        "transition_seconds",
    );
    refused(
        WeatherSettings {
            target: Weather {
                rain: 2.0,
                ..fair()
            },
            ..WeatherSettings::default()
        }
        .validate(),
        "rain",
    );
    let update = WeatherUpdate {
        settings: WeatherSettings {
            transition_seconds: -1.0,
            ..WeatherSettings::default()
        },
        cycle_start: None,
        restart_cycle: false,
    };
    refused(update.validate(), "transition_seconds");
}

#[test]
fn authored_text_keeps_field_order_and_round_trips() {
    let weather = Weather {
        coverage: 0.0,
        cloud_type: 0.12,
        base: 600.0,
        thickness: 900.0,
        density: 1.3,
        cirrus: 0.7,
        wind_speed: 8.5,
        wind_from: 359.5,
        evolution: 0.6,
        rain: 0.25,
    };
    assert_eq!(
        weather.to_text(),
        "coverage 0\ntype 0.12\nbase 600\nthickness 900\ndensity 1.3\ncirrus 0.7\n\
         wind speed=8.5 from=359.5\nevolution 0.6\nrain 0.25\n"
    );
    assert_eq!(Weather::parse(&weather.to_text()), Ok(weather));
}
