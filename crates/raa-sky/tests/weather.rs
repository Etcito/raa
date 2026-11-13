use raa_core::RaaError;
use raa_sky::{
    Observation, ResolvedWeatherState, Weather, WeatherCycle, WeatherPreset, WeatherSettings,
    WeatherUpdate,
};

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

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() < 1.0e-4
}

#[test]
fn observations_carry_percent_kmh_degrees_code_and_millimetre_units() {
    let report = Observation {
        cover_low: 40.0,
        cover_mid: 20.0,
        cover_high: 60.0,
        wind_kmh: 18.0,
        wind_from: 725.0,
        code: 2,
        precipitation: 2.0,
    };
    let weather = Weather::observed(&report);
    assert!(close(weather.coverage, 0.36), "{}", weather.coverage);
    assert!(close(weather.cloud_type, 0.95));
    assert_eq!(weather.base, 1000.0);
    assert_eq!(weather.thickness, 1500.0);
    assert_eq!(weather.density, 1.0);
    assert!(close(weather.cirrus, 0.6));
    assert!(close(weather.wind_speed, 10.0), "{}", weather.wind_speed);
    assert_eq!(weather.wind_from, 725.0);
    assert_eq!(weather.rain, 0.0);
    assert_eq!(weather.evolution, 1.0);
}

#[test]
fn nonfinite_reports_normalize_to_zero_and_negative_values_clamp() {
    let weather = Weather::observed(&Observation {
        cover_low: f32::NAN,
        cover_mid: f32::INFINITY,
        cover_high: -10.0,
        wind_kmh: -5.0,
        wind_from: f32::NEG_INFINITY,
        code: 1,
        precipitation: -2.0,
    });
    assert_eq!(weather.coverage, 0.0);
    assert_eq!(weather.cirrus, 0.0);
    assert_eq!(weather.wind_speed, 0.0);
    assert_eq!(weather.wind_from, 0.0);
    assert_eq!(weather.rain, 0.0);

    let clamped = Weather::observed(&Observation {
        cover_low: 250.0,
        code: 3,
        ..Default::default()
    });
    assert!(close(clamped.coverage, 0.9), "{}", clamped.coverage);
    let negative_cover = Weather::observed(&Observation {
        cover_low: -100.0,
        code: 3,
        ..Default::default()
    });
    assert_eq!(negative_cover.coverage, 0.0);
}

#[test]
fn unknown_wmo_codes_take_the_general_cover_branch() {
    let unknown = Weather::observed(&Observation {
        cover_low: 100.0,
        code: 42,
        ..Default::default()
    });
    assert!(unknown.coverage > 0.85);
    assert_eq!(unknown.rain, 0.0);
    assert!(unknown.thickness < 3000.0);

    let clear = Weather::observed(&Observation {
        cover_low: 100.0,
        code: 0,
        ..Default::default()
    });
    assert!(clear.coverage <= 0.05);
}

#[test]
fn report_codes_map_to_cloud_rain_and_wind_behavior() {
    let showers = Weather::observed(&Observation {
        cover_low: 80.0,
        code: 80,
        wind_kmh: 36.0,
        precipitation: 2.0,
        ..Default::default()
    });
    assert!(showers.cloud_type >= 0.85);
    assert_eq!(showers.thickness, 3500.0);
    assert_eq!(showers.density, 1.5);
    assert_eq!(showers.base, 800.0);
    assert!(close(showers.rain, 0.8));
    assert!(close(showers.evolution, 1.8));
    assert!(close(showers.wind_speed, 20.0));

    let storm = Weather::observed(&Observation {
        cover_low: 80.0,
        cover_high: 90.0,
        code: 95,
        precipitation: 10.0,
        ..Default::default()
    });
    assert_eq!(storm.cloud_type, 1.0);
    assert_eq!(storm.base, 900.0);
    assert_eq!(storm.thickness, 7000.0);
    assert_eq!(storm.density, 2.0);
    assert_eq!(storm.rain, 1.0);
    assert!(close(storm.cirrus, 0.9));
    assert_eq!(storm.evolution, 2.0);

    let rain = Weather::observed(&Observation {
        cover_mid: 60.0,
        code: 63,
        precipitation: 8.0,
        ..Default::default()
    });
    assert_eq!(rain.base, 800.0);
    assert_eq!(rain.thickness, 1200.0);
    assert_eq!(rain.density, 1.5);
    assert_eq!(rain.rain, 1.0);
    assert!(rain.cloud_type < 0.6);
}

#[test]
fn interpolation_eases_fields_and_turns_the_wind_the_short_way() {
    let west = Weather {
        wind_from: 350.0,
        ..Weather::default()
    };
    let east = Weather {
        wind_from: 10.0,
        ..Weather::default()
    };
    assert_eq!(west.lerp(&east, 0.0), west);
    assert_eq!(west.lerp(&east, 1.0), east);

    let quarter = west.lerp(&east, 0.25);
    assert!(close(quarter.wind_from, 355.0), "{}", quarter.wind_from);
    let half = west.lerp(&east, 0.5);
    assert!(
        half.wind_from < 1.0e-3 || (half.wind_from - 360.0).abs() < 1.0e-3,
        "{}",
        half.wind_from
    );
    let back = east.lerp(&west, 0.5);
    assert!(
        back.wind_from < 1.0e-3 || (back.wind_from - 360.0).abs() < 1.0e-3,
        "{}",
        back.wind_from
    );

    let clear = Weather::by_name("clear").unwrap();
    let storm = Weather::by_name("storm").unwrap();
    let middle = clear.lerp(&storm, 0.5);
    assert!(close(
        middle.coverage,
        (clear.coverage + storm.coverage) * 0.5
    ));
    assert!(close(middle.rain, storm.rain * 0.5));
}

#[test]
fn wind_velocity_is_the_from_direction_in_east_and_south_coordinates() {
    let weather = |from: f32| Weather {
        wind_speed: 10.0,
        wind_from: from,
        ..Weather::default()
    };
    let [east, south] = weather(0.0).wind_velocity();
    assert!(east.abs() < 1.0e-4 && close(south, 10.0));
    let [east, south] = weather(90.0).wind_velocity();
    assert!(close(east, -10.0) && south.abs() < 1.0e-4);
    let [east, south] = weather(180.0).wind_velocity();
    assert!(east.abs() < 1.0e-4 && close(south, -10.0));
    let [east, south] = weather(270.0).wind_velocity();
    assert!(close(east, 10.0) && south.abs() < 1.0e-4);

    let turned = weather(725.0).wind_velocity();
    let direct = weather(5.0).wind_velocity();
    assert!(close(turned[0], direct[0]) && close(turned[1], direct[1]));
}

#[test]
fn cycles_move_only_to_neighbouring_weather_and_are_deterministic() {
    let neighbours = |name: &str| match name {
        "clear" => vec!["fair", "cirrus"],
        "cirrus" => vec!["fair", "broken"],
        "fair" => vec!["broken", "clear"],
        "broken" => vec!["overcast", "fair"],
        "overcast" => vec!["storm", "broken"],
        "storm" => vec!["broken", "overcast"],
        other => panic!("unknown weather {other}"),
    };
    let run = |seed: u32| {
        let mut cycle = WeatherCycle::new("clear", seed);
        let mut visits = vec![cycle.current()];
        let mut since_change = 0.0;
        for _ in 0..(3 * 60 * 60) {
            since_change += 1.0;
            if let Some(next) = cycle.advance(1.0) {
                let previous = *visits.last().unwrap();
                assert!(
                    neighbours(previous).contains(&next),
                    "{previous} jumped to {next}"
                );
                assert!(since_change >= WeatherCycle::DWELL.0);
                assert!(Weather::by_name(next).is_some());
                visits.push(next);
                since_change = 0.0;
            }
        }
        visits
    };
    let first = run(7);
    for name in Weather::names() {
        assert!(first.contains(&name), "never reached {name}: {first:?}");
    }
    assert_eq!(run(7), first);
    assert_ne!(run(11), first);
}

#[test]
fn cycles_fall_back_to_fair_and_keep_their_dwell_and_transition() {
    assert_eq!(WeatherCycle::new("hail", 1).current(), "fair");
    assert_eq!(WeatherCycle::new("Clear", 1).current(), "fair");
    assert_eq!(WeatherCycle::new("cirrus", 1).current(), "cirrus");
    assert_eq!(WeatherCycle::DWELL, (120.0, 300.0));
    assert_eq!(WeatherCycle::TRANSITION, 90.0);

    let mut cycle = WeatherCycle::new("fair", 1);
    assert_eq!(cycle.advance(0.0), None);
    assert_eq!(cycle.advance(WeatherCycle::DWELL.0 - 0.5), None);
    assert!(cycle.advance(WeatherCycle::DWELL.1 + 0.5).is_some());
}

#[test]
fn cloud_samples_and_scale_default_and_are_validated() {
    assert_eq!(WeatherSettings::default().cloud_samples, 56);
    assert_eq!(WeatherSettings::default().cloud_scale, 0.5);
    for ok in [8_u32, 56, 256] {
        let s = WeatherSettings {
            cloud_samples: ok,
            ..WeatherSettings::default()
        };
        s.validate().unwrap();
    }
    for bad in [0_u32, 7, 257, u32::MAX] {
        let s = WeatherSettings {
            cloud_samples: bad,
            ..WeatherSettings::default()
        };
        refused(s.validate(), "cloud_samples");
    }
    for ok in [0.25_f32, 0.5, 1.0] {
        let s = WeatherSettings {
            cloud_scale: ok,
            ..WeatherSettings::default()
        };
        s.validate().unwrap();
    }
    for bad in [0.0_f32, 0.24, 1.01, f32::NAN, f32::INFINITY] {
        let s = WeatherSettings {
            cloud_scale: bad,
            ..WeatherSettings::default()
        };
        refused(s.validate(), "cloud_scale");
    }
    assert!(WeatherSettings::default().cloud_dither);
    assert_eq!(WeatherSettings::default().cloud_noise_size, 1);
    for ok in [1_u32, 8, 16] {
        WeatherSettings {
            cloud_noise_size: ok,
            ..WeatherSettings::default()
        }
        .validate()
        .unwrap();
    }
    for bad in [0_u32, 17, u32::MAX] {
        let s = WeatherSettings {
            cloud_noise_size: bad,
            ..WeatherSettings::default()
        };
        refused(s.validate(), "cloud_noise_size");
    }
}
