use super::*;
pub(crate) fn validate_instances(program: &TraceProgram) -> Result<(), TraceError> {
    for (index, instance) in program.instances.iter().enumerate() {
        let path = format!("instance[{index}]");
        if instance.source as usize >= program.sources.len() {
            refuse!(
                REFERENCE,
                format!("{path}.source"),
                "source {} is outside the {} sources",
                instance.source,
                program.sources.len()
            );
        }
        let material = instance.material;
        let material_ok =
            material == -1 || (material >= 0 && (material as usize) < program.materials.len());
        if !material_ok {
            refuse!(
                REFERENCE,
                format!("{path}.material"),
                "material override {material} is not -1 or a material index"
            );
        }
        let finite = instance
            .world_from_local
            .iter()
            .chain(instance.local_from_world.iter())
            .flatten()
            .chain(
                [
                    instance.dist_scale,
                    instance.bend_angle,
                    instance.bend_length,
                ]
                .iter(),
            )
            .all(|v| v.is_finite());
        if !finite {
            refuse!(
                NOT_FINITE,
                &path,
                "a matrix, distance scale or bend value is not finite"
            );
        }
        if instance.dist_scale <= 0.0 {
            refuse!(
                FORMAT,
                format!("{path}.distScale"),
                "the distance scale must be positive"
            );
        }
        let (along, toward) = (instance.axes & 7, (instance.axes >> 3) & 7);
        match instance.deform_kind {
            0 if instance.bend.is_none() => {}
            0 => refuse!(
                FORMAT,
                format!("{path}.bend"),
                "a bend record has no bent instance"
            ),
            1 if instance.bend_length > 0.0 => {
                if along > 5 || toward == 6 || instance.axes >> 6 != 0 {
                    refuse!(
                        FORMAT,
                        format!("{path}.bendAxes"),
                        "the bend frame is incomplete: along must be one of the six signed axes, toward one of them or 7 (no aim)"
                    );
                }
                let Some(b) = instance.bend else {
                    refuse!(FORMAT, format!("{path}.bend"), "the bend record is missing");
                };
                if !(0.0..=2.0).contains(&b[0])
                    || b[0].fract() != 0.0
                    || b.iter().any(|v| !v.is_finite())
                    || b[1] == 0.0
                    || b[4] <= b[3]
                {
                    refuse!(
                        FORMAT,
                        format!("{path}.bend"),
                        "the bend record is incomplete"
                    );
                }
            }
            1 => refuse!(
                FORMAT,
                format!("{path}.bendLength"),
                "the bend length must be positive"
            ),
            kind => refuse!(
                FORMAT,
                format!("{path}.deformKind"),
                "unknown deform kind {kind}"
            ),
        }
    }
    Ok(())
}
