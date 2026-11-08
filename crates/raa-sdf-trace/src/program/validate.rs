use super::*;
pub(super) fn validate(program: &TraceProgram) -> Result<(), TraceError> {
    for (index, material) in program.materials.iter().enumerate() {
        let values = [
            material.color[0],
            material.color[1],
            material.color[2],
            material.roughness,
            material.metallic,
        ];
        if values.iter().any(|v| !v.is_finite()) {
            refuse!(
                NOT_FINITE,
                format!("material[{index}]"),
                "a material value is not finite"
            );
        }
    }
    validate_sources(program)?;
    validate_code(program)?;
    validate_instances(program)
}

pub(super) fn validate_sources(program: &TraceProgram) -> Result<(), TraceError> {
    for (index, source) in program.sources.iter().enumerate() {
        let path = format!("source[{index}]");
        let code_end = source
            .code_start
            .checked_add(source.code_len)
            .ok_or_else(|| {
                TraceError::at(
                    LIMITS,
                    format!("{path}.code"),
                    "the instruction span overflows",
                )
            })?;
        if code_end as usize > program.code.len() {
            refuse!(
                SOURCE,
                format!("{path}.code"),
                "words {}..{code_end} are outside the {} code words",
                source.code_start,
                program.code.len()
            );
        }
        let data_end = source
            .data_start
            .checked_add(source.data_len)
            .ok_or_else(|| {
                TraceError::at(LIMITS, format!("{path}.data"), "the data span overflows")
            })?;
        if data_end as usize > program.data.len() {
            refuse!(
                SOURCE,
                format!("{path}.data"),
                "words {}..{data_end} are outside the {} data words",
                source.data_start,
                program.data.len()
            );
        }
        if source
            .min
            .iter()
            .chain(source.max.iter())
            .any(|v| !v.is_finite())
        {
            refuse!(
                NOT_FINITE,
                format!("{path}.bounds"),
                "a bound coordinate is not finite"
            );
        }
        if (0..3).any(|axis| source.min[axis] > source.max[axis]) {
            refuse!(
                SOURCE,
                format!("{path}.bounds"),
                "a minimum bound exceeds its maximum"
            );
        }
        if !source.lipschitz.is_finite() {
            refuse!(
                NOT_FINITE,
                format!("{path}.lipschitz"),
                "the Lipschitz bound is not finite"
            );
        }
        if source.lipschitz <= 0.0 {
            refuse!(
                SOURCE,
                format!("{path}.lipschitz"),
                "the Lipschitz bound must be positive"
            );
        }
    }
    Ok(())
}

pub(super) fn validate_code(program: &TraceProgram) -> Result<(), TraceError> {
    for (source_index, source) in program.sources.iter().enumerate() {
        let (code, data) = (source.code(program), source.data(program));
        let mut depth: usize = 0;
        for (index, instruction) in code.iter().enumerate() {
            let path = format!("source[{source_index}].code[{index}]");
            let operation = instruction & OPCODE_MASK;
            let offset = (instruction >> DATA_SHIFT) as usize;
            let needed = opcode::data_words(operation).ok_or_else(|| {
                TraceError::at(OPCODE, &path, format!("unknown opcode {operation}"))
            })?;
            let end = offset.checked_add(needed).ok_or_else(|| {
                TraceError::at(LIMITS, &path, "the instruction data offset overflows")
            })?;
            if end > data.len() {
                refuse!(
                    DATA,
                    &path,
                    "data words {offset}..{end} are outside the source's {} data words",
                    data.len()
                );
            }
            if data[offset..end].iter().any(|v| !v.is_finite()) {
                refuse!(NOT_FINITE, &path, "an instruction data word is not finite");
            }
            if opcode::is_primitive(operation) {
                let material = data[offset + 13];
                if material < 0.0
                    || material.fract() != 0.0
                    || material as u64 >= program.materials.len() as u64
                {
                    refuse!(
                        REFERENCE,
                        &path,
                        "primitive material {material} is not a material index"
                    );
                }
                if operation == opcode::SAMPLED {
                    let bound = crate::sampled::validate(program, &data[offset..end], &path)?;
                    if source.lipschitz * 1.00001 < bound {
                        refuse!(
                            DATA,
                            &path,
                            "source step bound is too small for the sampled leaf"
                        );
                    }
                }
                depth += 1;
                if depth > 64 {
                    refuse!(STACK, &path, "program exceeds 64 stack values");
                }
            } else {
                if depth < 2 {
                    refuse!(
                        STACK,
                        &path,
                        "opcode {operation} needs two operands, the stack has {depth}"
                    );
                }
                depth -= 1;
            }
        }
        if depth != 1 {
            refuse!(
                STACK,
                format!("source[{source_index}].code"),
                "the program ends with {depth} values on the stack, expected 1"
            );
        }
    }
    Ok(())
}
