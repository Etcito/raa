use super::*;
/// Parse and validate a section 67 trace program.
pub fn parse(bytes: &[u8]) -> Result<TraceProgram, TraceError> {
    let word_count = bytes.len() / 4;
    if !bytes.len().is_multiple_of(4) {
        refuse!(
            FORMAT,
            "bytes",
            "{} bytes is not a whole number of 32-bit words",
            bytes.len()
        );
    }
    let header: Vec<u32> = bytes.chunks_exact(4).take(HEADER_WORDS).map(word).collect();
    if header.len() != HEADER_WORDS {
        refuse!(FORMAT, "header", "the program ends inside the header");
    }
    if header[0] != MAGIC {
        refuse!(
            FORMAT,
            "header.magic",
            "the program does not start with SDTP"
        );
    }
    if header[1] != VERSION && header[1] != 1 {
        refuse!(
            VERSION_ERROR,
            "header.version",
            "version {} (this reader accepts {VERSION})",
            header[1]
        );
    }
    let expected = total_words(&header)
        .ok_or_else(|| TraceError::at(LIMITS, "header", "the section lengths overflow"))?;
    if expected != word_count as u64 {
        let detail = if expected > word_count as u64 {
            "the program is truncated"
        } else {
            "the program has extra words"
        };
        refuse!(
            FORMAT,
            "header",
            "the header names {expected} words, the program has {word_count}: {detail}"
        );
    }

    let body: Vec<u32> = bytes
        .get(HEADER_WORDS * 4..)
        .unwrap_or(&[])
        .chunks_exact(4)
        .map(word)
        .collect();
    let (source_words, rest) = cut(&body, header[2] as usize * SOURCE_WORDS, "sources")?;
    let (code, rest) = cut(rest, header[4] as usize, "code")?;
    let (data_words, rest) = cut(rest, header[5] as usize, "data")?;
    let (material_words, rest) = cut(rest, header[6] as usize * MATERIAL_WORDS, "materials")?;
    let (instance_words, bend_words) = cut(rest, header[3] as usize * INSTANCE_WORDS, "instances")?;
    let mut instances: Vec<Instance> = instance_words
        .chunks_exact(INSTANCE_WORDS)
        .map(read_instance)
        .collect();
    if header[7] as usize != instances.iter().filter(|i| i.deform_kind == 1).count() {
        refuse!(
            FORMAT,
            "bends",
            "the bend records do not match the bent instances"
        );
    }
    let mut records = bend_words.chunks_exact(16);
    instances
        .iter_mut()
        .filter(|i| i.deform_kind == 1)
        .for_each(|i| i.bend = records.next().map(read_bend));

    let program = TraceProgram {
        sources: source_words
            .chunks_exact(SOURCE_WORDS)
            .map(read_source)
            .collect(),
        code: code.to_vec(),
        data: data_words.iter().map(|w| f32::from_bits(*w)).collect(),
        materials: material_words
            .chunks_exact(MATERIAL_WORDS)
            .map(read_material)
            .collect(),
        instances,
    };
    if header[1] == 1
        && program
            .code
            .iter()
            .any(|w| w & OPCODE_MASK == opcode::SAMPLED)
    {
        refuse!(
            VERSION_ERROR,
            "code",
            "sampled leaves require SDTP version 2"
        );
    }
    validate(&program)?;
    Ok(program)
}
