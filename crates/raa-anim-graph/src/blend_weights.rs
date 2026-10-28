//! Runtime blend-space weights (S19): neighbouring 1D points and barycentric
//! weights over the compiled 2D triangles. Outside the hull the query projects
//! onto the nearest hull edge. Contributions are non-negative, sum to one and
//! stay in ascending point order.

use crate::error::{GraphError, codes};
use crate::model::BlendSpace;
use crate::triangulation::Triangulation;

type P = [f32; 2];

/// A blend point's document index and its non-negative weight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlendWeight {
    pub point: usize,
    pub weight: f32,
}

fn bad(code: &'static str, message: &str) -> GraphError {
    GraphError::new(code, "/blendSpaces", message)
}

fn coord(space: &BlendSpace, point: usize, axis: usize) -> Result<f32, GraphError> {
    space
        .points
        .get(point)
        .and_then(|p| p.at.get(axis))
        .map(|value| value.rt)
        .filter(|value| value.is_finite())
        .ok_or_else(|| bad(codes::BLEND_SPACE, "blend point coordinates must be finite"))
}

fn single(point: usize) -> Vec<BlendWeight> {
    vec![BlendWeight { point, weight: 1.0 }]
}

fn normalized(mut weights: Vec<BlendWeight>) -> Vec<BlendWeight> {
    weights.retain(|w| w.weight.is_finite() && w.weight > 0.0);
    let sum = weights.iter().fold(0.0, |sum, w| sum + w.weight);
    if sum.is_finite() && sum > 0.0 {
        for weight in &mut weights { weight.weight /= sum; }
    }
    weights.sort_by_key(|w| w.point);
    weights
}

/// Two-neighbour interpolation, clamped outside the first and last point.
pub fn evaluate_1d(space: &BlendSpace, value: f32) -> Result<Vec<BlendWeight>, GraphError> {
    if !value.is_finite() { return Err(bad(codes::PARAM, "blend-space value is not finite")); }
    let count = space.points.len();
    if count == 0 { return Err(bad(codes::BLEND_SPACE, "blend space has no points")); }
    for i in 0..count { coord(space, i, 0)?; }
    let x = |i: usize| space.points[i].at[0].rt;
    let mut order: Vec<usize> = (0..count).collect();
    order.sort_by(|&a, &b| x(a).total_cmp(&x(b)).then(a.cmp(&b)));
    for &i in &order {
        if x(i) == value { return Ok(single(i)); }
    }
    let (first, last) = (order[0], order[count - 1]);
    if value < x(first) { return Ok(single(first)); }
    if value > x(last) { return Ok(single(last)); }
    let mut pair = (first, last);
    for edge in order.windows(2) {
        let (a, b) = (edge[0], edge[1]);
        if x(a) < value && value < x(b) { pair = (a, b); break; }
    }
    let (a, b) = pair;
    if !(x(b) > x(a)) { return Ok(single(first)); }
    let u = ((value - x(a)) / (x(b) - x(a))).clamp(0.0, 1.0);
    Ok(normalized(vec![
        BlendWeight { point: a, weight: 1.0 - u },
        BlendWeight { point: b, weight: u },
    ]))
}

fn orient(a: P, b: P, c: P) -> f32 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

fn triangle_weights(triangle: [usize; 3], v: [P; 3], query: P) -> Vec<BlendWeight> {
    let total = orient(v[0], v[1], v[2]);
    if !(total > 0.0) || !total.is_finite() { return Vec::new(); }
    let edges = [
        orient(v[1], v[2], query),
        orient(v[2], v[0], query),
        orient(v[0], v[1], query),
    ];
    if edges.iter().any(|&edge| !(edge >= 0.0)) { return Vec::new(); }
    let weights = triangle.iter().zip(edges).map(|(&point, edge)| BlendWeight {
        point,
        weight: (edge / total).clamp(0.0, 1.0),
    });
    normalized(weights.collect())
}

/// Barycentric weights in the containing triangle, else the projection onto
/// the nearest hull edge; equal distances break on the sorted endpoint pair.
pub fn evaluate_2d(
    space: &BlendSpace,
    geometry: &Triangulation,
    query: P,
) -> Result<Vec<BlendWeight>, GraphError> {
    if !(query[0].is_finite() && query[1].is_finite()) {
        return Err(bad(codes::PARAM, "blend-space coordinates are not finite"));
    }
    if space.points.is_empty() { return Err(bad(codes::BLEND_SPACE, "blend space has no points")); }
    let at = |point: usize| -> Result<P, GraphError> {
        Ok([coord(space, point, 0)?, coord(space, point, 1)?])
    };
    for point in 0..space.points.len() {
        if at(point)? == query { return Ok(single(point)); }
    }
    for &triangle in &geometry.triangles {
        let v = [at(triangle[0])?, at(triangle[1])?, at(triangle[2])?];
        let weights = triangle_weights(triangle, v, query);
        if !weights.is_empty() { return Ok(weights); }
    }
    let mut best: Option<(f32, [usize; 2], [usize; 2], f32)> = None;
    for &edge in &geometry.hull {
        let (a, b) = (at(edge[0])?, at(edge[1])?);
        let d = [b[0] - a[0], b[1] - a[1]];
        let length2 = d[0] * d[0] + d[1] * d[1];
        let t = if length2.is_finite() && length2 > 0.0 {
            (((query[0] - a[0]) * d[0] + (query[1] - a[1]) * d[1]) / length2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let closest = [a[0] + d[0] * t, a[1] + d[1] * t];
        let distance = (closest[0] - query[0]) * (closest[0] - query[0])
            + (closest[1] - query[1]) * (closest[1] - query[1]);
        if !distance.is_finite() { continue; }
        let key = if edge[0] < edge[1] { edge } else { [edge[1], edge[0]] };
        let better = best.map_or(true, |(d, k, _, _)| distance < d || (distance == d && key < k));
        if better { best = Some((distance, key, edge, t)); }
    }
    let Some(&first) = geometry.hull.first() else {
        return Err(bad(codes::BLEND_SPACE, "2D hull has no edges"));
    };
    let (_, _, edge, t) = best.unwrap_or((0.0, first, first, 0.0));
    if t <= 0.0 { return Ok(single(edge[0])); }
    if t >= 1.0 { return Ok(single(edge[1])); }
    Ok(normalized(vec![
        BlendWeight { point: edge[0], weight: 1.0 - t },
        BlendWeight { point: edge[1], weight: t },
    ]))
}

/// Dispatch on the declared arity; a 2D space needs its triangulation.
pub fn evaluate(
    space: &BlendSpace,
    geometry: Option<&Triangulation>,
    coordinates: &[f32],
) -> Result<Vec<BlendWeight>, GraphError> {
    if coordinates.len() != space.params.len() {
        return Err(bad(codes::PARAM, "blend-space coordinate count"));
    }
    match coordinates {
        [value] => evaluate_1d(space, *value),
        [x, y] => geometry.map_or_else(
            || Err(bad(codes::BLEND_SPACE, "2D blend space needs a triangulation")),
            |geometry| evaluate_2d(space, geometry, [*x, *y]),
        ),
        _ => Err(bad(codes::BLEND_SPACE, "a blend space has one or two parameters")),
    }
}
