//! Distance kernels against values worked out by hand.

use hnsw::distance::{dot, norm, normalize, squared_l2, Cosine, Euclidean, InnerProduct};
use hnsw::{Distance, Metric};

/// A deliberately naive reference: one accumulator, left to right. The
/// chunked kernels must agree with it to within float rounding.
fn reference_dot(a: &[f32], b: &[f32]) -> f32 {
    let mut total = 0.0f32;
    for i in 0..a.len().min(b.len()) {
        total += a[i] * b[i];
    }
    total
}

fn reference_sql2(a: &[f32], b: &[f32]) -> f32 {
    let mut total = 0.0f32;
    for i in 0..a.len().min(b.len()) {
        let d = a[i] - b[i];
        total += d * d;
    }
    total
}

#[test]
fn dot_matches_hand_computed_value() {
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert_eq!(dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
}

#[test]
fn dot_of_empty_slices_is_zero() {
    assert_eq!(dot(&[], &[]), 0.0);
}

#[test]
fn dot_is_symmetric() {
    let a = [0.5, -1.5, 2.25, 7.0, -0.125];
    let b = [3.0, 1.0, -2.0, 0.5, 4.0];
    assert_eq!(dot(&a, &b), dot(&b, &a));
}

#[test]
fn squared_l2_matches_hand_computed_value() {
    // (1-4)^2 + (2-6)^2 = 9 + 16 = 25
    assert_eq!(squared_l2(&[1.0, 2.0], &[4.0, 6.0]), 25.0);
}

#[test]
fn squared_l2_of_identical_vectors_is_zero() {
    let v = [3.5, -2.0, 0.0, 11.25, 6.5, -4.0, 1.0, 9.0, 2.5];
    assert_eq!(squared_l2(&v, &v), 0.0);
}

#[test]
fn norm_matches_hand_computed_value() {
    // 3-4-5 triangle.
    assert_eq!(norm(&[3.0, 4.0]), 5.0);
}

#[test]
fn chunked_kernels_agree_with_the_naive_reference_at_every_length() {
    // Lengths either side of the eight-lane boundary exercise the main loop,
    // the remainder loop, and the case where there is no main loop at all.
    let mut rng = hnsw::Rng::new(7);
    for len in 0..40usize {
        let a: Vec<f32> = (0..len).map(|_| rng.range(-3.0, 3.0)).collect();
        let b: Vec<f32> = (0..len).map(|_| rng.range(-3.0, 3.0)).collect();

        let (got, want) = (dot(&a, &b), reference_dot(&a, &b));
        assert!(
            (got - want).abs() <= 1e-4 * want.abs().max(1.0),
            "dot mismatch at len {len}: {got} vs {want}"
        );

        let (got, want) = (squared_l2(&a, &b), reference_sql2(&a, &b));
        assert!(
            (got - want).abs() <= 1e-4 * want.abs().max(1.0),
            "squared_l2 mismatch at len {len}: {got} vs {want}"
        );
    }
}

#[test]
fn normalize_gives_unit_length() {
    let mut v = [3.0, 4.0];
    assert!(normalize(&mut v));
    assert!((norm(&v) - 1.0).abs() < 1e-6);
    assert!((v[0] - 0.6).abs() < 1e-6);
    assert!((v[1] - 0.8).abs() < 1e-6);
}

#[test]
fn normalize_refuses_the_zero_vector_and_leaves_it_alone() {
    let mut v = [0.0, 0.0, 0.0];
    assert!(!normalize(&mut v));
    assert_eq!(v, [0.0, 0.0, 0.0]);
}

#[test]
fn cosine_of_identical_directions_is_zero() {
    let d = Cosine.distance(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]);
    assert!(d.abs() < 1e-6, "expected 0, got {d}");
}

#[test]
fn cosine_of_orthogonal_vectors_is_one() {
    let d = Cosine.distance(&[1.0, 0.0], &[0.0, 5.0]);
    assert!((d - 1.0).abs() < 1e-6, "expected 1, got {d}");
}

#[test]
fn cosine_of_opposite_vectors_is_two() {
    let d = Cosine.distance(&[1.0, 1.0], &[-2.0, -2.0]);
    assert!((d - 2.0).abs() < 1e-6, "expected 2, got {d}");
}

#[test]
fn cosine_ignores_magnitude() {
    let short = Cosine.distance(&[1.0, 2.0], &[3.0, 1.0]);
    let long = Cosine.distance(&[100.0, 200.0], &[3.0, 1.0]);
    assert!((short - long).abs() < 1e-6);
}

#[test]
fn cosine_against_the_zero_vector_is_one_rather_than_nan() {
    let d = Cosine.distance(&[0.0, 0.0, 0.0], &[1.0, 2.0, 3.0]);
    assert!(d.is_finite(), "cosine produced {d}");
    assert!((d - 1.0).abs() < 1e-6, "expected 1, got {d}");
    assert!(Cosine.distance(&[0.0, 0.0], &[0.0, 0.0]).is_finite());
}

#[test]
fn preprocessed_cosine_equals_plain_cosine() {
    // The whole point of normalising at insert time: a bare dot product on
    // preprocessed vectors must give the same answer as the full formula.
    let mut rng = hnsw::Rng::new(11);
    for _ in 0..50 {
        let a: Vec<f32> = (0..16).map(|_| rng.range(-2.0, 2.0)).collect();
        let b: Vec<f32> = (0..16).map(|_| rng.range(-2.0, 2.0)).collect();
        let (mut pa, mut pb) = (a.clone(), b.clone());
        Cosine.preprocess(&mut pa);
        Cosine.preprocess(&mut pb);

        let plain = Cosine.distance(&a, &b);
        let fast = Cosine.distance_preprocessed(&pa, &pb);
        assert!(
            (plain - fast).abs() < 1e-5,
            "preprocessed cosine {fast} differs from plain {plain}"
        );
    }
}

#[test]
fn dot_distance_is_the_negated_inner_product() {
    assert_eq!(
        InnerProduct.distance(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]),
        -32.0
    );
}

#[test]
fn euclidean_distance_is_the_squared_form() {
    assert_eq!(Euclidean.distance(&[0.0, 0.0], &[3.0, 4.0]), 25.0);
}

#[test]
fn metric_enum_dispatches_to_the_unit_structs() {
    let a = [0.3, -1.2, 4.0, 0.5];
    let b = [1.0, 2.0, -0.5, 3.0];
    assert_eq!(Metric::Cosine.distance(&a, &b), Cosine.distance(&a, &b));
    assert_eq!(Metric::Dot.distance(&a, &b), InnerProduct.distance(&a, &b));
    assert_eq!(Metric::L2.distance(&a, &b), Euclidean.distance(&a, &b));
}

#[test]
fn metric_preprocess_only_normalises_for_cosine() {
    let original = [3.0f32, 4.0];
    for metric in Metric::ALL {
        let mut v = original;
        metric.preprocess(&mut v);
        if metric == Metric::Cosine {
            assert!((norm(&v) - 1.0).abs() < 1e-6);
        } else {
            assert_eq!(v, original, "{metric} must not touch the vector");
        }
    }
}

#[test]
fn metric_tags_and_names_round_trip() {
    for metric in Metric::ALL {
        assert_eq!(Metric::from_tag(metric.tag()), Some(metric));
        assert_eq!(Metric::parse(metric.name()), Some(metric));
        assert_eq!(metric.to_string(), metric.name());
    }
    assert_eq!(Metric::from_tag(200), None);
    assert_eq!(Metric::parse("hamming"), None);
    assert_eq!(Metric::parse("  COSINE "), Some(Metric::Cosine));
    assert_eq!(Metric::parse("euclidean"), Some(Metric::L2));
}

#[test]
fn every_metric_reports_zero_distance_to_a_point_from_itself() {
    // True for cosine and L2 by definition. Not true for the inner product,
    // which is exactly the caveat documented on `Metric::Dot` — assert the
    // real behaviour rather than a comfortable one.
    let v = [0.6f32, 0.8, 0.0, 0.0];
    assert!(Metric::Cosine.distance(&v, &v).abs() < 1e-6);
    assert_eq!(Metric::L2.distance(&v, &v), 0.0);
    assert_eq!(Metric::Dot.distance(&v, &v), -1.0);
}
