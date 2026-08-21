//! The PRNG and the level distribution it drives.

use hnsw::rng::{assign_level, default_ml, splitmix64, MAX_LEVEL};
use hnsw::Rng;

#[test]
fn the_same_seed_gives_the_same_stream() {
    let mut a = Rng::new(0xDEAD_BEEF);
    let mut b = Rng::new(0xDEAD_BEEF);
    for _ in 0..1000 {
        assert_eq!(a.next_u64(), b.next_u64());
    }
}

#[test]
fn different_seeds_give_different_streams() {
    let mut a = Rng::new(1);
    let mut b = Rng::new(2);
    let first: Vec<u64> = (0..64).map(|_| a.next_u64()).collect();
    let second: Vec<u64> = (0..64).map(|_| b.next_u64()).collect();
    assert_ne!(first, second);
}

#[test]
fn seed_zero_is_not_degenerate() {
    // xoshiro has an all-zero fixed point; SplitMix64 seeding is what keeps
    // seed 0 from landing on it.
    let mut rng = Rng::new(0);
    let draws: Vec<u64> = (0..16).map(|_| rng.next_u64()).collect();
    assert!(draws.iter().any(|&x| x != 0));
    assert_eq!(
        draws.iter().collect::<std::collections::HashSet<_>>().len(),
        16
    );
}

#[test]
fn state_round_trips_and_resumes_the_stream() {
    let mut rng = Rng::new(4242);
    for _ in 0..37 {
        rng.next_u64();
    }
    let mut resumed = Rng::from_state(rng.state());
    for _ in 0..100 {
        assert_eq!(rng.next_u64(), resumed.next_u64());
    }
}

#[test]
fn an_all_zero_state_is_repaired_rather_than_accepted() {
    let mut rng = Rng::from_state([0; 4]);
    let draws: Vec<u64> = (0..8).map(|_| rng.next_u64()).collect();
    assert!(draws.iter().any(|&x| x != 0));
}

#[test]
fn uniform_draws_stay_inside_the_unit_interval() {
    let mut rng = Rng::new(5);
    for _ in 0..100_000 {
        let u = rng.next_f64();
        assert!((0.0..1.0).contains(&u), "{u} escaped [0, 1)");
        let open = rng.open_unit();
        assert!(open > 0.0 && open <= 1.0, "{open} escaped (0, 1]");
    }
}

#[test]
fn uniform_draws_have_the_right_mean_and_spread() {
    let mut rng = Rng::new(6);
    const N: usize = 200_000;
    let mut sum = 0.0;
    let mut buckets = [0usize; 10];
    for _ in 0..N {
        let u = rng.next_f64();
        sum += u;
        buckets[(u * 10.0) as usize] += 1;
    }
    let mean = sum / N as f64;
    assert!((mean - 0.5).abs() < 0.005, "mean was {mean}");
    // Every decile should hold about a tenth of the draws.
    for (i, &count) in buckets.iter().enumerate() {
        let share = count as f64 / N as f64;
        assert!((share - 0.1).abs() < 0.01, "decile {i} held {share}");
    }
}

#[test]
fn gaussian_draws_have_unit_variance_and_zero_mean() {
    let mut rng = Rng::new(8);
    const N: usize = 200_000;
    let (mut sum, mut sum_sq) = (0.0, 0.0);
    for _ in 0..N {
        let x = rng.gaussian();
        sum += x;
        sum_sq += x * x;
    }
    let mean = sum / N as f64;
    let variance = sum_sq / N as f64 - mean * mean;
    assert!(mean.abs() < 0.01, "mean was {mean}");
    assert!((variance - 1.0).abs() < 0.02, "variance was {variance}");
}

#[test]
fn below_stays_in_range_and_covers_it() {
    let mut rng = Rng::new(9);
    let mut seen = [false; 7];
    for _ in 0..1000 {
        let v = rng.below(7);
        assert!(v < 7);
        seen[v] = true;
    }
    assert!(seen.iter().all(|&s| s));
    assert_eq!(rng.below(0), 0);
    assert_eq!(rng.below(1), 0);
}

#[test]
fn shuffle_is_a_permutation_and_depends_on_the_seed() {
    let mut items: Vec<u32> = (0..64).collect();
    Rng::new(3).shuffle(&mut items);
    let mut sorted = items.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..64).collect::<Vec<u32>>());
    assert_ne!(
        items, sorted,
        "a shuffle that changes nothing is suspicious"
    );

    let mut other: Vec<u32> = (0..64).collect();
    Rng::new(4).shuffle(&mut other);
    assert_ne!(items, other);
}

#[test]
fn splitmix64_is_a_deterministic_bijective_looking_mix() {
    let mut state = 0u64;
    let first: Vec<u64> = (0..8).map(|_| splitmix64(&mut state)).collect();
    let mut state = 0u64;
    let again: Vec<u64> = (0..8).map(|_| splitmix64(&mut state)).collect();
    assert_eq!(first, again);
    assert_eq!(
        first.iter().collect::<std::collections::HashSet<_>>().len(),
        8
    );
}

#[test]
fn layer_populations_decay_geometrically_by_a_factor_of_m() {
    // The defining property of the level distribution: P(level >= i) should
    // be exp(-i / ml), so with ml = 1/ln(M) each layer holds 1/M of the one
    // below. Get this wrong and the descent either has nothing to descend
    // through or spends all its time in near-empty upper layers.
    const N: usize = 400_000;
    for m in [4usize, 16, 32] {
        let ml = default_ml(m);
        let mut rng = Rng::new(0xABCD);
        let mut counts = vec![0usize; MAX_LEVEL + 2];
        for _ in 0..N {
            counts[assign_level(&mut rng, ml)] += 1;
        }

        // Convert the histogram of top levels into layer populations.
        let mut populations = Vec::new();
        let mut running = 0usize;
        for level in (0..counts.len()).rev() {
            running += counts[level];
            populations.push(running);
        }
        populations.reverse();

        assert_eq!(populations[0], N, "layer 0 must hold everything");
        // Compare consecutive layers while the sample is still large enough
        // for the ratio to be meaningful. How many layers that covers falls
        // as M rises, because the layers thin out by a factor of M each time:
        // at M = 4 the fifth layer still holds ~1500 of the 400000 draws, at
        // M = 32 the third holds ~400.
        let mut checked = 0;
        for level in 0..populations.len() - 1 {
            if populations[level + 1] < 1200 {
                break;
            }
            let ratio = populations[level] as f64 / populations[level + 1] as f64;
            assert!(
                (ratio - m as f64).abs() < m as f64 * 0.15,
                "M={m}: layer {level} to {} shrank by {ratio:.2}, expected about {m}",
                level + 1
            );
            checked += 1;
        }
        let needed = if m <= 16 { 2 } else { 1 };
        assert!(
            checked >= needed,
            "M={m}: only {checked} layer ratios had enough samples to test"
        );
    }
}

#[test]
fn level_zero_is_by_far_the_most_common_draw() {
    let ml = default_ml(16);
    let mut rng = Rng::new(77);
    let mut zeros = 0usize;
    const N: usize = 100_000;
    for _ in 0..N {
        if assign_level(&mut rng, ml) == 0 {
            zeros += 1;
        }
    }
    let share = zeros as f64 / N as f64;
    // P(level == 0) = 1 - exp(-1/ml) = 1 - 1/M = 15/16.
    assert!((share - 15.0 / 16.0).abs() < 0.01, "share was {share}");
}

#[test]
fn assign_level_is_capped_and_survives_hostile_parameters() {
    let mut rng = Rng::new(1);
    for _ in 0..10_000 {
        assert!(assign_level(&mut rng, default_ml(16)) <= MAX_LEVEL);
        // A wildly large ml would otherwise allocate an unbounded layer list.
        assert!(assign_level(&mut rng, 1e9) <= MAX_LEVEL);
    }
    assert_eq!(assign_level(&mut rng, 0.0), 0);
    assert_eq!(assign_level(&mut rng, -1.0), 0);
    assert_eq!(assign_level(&mut rng, f64::NAN), 0);
}

#[test]
fn default_ml_is_one_over_ln_m() {
    assert!((default_ml(16) - 1.0 / 16.0f64.ln()).abs() < 1e-12);
    assert!(default_ml(1).is_finite(), "m = 1 must not divide by zero");
}
