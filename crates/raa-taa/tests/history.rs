//! Contract `taa-history`: the CPU parity state clamps its size, flips on commit, keeps its
//! write index on invalidate and reports usability only with a valid frame history.

use raa_taa::HistoryState;

#[test]
fn new_is_invalid_at_the_clamped_size() {
    let state = HistoryState::new((64, 32));
    assert!(!state.is_valid());
    assert_eq!(state.accumulated_frames(), 0);
    assert_eq!(state.write_index(), 0);
    assert_eq!(state.read_index(), 1);
    assert_eq!(state.size(), (64, 32));
}

#[test]
fn zero_size_clamps_each_axis_to_one() {
    assert_eq!(HistoryState::new((0, 0)).size(), (1, 1));
}

#[test]
fn commit_flips_parity_and_accumulates() {
    let mut state = HistoryState::new((8, 8));
    state.commit();
    assert!(state.is_valid());
    assert_eq!(state.accumulated_frames(), 1);
    assert_eq!(state.write_index(), 1);
    assert_eq!(state.read_index(), 0);

    state.commit();
    assert_eq!(state.accumulated_frames(), 2);
    assert_eq!(state.write_index(), 0);
    assert_eq!(state.read_index(), 1);
}

#[test]
fn invalidate_clears_frames_and_keeps_the_write_index() {
    let mut state = HistoryState::new((8, 8));
    state.commit();
    state.invalidate();
    assert_eq!(state.accumulated_frames(), 0);
    assert!(!state.is_valid());
    assert_eq!(state.write_index(), 1);
}

#[test]
fn resize_reports_and_stores_only_a_new_clamped_size() {
    let mut state = HistoryState::new((8, 8));
    state.commit();

    assert!(!state.resize((8, 8)));
    assert_eq!(state.accumulated_frames(), 1);
    assert_eq!(state.size(), (8, 8));

    assert!(state.resize((16, 4)));
    assert_eq!(state.size(), (16, 4));
    assert_eq!(state.accumulated_frames(), 0);
    assert!(!state.is_valid());
    assert_eq!(state.write_index(), 1);

    assert!(state.resize((0, 5)));
    assert_eq!(state.size(), (1, 5));
}

#[test]
fn usable_needs_this_history_and_the_frame_history() {
    let mut state = HistoryState::new((8, 8));
    state.commit();
    assert!(!state.usable(false));
    assert_eq!(state.usable(true), state.is_valid());
}
