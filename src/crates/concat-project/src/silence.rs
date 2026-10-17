// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer and Concat contributors

//! Finding the pauses in a recording, so they can be cut.
//!
//! The input is the waveform the timeline already holds: the loudest
//! sample of each bucket, a thousand buckets a second. Nothing is decoded
//! again. The output is the stretches of the source worth taking out, in
//! source seconds, each already shrunk by the breath that is left on
//! either side of it.
//!
//! The threshold is not a fixed number of decibels. A phone recording in
//! a shop and a lapel microphone in a quiet room have floors twenty
//! decibels apart, so the threshold sits a margin above this recording's
//! own floor (its quiet tenth), never above a margin under its speech
//! (its loud tenth), and never below an absolute floor. A recording whose
//! speech is not clearly above its floor - music, constant noise - has no
//! pauses to find, and gets none rather than a hundred wrong ones.
//!
//! What makes a cut sound like a person edited it and not a machine is
//! mostly what is left: [`Strength::Natural`] keeps a short breath at both
//! sides of every cut, so sentences do not run into each other.

/// How hard to cut.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Strength {
    /// Only the long pauses, with a generous breath kept. For interviews
    /// and testimonials, where a pause carries meaning.
    Light,
    /// The holes, with the breath between sentences kept. The default.
    Natural,
    /// Sentence against sentence. Shortest, and can sound clipped.
    Aggressive,
}

/// The numbers a [`Strength`] stands for.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Params {
    /// The shortest pause that is cut at all, in seconds.
    pub min_pause: f64,
    /// Sound kept at each side of a cut, in seconds.
    pub breath: f64,
    /// How far above the recording's floor the threshold sits, in dB.
    pub margin_db: f32,
    /// The threshold never sits below this, in dBFS.
    pub absolute_db: f32,
}

impl Strength {
    /// Its numbers.
    pub fn params(self) -> Params {
        match self {
            Strength::Light => Params {
                min_pause: 0.9,
                breath: 0.30,
                margin_db: 6.0,
                absolute_db: -50.0,
            },
            Strength::Natural => Params {
                min_pause: 0.5,
                breath: 0.15,
                margin_db: 9.0,
                absolute_db: -45.0,
            },
            Strength::Aggressive => Params {
                min_pause: 0.25,
                breath: 0.06,
                margin_db: 12.0,
                absolute_db: -40.0,
            },
        }
    }

    /// The name the menus and the API use.
    pub fn id(self) -> &'static str {
        match self {
            Strength::Light => "light",
            Strength::Natural => "natural",
            Strength::Aggressive => "aggressive",
        }
    }

    /// The inverse of [`Strength::id`].
    pub fn from_id(id: &str) -> Option<Strength> {
        match id {
            "light" => Some(Strength::Light),
            "natural" => Some(Strength::Natural),
            "aggressive" => Some(Strength::Aggressive),
            _ => None,
        }
    }
}

/// Analysis windows are this long, in seconds: short enough to find the
/// gap between two words, long enough that one loud sample is not a word.
const WINDOW: f64 = 0.01;
/// A sound shorter than this inside a pause - a click, a lip smack - does
/// not end the pause.
const BLIP: f64 = 0.06;
/// Speech must stand this far above the floor for there to be pauses.
const MIN_CONTRAST_DB: f32 = 10.0;
/// The threshold stays at least this far under the speech.
const SPEECH_HEADROOM_DB: f32 = 6.0;
/// A cut shorter than this after the breath is taken off is not worth a
/// seam.
const MIN_CUT: f64 = 0.1;

/// The pauses in `[from, to)` of a recording, as source-second spans to
/// remove, earliest first.
///
/// `peaks` is the loudest absolute sample of each bucket and
/// `buckets_per_second` how many buckets cover a second. A pause touching
/// `from` or `to` keeps its breath only on the side with sound.
pub fn find_cuts(
    peaks: &[f32],
    buckets_per_second: f32,
    from: f64,
    to: f64,
    strength: Strength,
) -> Vec<(f64, f64)> {
    let params = strength.params();
    let rate = buckets_per_second as f64;
    if !rate.is_finite() || rate <= 0.0 || !from.is_finite() || !to.is_finite() || to <= from {
        return Vec::new();
    }
    if peaks.is_empty() {
        return Vec::new();
    }
    let per_window = ((rate * WINDOW).round() as usize).max(1);
    let first = ((from * rate).floor().max(0.0) as usize).min(peaks.len());
    let last = ((to * rate).ceil().max(0.0) as usize).min(peaks.len());
    if last <= first {
        return Vec::new();
    }
    let window_seconds = per_window as f64 / rate;

    // The level of each window, in dBFS.
    let levels: Vec<f32> = peaks[first..last]
        .chunks(per_window)
        .map(|chunk| {
            decibels(
                chunk
                    .iter()
                    .fold(0.0f32, |high, value| high.max(value.abs())),
            )
        })
        .collect();
    if levels.len() < 3 {
        return Vec::new();
    }
    let floor = percentile(&levels, 0.10);
    let speech = percentile(&levels, 0.90);
    if speech - floor < MIN_CONTRAST_DB {
        return Vec::new();
    }
    let threshold = (floor + params.margin_db)
        .max(params.absolute_db)
        .min(speech - SPEECH_HEADROOM_DB);

    // Runs of quiet windows, with blips inside a pause forgiven.
    let quiet: Vec<bool> = levels.iter().map(|level| *level < threshold).collect();
    let blip_windows = (BLIP / window_seconds).round() as usize;
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut index = 0;
    while index < quiet.len() {
        if !quiet[index] {
            index += 1;
            continue;
        }
        let start = index;
        while index < quiet.len() && quiet[index] {
            index += 1;
        }
        match runs.last_mut() {
            Some(previous) if start - previous.1 <= blip_windows => previous.1 = index,
            _ => runs.push((start, index)),
        }
    }

    let origin = first as f64 / rate;
    let mut cuts = Vec::new();
    for (start, end) in runs {
        let pause_start = (origin + start as f64 * window_seconds).max(from);
        let pause_end = (origin + end as f64 * window_seconds).min(to);
        if pause_end - pause_start < params.min_pause {
            continue;
        }
        // At the edge of the range there is no sound on that side to
        // breathe into: the pause is taken right up to the edge.
        let lead = if start == 0 { 0.0 } else { params.breath };
        let tail = if end >= levels.len() {
            0.0
        } else {
            params.breath
        };
        let cut_start = pause_start + lead;
        let cut_end = pause_end - tail;
        if cut_end - cut_start >= MIN_CUT {
            cuts.push((cut_start, cut_end));
        }
    }
    cuts
}

/// How many seconds the cuts take out in total.
pub fn total(cuts: &[(f64, f64)]) -> f64 {
    cuts.iter().map(|(start, end)| end - start).sum()
}

fn decibels(amplitude: f32) -> f32 {
    20.0 * amplitude.max(1e-6).log10()
}

/// The value `fraction` of the way up the sorted levels.
fn percentile(levels: &[f32], fraction: f32) -> f32 {
    let mut sorted = levels.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let index = ((sorted.len() - 1) as f32 * fraction).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: f32 = 1000.0;

    /// A recording laid out as (seconds, amplitude) stretches.
    fn recording(parts: &[(f64, f32)]) -> Vec<f32> {
        let mut peaks = Vec::new();
        for (seconds, amplitude) in parts {
            let count = (seconds * RATE as f64).round() as usize;
            for index in 0..count {
                // A little movement, so speech is not one flat level.
                let wobble = if index % 7 == 0 { 0.8 } else { 1.0 };
                peaks.push(amplitude * wobble);
            }
        }
        peaks
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 0.02
    }

    #[test]
    fn a_pause_between_two_sentences_is_cut_with_breath_kept() {
        let peaks = recording(&[(2.0, 0.5), (1.0, 0.002), (2.0, 0.5)]);
        let cuts = find_cuts(&peaks, RATE, 0.0, 5.0, Strength::Natural);
        assert_eq!(cuts.len(), 1, "{cuts:?}");
        let (start, end) = cuts[0];
        assert!(
            close(start, 2.15),
            "starts a breath after the words: {start}"
        );
        assert!(close(end, 2.85), "ends a breath before the next: {end}");
    }

    #[test]
    fn the_strengths_cut_more_as_they_get_harder() {
        let peaks = recording(&[
            (1.5, 0.5),
            (0.4, 0.002),
            (1.5, 0.5),
            (0.7, 0.002),
            (1.5, 0.5),
            (1.2, 0.002),
            (1.5, 0.5),
        ]);
        let end = peaks.len() as f64 / RATE as f64;
        let light = total(&find_cuts(&peaks, RATE, 0.0, end, Strength::Light));
        let natural = total(&find_cuts(&peaks, RATE, 0.0, end, Strength::Natural));
        let aggressive = total(&find_cuts(&peaks, RATE, 0.0, end, Strength::Aggressive));
        assert!(light > 0.0, "the long pause goes even at light");
        assert!(light < natural, "{light} < {natural}");
        assert!(natural < aggressive, "{natural} < {aggressive}");
        let light_cuts = find_cuts(&peaks, RATE, 0.0, end, Strength::Light);
        assert_eq!(light_cuts.len(), 1, "light leaves the short pauses alone");
    }

    #[test]
    fn a_click_inside_a_pause_does_not_split_it() {
        let peaks = recording(&[
            (2.0, 0.5),
            (0.5, 0.002),
            (0.02, 0.4),
            (0.5, 0.002),
            (2.0, 0.5),
        ]);
        let end = peaks.len() as f64 / RATE as f64;
        let cuts = find_cuts(&peaks, RATE, 0.0, end, Strength::Natural);
        assert_eq!(cuts.len(), 1, "one pause, one cut: {cuts:?}");
    }

    #[test]
    fn the_threshold_follows_a_noisy_floor() {
        // A shop: the floor is loud, the speech louder still.
        let peaks = recording(&[(2.0, 0.6), (1.0, 0.05), (2.0, 0.6)]);
        let cuts = find_cuts(&peaks, RATE, 0.0, 5.0, Strength::Natural);
        assert_eq!(cuts.len(), 1, "the room noise is still a pause: {cuts:?}");
    }

    #[test]
    fn music_with_no_pauses_gets_no_cuts() {
        let peaks = recording(&[(5.0, 0.4)]);
        assert!(find_cuts(&peaks, RATE, 0.0, 5.0, Strength::Aggressive).is_empty());
        let flat_noise = recording(&[(5.0, 0.01)]);
        assert!(find_cuts(&flat_noise, RATE, 0.0, 5.0, Strength::Aggressive).is_empty());
    }

    #[test]
    fn a_pause_at_the_start_is_cut_right_from_the_edge() {
        let peaks = recording(&[(1.0, 0.002), (3.0, 0.5)]);
        let cuts = find_cuts(&peaks, RATE, 0.0, 4.0, Strength::Natural);
        assert_eq!(cuts.len(), 1, "{cuts:?}");
        assert!(
            close(cuts[0].0, 0.0),
            "no breath before nothing: {:?}",
            cuts[0]
        );
        assert!(
            close(cuts[0].1, 0.85),
            "a breath before the first word: {:?}",
            cuts[0]
        );
    }

    #[test]
    fn only_the_asked_range_is_read() {
        let peaks = recording(&[
            (2.0, 0.5),
            (1.0, 0.002),
            (2.0, 0.5),
            (1.0, 0.002),
            (2.0, 0.5),
        ]);
        // The clip shows 3.5..8.0 of the file: only the second pause.
        let cuts = find_cuts(&peaks, RATE, 3.5, 8.0, Strength::Natural);
        assert_eq!(cuts.len(), 1, "{cuts:?}");
        assert!(cuts[0].0 > 5.0, "{:?}", cuts[0]);
    }

    #[test]
    fn nonsense_input_finds_nothing() {
        assert!(find_cuts(&[], RATE, 0.0, 1.0, Strength::Natural).is_empty());
        assert!(find_cuts(&[0.5; 100], 0.0, 0.0, 1.0, Strength::Natural).is_empty());
        assert!(find_cuts(&[0.5; 100], RATE, 1.0, 0.5, Strength::Natural).is_empty());
    }

    #[test]
    fn ids_round_trip() {
        for strength in [Strength::Light, Strength::Natural, Strength::Aggressive] {
            assert_eq!(Strength::from_id(strength.id()), Some(strength));
        }
        assert_eq!(Strength::from_id("loud"), None);
    }
}
