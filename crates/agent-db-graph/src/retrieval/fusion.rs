//! Multi-list Reciprocal Rank Fusion (RRF).

use std::collections::HashMap;
use std::hash::Hash;

/// Fuse N ranked lists via Reciprocal Rank Fusion.
///
/// Each input list is a `Vec<(Id, score)>` **pre-sorted by score descending**.
/// The function assigns each item a rank-based score `1 / (k + rank)` within
/// each list, then sums across all lists. Returns results sorted by fused
/// score descending.
///
/// `k` is the smoothing constant (typically 60.0). Higher `k` reduces the
/// influence of top ranks and produces a more uniform blending.
///
/// Empty lists and empty input are handled gracefully.
pub fn multi_list_rrf<Id>(ranked_lists: &[Vec<(Id, f32)>], k: f32) -> Vec<(Id, f32)>
where
    Id: Copy + Eq + Hash,
{
    let mut scores: HashMap<Id, f32> = HashMap::new();

    for list in ranked_lists {
        for (rank, &(id, _)) in list.iter().enumerate() {
            let rrf_score = 1.0 / (k + (rank + 1) as f32);
            *scores.entry(id).or_insert(0.0) += rrf_score;
        }
    }

    let mut results: Vec<(Id, f32)> = scores.into_iter().collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_list_rrf_empty() {
        let lists: Vec<Vec<(u64, f32)>> = vec![];
        let result = multi_list_rrf(&lists, 60.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_multi_list_rrf_single_list() {
        let lists = vec![vec![(1u64, 10.0), (2, 8.0), (3, 5.0)]];
        let result = multi_list_rrf(&lists, 60.0);
        assert_eq!(result.len(), 3);
        // Rank 1 gets 1/(60+1), rank 2 gets 1/(60+2), rank 3 gets 1/(60+3)
        assert_eq!(result[0].0, 1);
        assert_eq!(result[1].0, 2);
        assert_eq!(result[2].0, 3);
    }

    #[test]
    fn test_multi_list_rrf_two_lists() {
        // List 1: [A, B, C]
        // List 2: [C, A, D]
        // A appears at rank 1 in both => highest score
        // C appears at rank 3 and rank 1 => second
        let list1 = vec![(1u64, 10.0), (2, 8.0), (3, 5.0)];
        let list2 = vec![(3u64, 0.95), (1, 0.90), (4, 0.85)];
        let result = multi_list_rrf(&[list1, list2], 60.0);

        assert_eq!(result.len(), 4);
        // Item 1: 1/(60+1) + 1/(60+2) = 0.01639 + 0.01613 = 0.03252
        // Item 3: 1/(60+3) + 1/(60+1) = 0.01587 + 0.01639 = 0.03227
        // Item 1 should be first
        assert_eq!(result[0].0, 1);
        assert_eq!(result[1].0, 3);
    }

    #[test]
    fn test_multi_list_rrf_three_lists() {
        let list1 = vec![(10u64, 1.0), (20, 0.9)];
        let list2 = vec![(20u64, 1.0), (30, 0.9)];
        let list3 = vec![(20u64, 1.0), (10, 0.9)];
        let result = multi_list_rrf(&[list1, list2, list3], 60.0);

        // Item 20 appears at rank 2,1,1 => highest aggregate
        // Item 10 appears at rank 1,_,2
        assert_eq!(result[0].0, 20);
    }
}
