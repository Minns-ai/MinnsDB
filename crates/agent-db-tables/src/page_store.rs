//! Collection of pages for a table. Handles allocation and free-space tracking.

use std::collections::BTreeMap;

use rustc_hash::FxHashSet;

use crate::page::Page;
use crate::types::{RowPointer, SLOT_SIZE};

/// Free-space bucket granularity (bytes). Pages are bucketed by available space.
const BUCKET_SIZE: usize = 256;

fn bucket_for(free_space: usize) -> usize {
    free_space / BUCKET_SIZE
}

pub struct PageStore {
    /// All pages, indexed by page_id.
    pages: Vec<Page>,
    /// Pages with free space, bucketed by available space.
    /// Key = free_space / BUCKET_SIZE, Value = set of page_ids in that bucket.
    free_list: BTreeMap<usize, Vec<u32>>,
    /// Reverse map: page_id → bucket for O(1) removal from free_list.
    page_bucket: rustc_hash::FxHashMap<u32, usize>,
    /// Total live rows across all pages.
    live_row_count: usize,
    /// Pages modified since last persist.
    dirty_pages: FxHashSet<u32>,
}

impl Default for PageStore {
    fn default() -> Self {
        Self::new()
    }
}

impl PageStore {
    pub fn new() -> Self {
        PageStore {
            pages: Vec::new(),
            free_list: BTreeMap::new(),
            page_bucket: rustc_hash::FxHashMap::default(),
            live_row_count: 0,
            dirty_pages: FxHashSet::default(),
        }
    }

    /// Insert encoded row bytes. Finds a page with space or allocates new.
    /// Returns RowPointer.
    pub fn insert(&mut self, row_bytes: &[u8]) -> RowPointer {
        let needed = row_bytes.len() + SLOT_SIZE;
        let min_bucket = bucket_for(needed);

        // Find a page with enough space
        let mut found_page_id = None;
        for (&bucket, page_ids) in self.free_list.range(min_bucket..) {
            // Bucket might not have enough if it's the minimum bucket
            // but higher buckets definitely do
            if bucket > min_bucket {
                if let Some(&pid) = page_ids.last() {
                    found_page_id = Some(pid);
                    break;
                }
            } else {
                // Same bucket — check actual free space
                for &pid in page_ids.iter().rev() {
                    let page = &self.pages[pid as usize];
                    if page.free_space() >= needed {
                        found_page_id = Some(pid);
                        break;
                    }
                }
                if found_page_id.is_some() {
                    break;
                }
            }
        }

        let page_id = match found_page_id {
            Some(pid) => {
                // Remove from old bucket
                self.remove_from_free_list(pid);
                pid
            },
            None => {
                // Allocate new page
                self.allocate_page()
            },
        };

        let page = &mut self.pages[page_id as usize];
        let slot_idx = page.insert_row(row_bytes).expect("page should have space");

        // Re-bucket the page
        let new_free = page.free_space();
        if new_free > SLOT_SIZE {
            let bucket = bucket_for(new_free);
            self.free_list.entry(bucket).or_default().push(page_id);
            self.page_bucket.insert(page_id, bucket);
        } else {
            self.page_bucket.remove(&page_id);
        }

        self.dirty_pages.insert(page_id);
        self.live_row_count += 1;

        RowPointer { page_id, slot_idx }
    }

    /// Read row bytes at a RowPointer. O(1).
    pub fn read(&self, ptr: RowPointer) -> Option<&[u8]> {
        self.pages.get(ptr.page_id as usize)?.read_row(ptr.slot_idx)
    }

    /// Read mutable row bytes (for in-place valid_until writes).
    pub fn read_mut(&mut self, ptr: RowPointer) -> Option<&mut [u8]> {
        let page = self.pages.get_mut(ptr.page_id as usize)?;
        self.dirty_pages.insert(ptr.page_id);
        page.read_row_mut(ptr.slot_idx)
    }

    /// Mark a row as dead (slot becomes dead, space reclaimable on compaction).
    pub fn mark_dead(&mut self, ptr: RowPointer) {
        if let Some(page) = self.pages.get_mut(ptr.page_id as usize) {
            page.mark_dead(ptr.slot_idx);
            self.dirty_pages.insert(ptr.page_id);
            self.live_row_count = self.live_row_count.saturating_sub(1);
        }
    }

    /// Iterate all live rows across all pages.
    pub fn iter_live(&self) -> impl Iterator<Item = (RowPointer, &[u8])> {
        self.pages.iter().flat_map(|page| {
            let page_id = page.page_id();
            page.iter_live()
                .map(move |(slot_idx, data)| (RowPointer { page_id, slot_idx }, data))
        })
    }

    /// Update blake3 checksums on all dirty pages. Call before persistence.
    pub fn update_checksums(&mut self) {
        for &page_id in &self.dirty_pages.clone() {
            if let Some(page) = self.pages.get_mut(page_id as usize) {
                page.update_checksum();
            }
        }
    }

    /// Page count.
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// Live row count.
    pub fn live_row_count(&self) -> usize {
        self.live_row_count
    }

    /// Dirty pages since last persist.
    pub fn dirty_pages(&self) -> &FxHashSet<u32> {
        &self.dirty_pages
    }

    pub fn clear_dirty(&mut self) {
        self.dirty_pages.clear();
    }

    /// Get a page by id for persistence.
    pub fn get_page(&self, page_id: u32) -> Option<&Page> {
        self.pages.get(page_id as usize)
    }

    /// Compact a specific page (reclaim dead row space).
    pub fn compact_page(&mut self, page_id: u32) {
        if (page_id as usize) >= self.pages.len() {
            return;
        }
        self.remove_from_free_list(page_id);
        self.pages[page_id as usize].compact();
        let new_free = self.pages[page_id as usize].free_space();
        if new_free > SLOT_SIZE {
            let bucket = bucket_for(new_free);
            self.free_list.entry(bucket).or_default().push(page_id);
            self.page_bucket.insert(page_id, bucket);
        }
        self.dirty_pages.insert(page_id);
    }

    /// Restore from persisted pages.
    pub fn from_pages(pages: Vec<Page>) -> Self {
        let mut free_list = BTreeMap::new();
        let mut page_bucket = rustc_hash::FxHashMap::default();
        let mut live_row_count = 0usize;

        for page in &pages {
            let pid = page.page_id();
            let free = page.free_space();
            if free > SLOT_SIZE {
                let bucket = bucket_for(free);
                free_list.entry(bucket).or_insert_with(Vec::new).push(pid);
                page_bucket.insert(pid, bucket);
            }
            live_row_count += page.live_row_count() as usize;
        }

        PageStore {
            pages,
            free_list,
            page_bucket,
            live_row_count,
            dirty_pages: FxHashSet::default(),
        }
    }

    fn allocate_page(&mut self) -> u32 {
        let page_id = self.pages.len() as u32;
        let page = Page::new(page_id);
        self.pages.push(page);
        self.dirty_pages.insert(page_id);
        // Don't add to free list — caller will insert a row then re-add.
        page_id
    }

    fn remove_from_free_list(&mut self, page_id: u32) {
        if let Some(bucket) = self.page_bucket.remove(&page_id) {
            if let Some(page_ids) = self.free_list.get_mut(&bucket) {
                if let Some(pos) = page_ids.iter().position(|&pid| pid == page_id) {
                    page_ids.swap_remove(pos);
                }
                if page_ids.is_empty() {
                    self.free_list.remove(&bucket);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_read() {
        let mut store = PageStore::new();
        let row = b"test row data here";
        let ptr = store.insert(row);
        assert_eq!(ptr.page_id, 0);
        assert_eq!(ptr.slot_idx, 0);

        let data = store.read(ptr).unwrap();
        assert_eq!(data, row);
        assert_eq!(store.live_row_count(), 1);
        assert_eq!(store.page_count(), 1);
    }

    #[test]
    fn test_multi_page_allocation() {
        let mut store = PageStore::new();
        // Insert enough rows to fill multiple pages
        let row = vec![0u8; 1000]; // 1000 bytes + 4 slot = ~1004 per row
                                   // 8160 / 1004 ~ 8 rows per page
        let mut ptrs = Vec::new();
        for _ in 0..25 {
            ptrs.push(store.insert(&row));
        }

        assert!(store.page_count() >= 3);
        assert_eq!(store.live_row_count(), 25);

        // All readable
        for ptr in &ptrs {
            assert!(store.read(*ptr).is_some());
        }
    }

    #[test]
    fn test_mark_dead() {
        let mut store = PageStore::new();
        let p1 = store.insert(b"row1");
        let p2 = store.insert(b"row2");
        assert_eq!(store.live_row_count(), 2);

        store.mark_dead(p1);
        assert_eq!(store.live_row_count(), 1);
        assert!(store.read(p1).is_none());
        assert!(store.read(p2).is_some());
    }

    #[test]
    fn test_read_mut() {
        let mut store = PageStore::new();
        let ptr = store.insert(&[0u8; 20]);
        {
            let data = store.read_mut(ptr).unwrap();
            data[0] = 0xFF;
        }
        assert_eq!(store.read(ptr).unwrap()[0], 0xFF);
        assert!(store.dirty_pages().contains(&ptr.page_id));
    }

    #[test]
    fn test_iter_live() {
        let mut store = PageStore::new();
        store.insert(b"a");
        store.insert(b"b");
        let p3 = store.insert(b"c");
        store.mark_dead(p3);

        let live: Vec<_> = store.iter_live().collect();
        assert_eq!(live.len(), 2);
    }

    #[test]
    fn test_dirty_tracking() {
        let mut store = PageStore::new();
        store.insert(b"row");
        assert!(!store.dirty_pages().is_empty());
        store.clear_dirty();
        assert!(store.dirty_pages().is_empty());

        // Mutating marks dirty again
        let ptr = store.insert(b"row2");
        store.clear_dirty();
        store.read_mut(ptr);
        assert!(store.dirty_pages().contains(&ptr.page_id));
    }

    #[test]
    fn test_from_pages() {
        let mut page = Page::new(0);
        page.insert_row(b"hello").unwrap();
        page.insert_row(b"world").unwrap();

        let store = PageStore::from_pages(vec![page]);
        assert_eq!(store.page_count(), 1);
        assert_eq!(store.live_row_count(), 2);
        assert!(store
            .read(RowPointer {
                page_id: 0,
                slot_idx: 0
            })
            .is_some());
    }
}
