//! 8KB slotted page. Rows packed contiguously.
//!
//! Layout:
//!   [PageHeader 32B] [Slot Directory ->] ... [Free Space] ... [<- Row Data]
//!
//! Slot directory grows forward, row data grows backward. Page is full
//! when they meet.

use crate::types::{PAGE_HEADER_SIZE, PAGE_SIZE, SLOT_SIZE};

/// Page header layout (32 bytes):
///   page_id:    u32  (0..4)
///   row_count:  u16  (4..6)
///   free_start: u16  (6..8)   end of slot directory
///   free_end:   u16  (8..10)  start of row data region
///   flags:      u16  (10..12)
///   checksum:   [u8; 16] (12..28)  truncated blake3 hash of bytes 32..PAGE_SIZE
///   _reserved:  [u8; 4] (28..32)
const HDR_PAGE_ID: usize = 0;
const HDR_ROW_COUNT: usize = 4;
const HDR_FREE_START: usize = 6;
const HDR_FREE_END: usize = 8;
const _HDR_FLAGS: usize = 10;
const HDR_CHECKSUM: usize = 12;
const CHECKSUM_LEN: usize = 16;

/// Slot entry layout (4 bytes):
///   offset: u16  (byte offset of row data from page start)
///   length: u16  (byte length of encoded row)
/// A dead slot has offset = 0 AND length = 0.
pub struct Page {
    data: Box<[u8; PAGE_SIZE]>,
}

impl Page {
    pub fn new(page_id: u32) -> Self {
        let mut data = Box::new([0u8; PAGE_SIZE]);
        // Write page_id
        data[HDR_PAGE_ID..HDR_PAGE_ID + 4].copy_from_slice(&page_id.to_le_bytes());
        // row_count = 0
        data[HDR_ROW_COUNT..HDR_ROW_COUNT + 2].copy_from_slice(&0u16.to_le_bytes());
        // free_start = PAGE_HEADER_SIZE (first slot would go here)
        data[HDR_FREE_START..HDR_FREE_START + 2]
            .copy_from_slice(&(PAGE_HEADER_SIZE as u16).to_le_bytes());
        // free_end = PAGE_SIZE (row data grows backward from end)
        data[HDR_FREE_END..HDR_FREE_END + 2].copy_from_slice(&(PAGE_SIZE as u16).to_le_bytes());
        Page { data }
    }

    /// Reset this page to an empty state for reuse from the pool.
    pub fn reset(&mut self) {
        // Clear all slots and data, keeping the page_id
        let page_id = self.page_id();
        *self = Page::new(page_id);
    }

    pub fn page_id(&self) -> u32 {
        u32::from_le_bytes(self.data[HDR_PAGE_ID..HDR_PAGE_ID + 4].try_into().unwrap())
    }

    fn row_count(&self) -> u16 {
        u16::from_le_bytes(
            self.data[HDR_ROW_COUNT..HDR_ROW_COUNT + 2]
                .try_into()
                .unwrap(),
        )
    }

    fn set_row_count(&mut self, count: u16) {
        self.data[HDR_ROW_COUNT..HDR_ROW_COUNT + 2].copy_from_slice(&count.to_le_bytes());
    }

    fn free_start(&self) -> u16 {
        u16::from_le_bytes(
            self.data[HDR_FREE_START..HDR_FREE_START + 2]
                .try_into()
                .unwrap(),
        )
    }

    fn set_free_start(&mut self, val: u16) {
        self.data[HDR_FREE_START..HDR_FREE_START + 2].copy_from_slice(&val.to_le_bytes());
    }

    fn free_end(&self) -> u16 {
        u16::from_le_bytes(
            self.data[HDR_FREE_END..HDR_FREE_END + 2]
                .try_into()
                .unwrap(),
        )
    }

    fn set_free_end(&mut self, val: u16) {
        self.data[HDR_FREE_END..HDR_FREE_END + 2].copy_from_slice(&val.to_le_bytes());
    }

    /// Available free space in bytes.
    pub fn free_space(&self) -> usize {
        let fs = self.free_start() as usize;
        let fe = self.free_end() as usize;
        fe.saturating_sub(fs)
    }

    /// Read slot entry at index.
    fn read_slot(&self, slot_idx: u16) -> (u16, u16) {
        let pos = PAGE_HEADER_SIZE + (slot_idx as usize) * SLOT_SIZE;
        let offset = u16::from_le_bytes(self.data[pos..pos + 2].try_into().unwrap());
        let length = u16::from_le_bytes(self.data[pos + 2..pos + 4].try_into().unwrap());
        (offset, length)
    }

    /// Write slot entry at index.
    fn write_slot(&mut self, slot_idx: u16, offset: u16, length: u16) {
        let pos = PAGE_HEADER_SIZE + (slot_idx as usize) * SLOT_SIZE;
        self.data[pos..pos + 2].copy_from_slice(&offset.to_le_bytes());
        self.data[pos + 2..pos + 4].copy_from_slice(&length.to_le_bytes());
    }

    /// Insert encoded row bytes. Returns slot index, or None if page is full.
    /// Reuses dead slots when available to avoid growing the slot directory.
    pub fn insert_row(&mut self, row_bytes: &[u8]) -> Option<u16> {
        let row_len = row_bytes.len();

        // Try to find a dead slot to reuse (avoids growing the slot directory)
        let dead_slot = self.find_dead_slot();

        let needed = if dead_slot.is_some() {
            row_len // No new slot entry needed
        } else {
            row_len + SLOT_SIZE // Need space for both slot and data
        };

        if self.free_space() < needed {
            return None;
        }

        let new_free_end = self.free_end() - row_len as u16;

        // Row data grows backward from free_end
        self.data[new_free_end as usize..new_free_end as usize + row_len]
            .copy_from_slice(row_bytes);

        let slot_idx = if let Some(dead_idx) = dead_slot {
            // Reuse dead slot — no directory growth
            self.write_slot(dead_idx, new_free_end, row_len as u16);
            dead_idx
        } else {
            // Allocate new slot at the end of the directory
            let idx = self.row_count();
            let new_free_start = self.free_start() + SLOT_SIZE as u16;
            self.write_slot(idx, new_free_end, row_len as u16);
            self.set_row_count(idx + 1);
            self.set_free_start(new_free_start);
            idx
        };

        self.set_free_end(new_free_end);
        Some(slot_idx)
    }

    /// Find the first dead slot in the directory (offset=0, length=0).
    fn find_dead_slot(&self) -> Option<u16> {
        let total = self.row_count();
        (0..total).find(|&i| self.is_dead(i))
    }

    /// Read row bytes at a given slot index. Returns &[u8] slice into page.
    pub fn read_row(&self, slot_idx: u16) -> Option<&[u8]> {
        if slot_idx >= self.row_count() {
            return None;
        }
        let (offset, length) = self.read_slot(slot_idx);
        if offset == 0 && length == 0 {
            return None; // dead slot
        }
        Some(&self.data[offset as usize..(offset + length) as usize])
    }

    /// Read a mutable reference to row bytes (for in-place valid_until update).
    pub fn read_row_mut(&mut self, slot_idx: u16) -> Option<&mut [u8]> {
        if slot_idx >= self.row_count() {
            return None;
        }
        let (offset, length) = self.read_slot(slot_idx);
        if offset == 0 && length == 0 {
            return None; // dead slot
        }
        Some(&mut self.data[offset as usize..(offset + length) as usize])
    }

    /// Mark a slot as dead (for compaction). Does not reclaim space immediately.
    pub fn mark_dead(&mut self, slot_idx: u16) {
        self.write_slot(slot_idx, 0, 0);
    }

    fn is_dead(&self, slot_idx: u16) -> bool {
        let (offset, length) = self.read_slot(slot_idx);
        offset == 0 && length == 0
    }

    /// Number of live (non-dead) rows.
    pub fn live_row_count(&self) -> u16 {
        let total = self.row_count();
        let mut live = 0u16;
        for i in 0..total {
            if !self.is_dead(i) {
                live += 1;
            }
        }
        live
    }

    /// Dead space: total bytes occupied by dead rows.
    pub fn dead_space(&self) -> usize {
        let total = self.row_count();
        let total_row_area = PAGE_SIZE - self.free_end() as usize;
        let mut live_row_bytes = 0usize;
        for i in 0..total {
            let (offset, length) = self.read_slot(i);
            if offset != 0 || length != 0 {
                live_row_bytes += length as usize;
            }
        }
        total_row_area.saturating_sub(live_row_bytes)
    }

    /// Compact: rebuild page with only live rows, reclaiming dead space.
    /// Preserves slot indices for live rows. Dead slots remain as dead markers.
    pub fn compact(&mut self) {
        let total = self.row_count();
        if total == 0 {
            return;
        }

        // Collect live rows (slot_idx, data)
        let mut live_rows: Vec<(u16, Vec<u8>)> = Vec::new();
        for i in 0..total {
            if let Some(data) = self.read_row(i) {
                live_rows.push((i, data.to_vec()));
            }
        }

        // Reset row data area
        let mut new_free_end = PAGE_SIZE as u16;

        // Re-pack rows from end, preserving slot indices
        for (slot_idx, data) in &live_rows {
            new_free_end -= data.len() as u16;
            self.data[new_free_end as usize..new_free_end as usize + data.len()]
                .copy_from_slice(data);
            self.write_slot(*slot_idx, new_free_end, data.len() as u16);
        }

        self.set_free_end(new_free_end);
        // free_start stays the same (slot directory size unchanged)
    }

    /// Iterate all live slots: yields (slot_idx, &[u8]) pairs.
    pub fn iter_live(&self) -> impl Iterator<Item = (u16, &[u8])> {
        let total = self.row_count();
        (0..total).filter_map(move |i| self.read_row(i).map(|data| (i, data)))
    }

    /// Compute blake3 checksum covering all page bytes except the checksum field
    /// itself (bytes 12..28). This includes header fields so that corruption in
    /// page_id / row_count / free_start / free_end / flags is detected.
    pub fn update_checksum(&mut self) {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.data[..HDR_CHECKSUM]);
        hasher.update(&self.data[HDR_CHECKSUM + CHECKSUM_LEN..]);
        let hash = hasher.finalize();
        self.data[HDR_CHECKSUM..HDR_CHECKSUM + CHECKSUM_LEN]
            .copy_from_slice(&hash.as_bytes()[..CHECKSUM_LEN]);
    }

    /// Verify the page checksum. Returns true if valid or if checksum is all zeros
    /// (legacy page without checksum).
    pub fn verify_checksum(&self) -> bool {
        let stored = &self.data[HDR_CHECKSUM..HDR_CHECKSUM + CHECKSUM_LEN];
        if stored.iter().all(|&b| b == 0) {
            return true;
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.data[..HDR_CHECKSUM]);
        hasher.update(&self.data[HDR_CHECKSUM + CHECKSUM_LEN..]);
        let hash = hasher.finalize();
        let expected = &hash.as_bytes()[..CHECKSUM_LEN];
        stored == expected
    }

    /// Raw page bytes for persistence (write whole page to ReDB).
    /// Automatically updates the checksum before returning.
    pub fn as_bytes(&mut self) -> &[u8; PAGE_SIZE] {
        self.update_checksum();
        &self.data
    }

    /// Raw page bytes without updating checksum (for read-only access).
    pub fn as_bytes_readonly(&self) -> &[u8; PAGE_SIZE] {
        &self.data
    }

    /// Restore from raw bytes (load from ReDB).
    /// Validates structural invariants and checksum; resets the page to empty
    /// if corruption is detected so callers never operate on garbage data.
    pub fn from_bytes(data: Box<[u8; PAGE_SIZE]>) -> Self {
        let mut page = Page { data };
        let page_id = page.page_id();
        let fs = page.free_start() as usize;
        let fe = page.free_end() as usize;
        if fs < PAGE_HEADER_SIZE || fe > PAGE_SIZE || fs > fe {
            tracing::warn!(
                "page {} has invalid header: free_start={}, free_end={} — resetting to empty",
                page_id,
                fs,
                fe
            );
            page.reset();
            return page;
        }
        if !page.verify_checksum() {
            tracing::warn!("page {} checksum mismatch — resetting to empty", page_id);
            page.reset();
        }
        page
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_page() {
        let page = Page::new(42);
        assert_eq!(page.page_id(), 42);
        assert_eq!(page.row_count(), 0);
        assert_eq!(page.live_row_count(), 0);
        assert_eq!(page.free_space(), PAGE_SIZE - PAGE_HEADER_SIZE);
    }

    #[test]
    fn test_insert_and_read() {
        let mut page = Page::new(0);
        let row = b"hello world row data";
        let slot = page.insert_row(row).unwrap();
        assert_eq!(slot, 0);
        assert_eq!(page.row_count(), 1);
        assert_eq!(page.live_row_count(), 1);

        let read_back = page.read_row(0).unwrap();
        assert_eq!(read_back, row);
    }

    #[test]
    fn test_multiple_inserts() {
        let mut page = Page::new(0);
        let rows: Vec<Vec<u8>> = (0..10)
            .map(|i| format!("row-{:04}", i).into_bytes())
            .collect();

        for (i, row) in rows.iter().enumerate() {
            let slot = page.insert_row(row).unwrap();
            assert_eq!(slot, i as u16);
        }

        assert_eq!(page.row_count(), 10);
        assert_eq!(page.live_row_count(), 10);

        for (i, row) in rows.iter().enumerate() {
            assert_eq!(page.read_row(i as u16).unwrap(), row.as_slice());
        }
    }

    #[test]
    fn test_page_full() {
        let mut page = Page::new(0);
        // Each insert needs row_bytes + 4 bytes slot
        // Free space = 8192 - 32 = 8160
        // Insert rows until full
        let row = vec![0u8; 100]; // 100 bytes + 4 slot = 104 per row
        let max_rows = 8160 / 104; // ~78
        for _ in 0..max_rows {
            assert!(page.insert_row(&row).is_some());
        }
        // Eventually should fail
        // Try one more — might or might not fit depending on rounding
        // Just verify we can't insert indefinitely
        let mut inserted = max_rows;
        while page.insert_row(&row).is_some() {
            inserted += 1;
            if inserted > 200 {
                panic!("page never fills up");
            }
        }
        assert!(inserted <= 82); // sanity check
    }

    #[test]
    fn test_mark_dead_and_read() {
        let mut page = Page::new(0);
        page.insert_row(b"row0").unwrap();
        page.insert_row(b"row1").unwrap();
        page.insert_row(b"row2").unwrap();

        assert_eq!(page.live_row_count(), 3);

        page.mark_dead(1);
        assert_eq!(page.live_row_count(), 2);
        assert!(page.read_row(1).is_none());
        assert_eq!(page.read_row(0).unwrap(), b"row0");
        assert_eq!(page.read_row(2).unwrap(), b"row2");
    }

    #[test]
    fn test_compact() {
        let mut page = Page::new(0);
        page.insert_row(b"row0-data").unwrap();
        page.insert_row(b"row1-data").unwrap();
        page.insert_row(b"row2-data").unwrap();

        let free_before = page.free_space();
        page.mark_dead(1);
        // Free space doesn't change from mark_dead alone
        assert_eq!(page.free_space(), free_before);

        page.compact();
        // After compaction, free space should increase
        assert!(page.free_space() > free_before);

        // Live rows still accessible at original slot indices
        assert_eq!(page.read_row(0).unwrap(), b"row0-data");
        assert!(page.read_row(1).is_none()); // still dead
        assert_eq!(page.read_row(2).unwrap(), b"row2-data");
    }

    #[test]
    fn test_read_row_mut() {
        let mut page = Page::new(0);
        let row = vec![0u8; 20];
        page.insert_row(&row).unwrap();

        let mutable = page.read_row_mut(0).unwrap();
        mutable[0] = 0xFF;
        mutable[19] = 0xAB;

        let read_back = page.read_row(0).unwrap();
        assert_eq!(read_back[0], 0xFF);
        assert_eq!(read_back[19], 0xAB);
    }

    #[test]
    fn test_iter_live() {
        let mut page = Page::new(0);
        page.insert_row(b"a").unwrap();
        page.insert_row(b"b").unwrap();
        page.insert_row(b"c").unwrap();
        page.mark_dead(1);

        let live: Vec<(u16, Vec<u8>)> = page.iter_live().map(|(i, d)| (i, d.to_vec())).collect();
        assert_eq!(live.len(), 2);
        assert_eq!(live[0].0, 0);
        assert_eq!(live[0].1, b"a");
        assert_eq!(live[1].0, 2);
        assert_eq!(live[1].1, b"c");
    }

    #[test]
    fn test_bytes_roundtrip() {
        let mut page = Page::new(7);
        page.insert_row(b"test data").unwrap();

        let bytes = *page.as_bytes();
        let restored = Page::from_bytes(Box::new(bytes));
        assert_eq!(restored.page_id(), 7);
        assert_eq!(restored.read_row(0).unwrap(), b"test data");
    }
}
