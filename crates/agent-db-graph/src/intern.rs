//! String interning pool for deduplicating repeated string values.
//!
//! Uses `FxHashSet<Arc<str>>` to ensure that identical strings share a single
//! heap allocation. This is particularly effective for limited-vocabulary fields
//! like `event_type`, `tool_name`, `tool_type`, `context_type`, etc.

use rustc_hash::FxHashSet;
use std::sync::Arc;

/// A string interner that deduplicates string allocations.
///
/// When the same string is interned multiple times, all callers receive
/// an `Arc<str>` pointing to the same allocation, reducing memory usage
/// and improving cache locality.
#[derive(Debug, Clone)]
pub struct Interner {
    pool: FxHashSet<Arc<str>>,
}

impl Default for Interner {
    fn default() -> Self {
        Self::new()
    }
}

impl Interner {
    pub fn new() -> Self {
        Self {
            pool: FxHashSet::default(),
        }
    }

    /// Intern a string, returning a shared reference.
    ///
    /// If the string has been interned before, returns a clone of the existing
    /// `Arc<str>` (cheap ref-count bump). Otherwise, allocates a new `Arc<str>`
    /// and stores it in the pool.
    pub fn intern(&mut self, s: &str) -> Arc<str> {
        if let Some(existing) = self.pool.get(s) {
            return Arc::clone(existing);
        }
        let arc: Arc<str> = Arc::from(s);
        self.pool.insert(Arc::clone(&arc));
        arc
    }

    /// Number of unique strings in the pool.
    pub fn len(&self) -> usize {
        self.pool.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pool.is_empty()
    }

    /// Remove a string from the pool.
    pub fn remove(&mut self, s: &str) {
        self.pool.remove(s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intern_returns_same_arc() {
        let mut interner = Interner::new();
        let a = interner.intern("hello");
        let b = interner.intern("hello");
        assert!(Arc::ptr_eq(&a, &b));
        assert_eq!(interner.len(), 1);
    }

    #[test]
    fn test_intern_different_strings() {
        let mut interner = Interner::new();
        let a = interner.intern("hello");
        let b = interner.intern("world");
        assert!(!Arc::ptr_eq(&a, &b));
        assert_eq!(interner.len(), 2);
    }

    #[test]
    fn test_intern_remove() {
        let mut interner = Interner::new();
        interner.intern("hello");
        assert_eq!(interner.len(), 1);
        interner.remove("hello");
        assert_eq!(interner.len(), 0);
    }

    #[test]
    fn test_intern_empty_string() {
        let mut interner = Interner::new();
        let a = interner.intern("");
        let b = interner.intern("");
        assert!(Arc::ptr_eq(&a, &b));
    }
}
