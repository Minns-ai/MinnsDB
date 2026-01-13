//! Common utilities for the Agentic Database

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Calculate hash of any hashable value
pub fn calculate_hash<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

/// Calculate cosine similarity between two vectors
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

/// Format bytes as human readable string
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB", "PB"];
    
    if bytes == 0 {
        return "0 B".to_string();
    }
    
    let mut size = bytes as f64;
    let mut unit_index = 0;
    
    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }
    
    format!("{:.1} {}", size, UNITS[unit_index])
}

/// Format duration as human readable string
pub fn format_duration(duration: std::time::Duration) -> String {
    let total_seconds = duration.as_secs();
    
    if total_seconds < 60 {
        if duration.as_millis() < 1000 {
            return format!("{}ms", duration.as_millis());
        }
        return format!("{}s", total_seconds);
    }
    
    let minutes = total_seconds / 60;
    let seconds = total_seconds % 60;
    
    if minutes < 60 {
        return format!("{}m {}s", minutes, seconds);
    }
    
    let hours = minutes / 60;
    let minutes = minutes % 60;
    
    if hours < 24 {
        return format!("{}h {}m", hours, minutes);
    }
    
    let days = hours / 24;
    let hours = hours % 24;
    
    format!("{}d {}h", days, hours)
}

/// Simple exponential backoff iterator
pub struct ExponentialBackoff {
    current_delay: std::time::Duration,
    max_delay: std::time::Duration,
    multiplier: f64,
    attempts: usize,
    max_attempts: usize,
}

impl ExponentialBackoff {
    pub fn new() -> Self {
        Self {
            current_delay: std::time::Duration::from_millis(100),
            max_delay: std::time::Duration::from_secs(30),
            multiplier: 2.0,
            attempts: 0,
            max_attempts: 10,
        }
    }
    
    pub fn with_max_attempts(mut self, max_attempts: usize) -> Self {
        self.max_attempts = max_attempts;
        self
    }
    
    pub fn with_max_delay(mut self, max_delay: std::time::Duration) -> Self {
        self.max_delay = max_delay;
        self
    }
    
    pub fn next_delay(&mut self) -> Option<std::time::Duration> {
        if self.attempts >= self.max_attempts {
            return None;
        }
        
        let delay = self.current_delay;
        self.attempts += 1;
        
        self.current_delay = std::cmp::min(
            self.max_delay,
            std::time::Duration::from_millis(
                (self.current_delay.as_millis() as f64 * self.multiplier) as u64
            ),
        );
        
        Some(delay)
    }
    
    pub fn reset(&mut self) {
        self.current_delay = std::time::Duration::from_millis(100);
        self.attempts = 0;
    }
}

/// Simple rate limiter using token bucket algorithm
pub struct RateLimiter {
    tokens: f64,
    last_update: std::time::Instant,
    rate: f64, // tokens per second
    capacity: f64,
}

impl RateLimiter {
    pub fn new(rate_per_second: f64, capacity: f64) -> Self {
        Self {
            tokens: capacity,
            last_update: std::time::Instant::now(),
            rate: rate_per_second,
            capacity,
        }
    }
    
    pub fn try_acquire(&mut self, tokens: f64) -> bool {
        self.refill();
        
        if self.tokens >= tokens {
            self.tokens -= tokens;
            true
        } else {
            false
        }
    }
    
    fn refill(&mut self) {
        let now = std::time::Instant::now();
        let elapsed = now.duration_since(self.last_update).as_secs_f64();
        
        self.tokens = (self.tokens + elapsed * self.rate).min(self.capacity);
        self.last_update = now;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < f32::EPSILON);
        
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine_similarity(&a, &b) - 0.0).abs() < f32::EPSILON);
        
        // Different lengths should return 0
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }
    
    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512.0 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1536), "1.5 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.0 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.0 GB");
    }
    
    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(std::time::Duration::from_millis(500)), "500ms");
        assert_eq!(format_duration(std::time::Duration::from_secs(30)), "30s");
        assert_eq!(format_duration(std::time::Duration::from_secs(90)), "1m 30s");
        assert_eq!(format_duration(std::time::Duration::from_secs(3661)), "1h 1m");
    }
    
    #[test]
    fn test_exponential_backoff() {
        let mut backoff = ExponentialBackoff::new().with_max_attempts(3);
        
        let delay1 = backoff.next_delay().unwrap();
        let delay2 = backoff.next_delay().unwrap();
        let delay3 = backoff.next_delay().unwrap();
        let delay4 = backoff.next_delay();
        
        assert!(delay2 > delay1);
        assert!(delay3 > delay2);
        assert!(delay4.is_none());
    }
    
    #[test]
    fn test_rate_limiter() {
        let mut limiter = RateLimiter::new(10.0, 10.0); // 10 tokens per second, capacity 10
        
        // Should be able to acquire 10 tokens immediately
        assert!(limiter.try_acquire(10.0));
        
        // Should not be able to acquire more
        assert!(!limiter.try_acquire(1.0));
        
        // After some time, should be able to acquire more
        std::thread::sleep(std::time::Duration::from_millis(200));
        assert!(limiter.try_acquire(1.0));
    }
}