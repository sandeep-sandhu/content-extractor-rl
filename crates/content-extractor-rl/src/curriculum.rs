
//! Curriculum learning manager

// ============================================================================
// FILE: crates/content-extractor-rl/src/curriculum.rs
// ============================================================================

pub struct CurriculumManager {
    current_threshold: f32,
    max_threshold: f32,
    increment_rate: f32,
}

impl CurriculumManager {
    /// Create new curriculum manager
    pub fn new() -> Self {
        Self {
            current_threshold: 0.3,
            max_threshold: 1.0,
            increment_rate: 0.01,
        }
    }

    /// Update difficulty threshold
    pub fn update_threshold(&mut self, episode: usize) {
        if episode.is_multiple_of(100) {
            self.current_threshold = (self.current_threshold + self.increment_rate)
                .min(self.max_threshold);
        }
    }

    /// Get current difficulty threshold
    pub fn get_threshold(&self) -> f32 {
        self.current_threshold
    }

    /// Estimate HTML difficulty
    pub fn estimate_difficulty(&self, html: &str) -> f32 {
        let html_len = html.len();
        let script_count = html.matches("<script").count();
        let div_count = html.matches("<div").count();
        let has_article = html.to_lowercase().contains("<article");

        let mut difficulty: f32 = 0.0;

        if html_len > 100_000 {
            difficulty += 0.3;
        } else if html_len > 50_000 {
            difficulty += 0.2;
        }

        if script_count > 20 {
            difficulty += 0.3;
        } else if script_count > 10 {
            difficulty += 0.2;
        }

        if div_count > 100 {
            difficulty += 0.2;
        }

        if has_article {
            difficulty -= 0.2;
        }

        difficulty.clamp(0.0, 1.0)
    }

    /// Check if HTML is appropriate for current curriculum stage
    pub fn is_appropriate(&self, html: &str) -> bool {
        let difficulty = self.estimate_difficulty(html);
        difficulty <= self.current_threshold
    }
}

impl Default for CurriculumManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_curriculum_manager() {
        let mut manager = CurriculumManager::new();
        assert_eq!(manager.get_threshold(), 0.3);

        manager.update_threshold(100);
        assert!(manager.get_threshold() > 0.3);
    }
}
