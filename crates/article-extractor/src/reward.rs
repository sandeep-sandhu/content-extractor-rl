use crate::text_utils::TextUtils;
use std::collections::HashSet;

/// Improved reward calculator with multiple components
pub struct ImprovedRewardCalculator {
    stopwords: HashSet<String>,
}

impl ImprovedRewardCalculator {
    /// Create new reward calculator
    pub fn new(stopwords: HashSet<String>) -> Self {
        Self { stopwords }
    }

    /// Calculate multi-component reward
    pub fn calculate_reward(
        &self,
        extracted_text: &str,
        baseline_score: f32,
    ) -> f32 {
        if extracted_text.is_empty() || extracted_text.len() < 20 {
            return -1.0;
        }

        let quality = self.calculate_quality(extracted_text);
        let length_bonus = self.length_reward(extracted_text);
        let structure_bonus = self.structure_reward(extracted_text);
        let improvement_bonus = self.improvement_reward(quality, baseline_score);
        let coherence_bonus = self.coherence_reward(extracted_text);

        let raw_reward = quality * 0.50
            + length_bonus
            + structure_bonus
            + improvement_bonus
            + coherence_bonus;

        // Scale to [-1, 1] range
        (raw_reward * 2.0 - 1.0).clamp(-1.0, 1.0)
    }

    /// Calculate base quality score
    fn calculate_quality(&self, text: &str) -> f32 {
        let tokens = TextUtils::tokenize(text);
        if tokens.is_empty() {
            return 0.0;
        }

        let mut score = 0.0;

        // Stopword ratio
        let stopword_count = tokens.iter()
            .filter(|t| self.stopwords.contains(*t))
            .count();
        let stopword_ratio = stopword_count as f32 / tokens.len() as f32;

        if (0.35..=0.55).contains(&stopword_ratio) {
            score += 0.35;
        } else {
            score += 0.35 * (1.0 - (stopword_ratio - 0.45).abs() / 0.45).max(0.0);
        }

        // Sentence structure
        let sentences = TextUtils::split_sentences(text);
        if !sentences.is_empty() {
            let avg_len = tokens.len() as f32 / sentences.len() as f32;
            if (12.0..=28.0).contains(&avg_len) {
                score += 0.25;
            } else {
                score += 0.25 * (1.0 - (avg_len - 20.0).abs() / 20.0).max(0.0);
            }
        }

        // Word count
        let word_count = tokens.len();
        if (100..=2000).contains(&word_count) {
            score += 0.20;
        } else if (50..100).contains(&word_count) {
            score += 0.10;
        }

        // Diversity
        let unique: HashSet<_> = tokens.iter().collect();
        let diversity = unique.len() as f32 / tokens.len() as f32;
        if (0.4..=0.8).contains(&diversity) {
            score += 0.20;
        }

        score.clamp(0.0, 1.0)
    }

    /// Length reward
    fn length_reward(&self, text: &str) -> f32 {
        let word_count = text.split_whitespace().count();

        match word_count {
            200..=1500 => 0.2,
            100..=199 => 0.1,
            50..=99 => 0.0,
            1501.. => -0.1 * ((word_count - 1500) as f32 / 1500.0).min(1.0),
            _ => -0.2,
        }
    }

    /// Structure reward
    fn structure_reward(&self, text: &str) -> f32 {
        let paragraphs: Vec<_> = text.split("\n\n")
            .filter(|p| !p.trim().is_empty())
            .collect();

        if paragraphs.is_empty() {
            return -0.1;
        }

        let mut score = 0.0;

        if (3..=20).contains(&paragraphs.len()) {
            score += 0.1;
        }

        let para_lengths: Vec<_> = paragraphs.iter()
            .map(|p| p.split_whitespace().count())
            .collect();

        if !para_lengths.is_empty() {
            let avg_para_len: f32 = para_lengths.iter().sum::<usize>() as f32 / para_lengths.len() as f32;
            if (30.0..=150.0).contains(&avg_para_len) {
                score += 0.1;
            }
        }

        score
    }

    /// Improvement over baseline reward
    fn improvement_reward(&self, quality: f32, baseline: f32) -> f32 {
        if baseline == 0.0 {
            return 0.0;
        }

        let improvement = quality - baseline;
        if improvement > 0.1 {
            0.3
        } else if improvement > 0.05 {
            0.2
        } else if improvement > 0.0 {
            0.1
        } else {
            0.0
        }
    }

    /// Coherence reward
    fn coherence_reward(&self, text: &str) -> f32 {
        let text_lower = text.to_lowercase();
        let words: Vec<_> = text_lower.split_whitespace().collect();
        if words.len() < 10 {
            return 0.0;
        }

        let mut score = 0.0;

        // Bigram diversity
        let bigrams: Vec<_> = words.windows(2)
            .map(|w| format!("{}_{}", w[0], w[1]))
            .collect();

        if !bigrams.is_empty() {
            let unique: HashSet<_> = bigrams.iter().collect();
            let bigram_diversity = unique.len() as f32 / bigrams.len() as f32;
            if bigram_diversity > 0.8 {
                score += 0.1;
            }
        }

        // Check for noise
        let url_count = text.to_lowercase().matches("http").count();
        let email_count = text.matches('@').count();

        if url_count < 2 && email_count < 2 {
            score += 0.1;
        }

        score
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reward_calculator() {
        let stopwords: HashSet<_> = vec!["the", "a", "is"]
            .into_iter()
            .map(|s| s.to_string())
            .collect();

        let calculator = ImprovedRewardCalculator::new(stopwords);

        let good_text = "This is a well-written article with proper structure. \
                         It contains multiple sentences and appropriate punctuation. \
                         The content is substantial and informative.";

        let reward = calculator.calculate_reward(good_text, 0.0);
        assert!(reward > 0.0);
    }
}
