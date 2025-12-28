//! Enhanced evaluation against ground truth data from pre-extracted JSON files
// ============================================================================
// FILE: crates/article-extractor/src/ground_truth.rs
// ============================================================================


use crate::{Result, text_utils::TextUtils};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::collections::HashSet;
// use tracing::{info, warn};

/// Ground truth data from pre-extracted JSON
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundTruthData {
    #[serde(rename = "type")]
    pub data_type: Option<String>,
    pub data_key: Option<String>,
    pub fetch_timestamp: Option<String>,
    pub session_id: Option<String>,
    pub mod_date: Option<String>,
    pub title: Option<String>,
    pub text: Option<String>,
    #[serde(rename = "URL")]
    pub url: Option<String>,
    pub pub_date: Option<String>,
    #[serde(rename = "pubDate")]
    pub pubdate: Option<String>,
    pub author: Option<String>,
    #[serde(rename = "sourceName")]
    pub source_name: Option<Vec<String>>,
    pub language: Option<String>,
    pub keywords: Option<Vec<String>>,
    pub industries: Option<Vec<String>>,
    #[serde(rename = "uniqueID")]
    pub unique_id: Option<String>,
    pub module: Option<String>,
}

impl GroundTruthData {
    /// Load from JSON file
    pub fn load(path: &Path) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let data: GroundTruthData = serde_json::from_str(&json)
            .map_err(|e| crate::ExtractionError::ParseError(
                format!("Failed to parse ground truth JSON: {}", e)
            ))?;
        Ok(data)
    }

    /// Get the ground truth text
    pub fn get_text(&self) -> &str {
        self.text.as_deref().unwrap_or("")
    }

    /// Get the ground truth title
    pub fn get_title(&self) -> &str {
        self.title.as_deref().unwrap_or("")
    }

    /// Get the publication date (handles both variants)
    pub fn get_pubdate(&self) -> Option<&str> {
        self.pubdate.as_deref().or_else(|| self.pub_date.as_deref())
    }

    /// Get the URL
    pub fn get_url(&self) -> &str {
        self.url.as_deref().unwrap_or("")
    }

    /// Get author name(s)
    pub fn get_author(&self) -> Option<String> {
        self.author.clone().or_else(|| {
            self.source_name.as_ref().and_then(|names| {
                if names.is_empty() {
                    None
                } else {
                    Some(names.join(", "))
                }
            })
        })
    }
}

/// Evaluation metrics comparing extracted vs ground truth
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationMetrics {
    // Text similarity metrics
    pub text_jaccard_similarity: f32,  // Jaccard similarity of words
    pub text_precision: f32,            // % of extracted words in ground truth
    pub text_recall: f32,               // % of ground truth words in extracted
    pub text_f1_score: f32,             // Harmonic mean of precision/recall

    // Length-based metrics
    pub length_ratio: f32,              // extracted_len / ground_truth_len
    pub length_difference: i32,         // Absolute difference in characters

    // Semantic metrics
    pub sentence_overlap: f32,          // Overlap in sentence count
    pub paragraph_overlap: f32,         // Overlap in paragraph structure

    // Title matching
    pub title_jaccard_similarity: f32,  // Jaccard similarity for title
    pub title_match_score: f32,         // Overall title match (0-1)

    // Combined quality metrics
    pub text_similarity_score: f32,     // Weighted text similarity (40%)
    pub title_similarity_score: f32,    // Weighted title similarity (20%)
    pub existing_quality_score: f32,    // Existing quality metrics (40%)
    pub combined_quality: f32,          // Final weighted combination
}

impl EvaluationMetrics {
    /// Calculate combined quality score with proper weighting
    /// Text similarity: 40%, Title match: 20%, Existing quality: 40%
    pub fn calculate_combined_quality(&mut self, existing_quality: f32) {
        // Text similarity component (40%)
        self.text_similarity_score = self.text_jaccard_similarity * 0.4 +
                self.text_f1_score * 0.4 +
                self.sentence_overlap * 0.1 +
                self.paragraph_overlap * 0.1;

        // Title similarity component (20%)
        self.title_similarity_score = self.title_jaccard_similarity * 0.5 +
                self.title_match_score * 0.5;

        // Existing quality component (40%)
        self.existing_quality_score = existing_quality;

        // Final weighted combination
        self.combined_quality = self.text_similarity_score * 0.40 +
                self.title_similarity_score * 0.20 +
                self.existing_quality_score * 0.40;

        self.combined_quality = self.combined_quality.clamp(0.0, 1.0);
    }
}

/// Evaluator for comparing extracted text against ground truth
pub struct GroundTruthEvaluator {
    stopwords: HashSet<String>,
}

impl GroundTruthEvaluator {
    /// Create new evaluator
    pub fn new(stopwords: HashSet<String>) -> Self {
        Self { stopwords }
    }

    /// Evaluate extracted text against ground truth
    pub fn evaluate(
        &self,
        extracted_text: &str,
        extracted_title: Option<&str>,
        ground_truth: &GroundTruthData,
        existing_quality: f32,
    ) -> EvaluationMetrics {
        let gt_text = ground_truth.get_text();
        let gt_title = ground_truth.get_title();

        // Tokenize texts
        let extracted_words = self.tokenize_and_normalize(extracted_text);
        let gt_words = self.tokenize_and_normalize(gt_text);

        // Calculate text Jaccard similarity
        let text_jaccard_similarity = self.calculate_jaccard_similarity(&extracted_words, &gt_words);

        // Calculate precision and recall
        let text_precision = if extracted_words.is_empty() {
            0.0
        } else {
            let intersection: HashSet<_> = extracted_words.intersection(&gt_words).collect();
            intersection.len() as f32 / extracted_words.len() as f32
        };

        let text_recall = if gt_words.is_empty() {
            0.0
        } else {
            let intersection: HashSet<_> = extracted_words.intersection(&gt_words).collect();
            intersection.len() as f32 / gt_words.len() as f32
        };

        // Calculate F1 score
        let text_f1_score = if text_precision + text_recall == 0.0 {
            0.0
        } else {
            2.0 * text_precision * text_recall / (text_precision + text_recall)
        };

        // Length metrics
        let extracted_len = extracted_text.len();
        let gt_len = gt_text.len();
        let length_ratio = if gt_len == 0 {
            0.0
        } else {
            extracted_len as f32 / gt_len as f32
        };
        let length_difference = (extracted_len as i32 - gt_len as i32).abs();

        // Sentence and paragraph overlap
        let extracted_sentences = TextUtils::split_sentences(extracted_text);
        let gt_sentences = TextUtils::split_sentences(gt_text);
        let sentence_overlap = if gt_sentences.is_empty() {
            0.0
        } else {
            (extracted_sentences.len().min(gt_sentences.len()) as f32) /
                (gt_sentences.len() as f32)
        };

        let extracted_paragraphs = extracted_text.split("\n\n").filter(|p| !p.trim().is_empty()).count();
        let gt_paragraphs = gt_text.split("\n\n").filter(|p| !p.trim().is_empty()).count();
        let paragraph_overlap = if gt_paragraphs == 0 {
            0.0
        } else {
            (extracted_paragraphs.min(gt_paragraphs) as f32) / (gt_paragraphs as f32)
        };

        // Title matching
        let (title_jaccard_similarity, title_match_score) = if let Some(ext_title) = extracted_title {
            self.calculate_title_metrics(ext_title, gt_title)
        } else {
            (0.0, 0.0)
        };

        // Create metrics
        let mut metrics = EvaluationMetrics {
            text_jaccard_similarity,
            text_precision,
            text_recall,
            text_f1_score,
            length_ratio,
            length_difference,
            sentence_overlap,
            paragraph_overlap,
            title_jaccard_similarity,
            title_match_score,
            text_similarity_score: 0.0,
            title_similarity_score: 0.0,
            existing_quality_score: existing_quality,
            combined_quality: 0.0,
        };

        metrics.calculate_combined_quality(existing_quality);

        metrics
    }

    /// Calculate Jaccard similarity between two sets
    fn calculate_jaccard_similarity(&self, set1: &HashSet<String>, set2: &HashSet<String>) -> f32 {
        if set1.is_empty() && set2.is_empty() {
            return 1.0;
        }

        let intersection: HashSet<_> = set1.intersection(set2).collect();
        let union: HashSet<_> = set1.union(set2).collect();

        if union.is_empty() {
            0.0
        } else {
            intersection.len() as f32 / union.len() as f32
        }
    }

    /// Tokenize and normalize text (lowercase, remove stopwords)
    fn tokenize_and_normalize(&self, text: &str) -> HashSet<String> {
        TextUtils::tokenize(text)
            .into_iter()
            .filter(|word| !self.stopwords.contains(word) && word.len() > 2)
            .collect()
    }

    /// Calculate title metrics (Jaccard similarity and match score)
    fn calculate_title_metrics(&self, extracted: &str, ground_truth: &str) -> (f32, f32) {
        if ground_truth.is_empty() {
            return (0.5, 0.5); // No ground truth to compare
        }

        if extracted.is_empty() {
            return (0.0, 0.0);
        }

        // Tokenize titles
        let extracted_words: HashSet<_> = TextUtils::tokenize(extracted)
            .into_iter()
            .filter(|w| w.len() > 2)
            .collect();
        let gt_words: HashSet<_> = TextUtils::tokenize(ground_truth)
            .into_iter()
            .filter(|w| w.len() > 2)
            .collect();

        if gt_words.is_empty() {
            return (0.5, 0.5);
        }

        // Jaccard similarity
        let jaccard = self.calculate_jaccard_similarity(&extracted_words, &gt_words);

        // Calculate F1 score for title
        let intersection = extracted_words.intersection(&gt_words).count();
        let recall = intersection as f32 / gt_words.len() as f32;
        let precision = if extracted_words.is_empty() {
            0.0
        } else {
            intersection as f32 / extracted_words.len() as f32
        };

        let f1_score = if recall + precision == 0.0 {
            0.0
        } else {
            2.0 * recall * precision / (recall + precision)
        };

        (jaccard, f1_score)
    }

    /// Evaluate batch of extractions
    pub fn evaluate_batch(
        &self,
        extractions: Vec<(String, Option<String>, &GroundTruthData, f32)>,
    ) -> Vec<EvaluationMetrics> {
        extractions
            .into_iter()
            .map(|(text, title, gt, quality)| {
                self.evaluate(&text, title.as_deref(), gt, quality)
            })
            .collect()
    }

    /// Calculate average metrics across batch
    pub fn average_metrics(metrics: &[EvaluationMetrics]) -> EvaluationMetrics {
        if metrics.is_empty() {
            return EvaluationMetrics {
                text_jaccard_similarity: 0.0,
                text_precision: 0.0,
                text_recall: 0.0,
                text_f1_score: 0.0,
                length_ratio: 0.0,
                length_difference: 0,
                sentence_overlap: 0.0,
                paragraph_overlap: 0.0,
                title_jaccard_similarity: 0.0,
                title_match_score: 0.0,
                text_similarity_score: 0.0,
                title_similarity_score: 0.0,
                existing_quality_score: 0.0,
                combined_quality: 0.0,
            };
        }

        let n = metrics.len() as f32;

        EvaluationMetrics {
            text_jaccard_similarity: metrics.iter().map(|m| m.text_jaccard_similarity).sum::<f32>() / n,
            text_precision: metrics.iter().map(|m| m.text_precision).sum::<f32>() / n,
            text_recall: metrics.iter().map(|m| m.text_recall).sum::<f32>() / n,
            text_f1_score: metrics.iter().map(|m| m.text_f1_score).sum::<f32>() / n,
            length_ratio: metrics.iter().map(|m| m.length_ratio).sum::<f32>() / n,
            length_difference: (metrics.iter().map(|m| m.length_difference).sum::<i32>() as f32 / n) as i32,
            sentence_overlap: metrics.iter().map(|m| m.sentence_overlap).sum::<f32>() / n,
            paragraph_overlap: metrics.iter().map(|m| m.paragraph_overlap).sum::<f32>() / n,
            title_jaccard_similarity: metrics.iter().map(|m| m.title_jaccard_similarity).sum::<f32>() / n,
            title_match_score: metrics.iter().map(|m| m.title_match_score).sum::<f32>() / n,
            text_similarity_score: metrics.iter().map(|m| m.text_similarity_score).sum::<f32>() / n,
            title_similarity_score: metrics.iter().map(|m| m.title_similarity_score).sum::<f32>() / n,
            existing_quality_score: metrics.iter().map(|m| m.existing_quality_score).sum::<f32>() / n,
            combined_quality: metrics.iter().map(|m| m.combined_quality).sum::<f32>() / n,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluation() {
        let stopwords: HashSet<String> = vec!["the", "a", "is"]
            .into_iter()
            .map(|s| s.to_string())
            .collect();

        let evaluator = GroundTruthEvaluator::new(stopwords);

        let gt = GroundTruthData {
            data_type: Some("news".to_string()),
            data_key: None,
            fetch_timestamp: None,
            session_id: None,
            mod_date: None,
            title: Some("Test Article Title".to_string()),
            text: Some("This is the ground truth article text with several sentences. It contains important information.".to_string()),
            url: Some("https://example.com/article".to_string()),
            pub_date: None,
            pubdate: Some("2025-01-01".to_string()),
            author: None,
            source_name: None,
            language: Some("en".to_string()),
            keywords: None,
            industries: None,
            unique_id: None,
            module: None,
        };

        let extracted = "This is the extracted article text with several sentences.";
        let title = Some("Test Article");

        let metrics = evaluator.evaluate(extracted, title, &gt, 0.8);

        assert!(metrics.text_f1_score > 0.0);
        assert!(metrics.combined_quality > 0.0);
        assert!(metrics.title_match_score > 0.0);
        assert_eq!(metrics.existing_quality_score, 0.8);
    }

    #[test]
    fn test_jaccard_similarity() {
        let stopwords: HashSet<String> = HashSet::new();
        let evaluator = GroundTruthEvaluator::new(stopwords);

        let set1: HashSet<String> = vec!["hello", "world"]
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        let set2: HashSet<String> = vec!["hello", "world", "test"]
            .into_iter()
            .map(|s| s.to_string())
            .collect();

        let similarity = evaluator.calculate_jaccard_similarity(&set1, &set2);
        assert!((similarity - 0.666).abs() < 0.01);
    }
}