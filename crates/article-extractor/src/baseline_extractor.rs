use scraper::{Html, Selector, ElementRef};
use crate::text_utils::TextUtils;
use crate::html_parser::HtmlParser;
use crate::site_profile::ExtractionResult;
use crate::Result;
use std::collections::HashSet;

/// Baseline article extractor using heuristics
pub struct BaselineExtractor {
    stopwords: HashSet<String>,
}

impl BaselineExtractor {
    /// Create new baseline extractor
    pub fn new(stopwords: HashSet<String>) -> Self {
        Self { stopwords }
    }

    /// Extract article from HTML
    pub fn extract(&self, html: &str) -> Result<ExtractionResult> {
        let document = HtmlParser::clean_html(html)?;
        let candidates = self.get_candidates(&document);

        if candidates.is_empty() {
            return Ok(ExtractionResult {
                text: String::new(),
                xpath: String::new(),
                quality_score: 0.0,
                parameters: std::collections::HashMap::new(),
            });
        }

        // Find best candidate
        let (best_node, _score) = candidates.into_iter()
            .max_by(|(_, score_a), (_, score_b)| {
                score_a.partial_cmp(score_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        let text = self.extract_text(best_node);
        let xpath = HtmlParser::get_element_path(best_node);
        let quality = TextUtils::calculate_text_quality(&text, &self.stopwords);

        Ok(ExtractionResult {
            text,
            xpath,
            quality_score: quality,
            parameters: std::collections::HashMap::new(),
        })
    }

    /// Get candidate nodes with scores
    fn get_candidates(&self, document: &Html) -> Vec<(ElementRef, f64)> {
        let mut candidates = Vec::new();

        // Try different selectors
        let selectors = vec!["article", "div", "section"];

        for selector_str in selectors {
            if let Ok(selector) = Selector::parse(selector_str) {
                for element in document.select(&selector) {
                    let score = self.score_node(element);
                    if score > 0.0 {
                        candidates.push((element, score));
                    }
                }
            }
        }

        // Sort by score and take top 10
        candidates.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(10);

        candidates
    }

    /// Score node using stopword density
    fn score_node(&self, node: ElementRef) -> f64 {
        let text = HtmlParser::extract_text(node);

        if text.len() < 50 {
            return 0.0;
        }

        // Base score: stopword count squared
        let stopword_count = TextUtils::count_stopwords(&text, &self.stopwords);
        let mut score = (stopword_count * stopword_count) as f64;

        // Boost for paragraphs
        let paragraphs = HtmlParser::extract_paragraphs(node);
        let paragraph_count = paragraphs.len().min(5);
        score *= 1.0 + 0.5 * paragraph_count as f64;

        // Penalty for high link density
        if let Ok(link_selector) = Selector::parse("a") {
            let link_text: String = node.select(&link_selector)
                .map(|a| HtmlParser::extract_text(a))
                .collect();

            if !text.is_empty() {
                let link_density = link_text.len() as f64 / text.len() as f64;
                if link_density > 0.5 {
                    score *= 1.0 - link_density;
                }
            }
        }

        score
    }

    /// Extract clean text from node
    fn extract_text(&self, node: ElementRef) -> String {
        let paragraphs = HtmlParser::extract_paragraphs(node);

        let filtered: Vec<String> = paragraphs.into_iter()
            .filter(|p| {
                let words: Vec<_> = p.split_whitespace().collect();

                // Minimum word threshold
                if words.len() < 4 {
                    return false;
                }

                // Link density check (simplified)
                true
            })
            .collect();

        filtered.join("\n\n")
    }

    /// Get candidate nodes for environment
    pub fn get_candidate_nodes(&self, document: &Html, top_k: usize) -> Vec<ElementRef> {
        self.get_candidates(document)
            .into_iter()
            .take(top_k)
            .map(|(node, _)| node)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_baseline_extractor() {
        let html = r#"
            <html>
                <body>
                    <article>
                        <h1>Test Article</h1>
                        <p>This is the first paragraph of the article.</p>
                        <p>This is the second paragraph with more content.</p>
                    </article>
                </body>
            </html>
        "#;

        let stopwords: HashSet<String> = vec!["the", "is", "of"]
            .into_iter()
            .map(|s| s.to_string())
            .collect();

        let extractor = BaselineExtractor::new(stopwords);
        let result = extractor.extract(html).unwrap();

        assert!(!result.text.is_empty());
        assert!(result.quality_score > 0.0);
    }
}