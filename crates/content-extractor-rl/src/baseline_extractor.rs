// ============================================================================
// FILE: crates/content-extractor-rl/src/baseline_extractor.rs
// ============================================================================

use scraper::{Html, Selector, ElementRef};
use crate::text_utils::TextUtils;
use crate::html_parser::HtmlParser;
use crate::site_profile::ExtractionResult;
use crate::Result;
use std::collections::HashSet;
use chrono::{NaiveDate, NaiveDateTime};
use regex::Regex;

/// Baseline content extractor rl using heuristics
#[derive(Clone)]
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
        // Extract metadata first
        let title = MetadataExtractor::extract_title(html);
        let date = MetadataExtractor::extract_date(html);

        let document = HtmlParser::clean_html(html)?;
        let candidates = self.get_candidates(&document);

        if candidates.is_empty() {
            return Ok(ExtractionResult {
                text: String::new(),
                xpath: String::new(),
                quality_score: 0.0,
                parameters: std::collections::HashMap::new(),
                title,
                date,
            });
        }

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
            title,
            date,
        })
    }

    /// Get candidate nodes with scores
    fn get_candidates<'a>(&self, document: &'a Html) -> Vec<(ElementRef<'a>, f64)> {
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

                true
            })
            .collect();

        filtered.join("\n\n")
    }

    /// Get candidate nodes for environment
    pub fn get_candidate_nodes<'a>(&self, document: &'a Html, top_k: usize) -> Vec<ElementRef<'a>> {
        self.get_candidates(document)
            .into_iter()
            .take(top_k)
            .map(|(node, _)| node)
            .collect()
    }
}


/// Extract metadata (title, date, author) from HTML
pub struct MetadataExtractor;

impl MetadataExtractor {
    /// Extract title from HTML
    pub fn extract_title(html: &str) -> Option<String> {
        let document = Html::parse_document(html);

        // Try multiple strategies in order of preference

        // 1. OpenGraph meta tag
        if let Some(title) = Self::extract_meta_tag(&document, "og:title") {
            return Some(title);
        }

        // 2. Twitter card meta tag
        if let Some(title) = Self::extract_meta_tag(&document, "twitter:title") {
            return Some(title);
        }

        // 3. Article title meta tag
        if let Some(title) = Self::extract_meta_tag(&document, "article:title") {
            return Some(title);
        }

        // 4. Standard <title> tag
        if let Ok(selector) = Selector::parse("title") {
            if let Some(title_elem) = document.select(&selector).next() {
                let title = title_elem.text().collect::<String>().trim().to_string();
                if !title.is_empty() {
                    return Some(Self::clean_title(&title));
                }
            }
        }

        // 5. h1 tag (often the article title)
        if let Ok(selector) = Selector::parse("h1") {
            if let Some(h1_elem) = document.select(&selector).next() {
                let title = h1_elem.text().collect::<String>().trim().to_string();
                if !title.is_empty() && title.len() > 10 {
                    return Some(title);
                }
            }
        }

        // 6. article > header > h1
        if let Ok(selector) = Selector::parse("article header h1, article h1") {
            if let Some(elem) = document.select(&selector).next() {
                let title = elem.text().collect::<String>().trim().to_string();
                if !title.is_empty() && title.len() > 10 {
                    return Some(title);
                }
            }
        }

        None
    }

    /// Extract publication date from HTML
    pub fn extract_date(html: &str) -> Option<String> {
        let document = Html::parse_document(html);

        // Try multiple strategies

        // 1. OpenGraph meta tag
        if let Some(date) = Self::extract_meta_tag(&document, "article:published_time") {
            if let Some(normalized) = Self::normalize_date(&date) {
                return Some(normalized);
            }
        }

        // 2. Schema.org meta tags
        if let Some(date) = Self::extract_meta_tag(&document, "datePublished") {
            if let Some(normalized) = Self::normalize_date(&date) {
                return Some(normalized);
            }
        }

        // 3. Standard meta tags
        for name in &["pubdate", "publishdate", "date", "DC.date"] {
            if let Some(date) = Self::extract_meta_tag(&document, name) {
                if let Some(normalized) = Self::normalize_date(&date) {
                    return Some(normalized);
                }
            }
        }

        // 4. time tag with datetime attribute
        if let Ok(selector) = Selector::parse("time[datetime], time[pubdate]") {
            if let Some(time_elem) = document.select(&selector).next() {
                if let Some(datetime) = time_elem.value().attr("datetime")
                    .or_else(|| time_elem.value().attr("pubdate")) {
                    if let Some(normalized) = Self::normalize_date(datetime) {
                        return Some(normalized);
                    }
                }
            }
        }

        // 5. Common date patterns in text
        if let Some(date) = Self::extract_date_from_text(html) {
            return Some(date);
        }

        None
    }

    /// Extract meta tag content
    fn extract_meta_tag(document: &Html, property: &str) -> Option<String> {
        // Try property attribute
        let selector_str = format!("meta[property='{}']", property);
        if let Ok(selector) = Selector::parse(&selector_str) {
            if let Some(elem) = document.select(&selector).next() {
                if let Some(content) = elem.value().attr("content") {
                    return Some(content.to_string());
                }
            }
        }

        // Try name attribute
        let selector_str = format!("meta[name='{}']", property);
        if let Ok(selector) = Selector::parse(&selector_str) {
            if let Some(elem) = document.select(&selector).next() {
                if let Some(content) = elem.value().attr("content") {
                    return Some(content.to_string());
                }
            }
        }

        None
    }

    /// Clean title by removing site name suffixes
    fn clean_title(title: &str) -> String {
        // Common separators between title and site name
        let separators = [" - ", " | ", " – ", " — ", " :: ", " » "];

        for sep in &separators {
            if let Some(pos) = title.rfind(sep) {
                let cleaned = &title[..pos];
                if cleaned.len() > 10 {
                    return cleaned.trim().to_string();
                }
            }
        }

        title.trim().to_string()
    }

    /// Normalize date to ISO 8601 format
    fn normalize_date(date_str: &str) -> Option<String> {
        // Already in ISO format
        if date_str.contains('T') || date_str.contains("Z") {
            return Some(date_str.to_string());
        }

        // Try parsing common formats
        let formats = [
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%B %d, %Y",
            "%b %d, %Y",
            "%d %B %Y",
            "%d %b %Y",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
        ];

        for format in &formats {
            if let Ok(parsed) = NaiveDate::parse_from_str(date_str, format) {
                return Some(parsed.format("%Y-%m-%d").to_string());
            }
            if let Ok(parsed) = NaiveDateTime::parse_from_str(date_str, format) {
                return Some(parsed.format("%Y-%m-%d").to_string());
            }
        }

        None
    }

    /// Extract date from common text patterns
    fn extract_date_from_text(html: &str) -> Option<String> {
        lazy_static::lazy_static! {
            static ref DATE_PATTERNS: Vec<Regex> = vec![
                // ISO format: 2021-04-05
                Regex::new(r"(\d{4}-\d{2}-\d{2})").unwrap(),
                // US format: April 5, 2021
                Regex::new(r"([A-Z][a-z]+ \d{1,2}, \d{4})").unwrap(),
                // European: 5 April 2021
                Regex::new(r"(\d{1,2} [A-Z][a-z]+ \d{4})").unwrap(),
            ];
        }

        for pattern in DATE_PATTERNS.iter() {
            if let Some(captures) = pattern.captures(html) {
                if let Some(matched) = captures.get(1) {
                    if let Some(normalized) = Self::normalize_date(matched.as_str()) {
                        return Some(normalized);
                    }
                }
            }
        }

        None
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_title_from_og_tag() {
        let html = r#"
            <html>
                <head>
                    <meta property="og:title" content="Test Article Title" />
                </head>
            </html>
        "#;

        let title = MetadataExtractor::extract_title(html);
        assert_eq!(title, Some("Test Article Title".to_string()));
    }

    #[test]
    fn test_extract_title_from_title_tag() {
        let html = r#"
            <html>
                <head>
                    <title>Test Article - Site Name</title>
                </head>
            </html>
        "#;

        let title = MetadataExtractor::extract_title(html);
        assert_eq!(title, Some("Test Article".to_string()));
    }

    #[test]
    fn test_extract_date_from_meta() {
        let html = r#"
            <html>
                <head>
                    <meta property="article:published_time" content="2021-04-05T10:30:00Z" />
                </head>
            </html>
        "#;

        let date = MetadataExtractor::extract_date(html);
        assert!(date.is_some());
    }

    #[test]
    fn test_normalize_date() {
        assert_eq!(
            MetadataExtractor::normalize_date("2021-04-05"),
            Some("2021-04-05".to_string())
        );

        assert_eq!(
            MetadataExtractor::normalize_date("April 5, 2021"),
            Some("2021-04-05".to_string())
        );
    }

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
