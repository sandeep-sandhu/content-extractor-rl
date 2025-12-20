use scraper::{Html, Selector, ElementRef};
use crate::Result;
use std::collections::HashMap;

/// HTML parsing and DOM manipulation utilities
pub struct HtmlParser;

impl HtmlParser {
    /// Parse HTML string into document
    pub fn parse(html: &str) -> Result<Html> {
        Ok(Html::parse_document(html))
    }

    /// Extract text content from element
    pub fn extract_text(element: ElementRef) -> String {
        element.text().collect::<Vec<_>>().join(" ")
    }

    /// Get XPath-like selector for element
    pub fn get_element_path(element: ElementRef) -> String {
        let mut path = Vec::new();
        let mut current = Some(element);

        while let Some(elem) = current {
            let tag = elem.value().name();

            // Get position among siblings
            let position = elem.prev_siblings()
                .filter(|s| s.value().as_element().map_or(false, |e| e.name() == tag))
                .count() + 1;

            path.push(format!("{}[{}]", tag, position));
            current = elem.parent().and_then(|p| ElementRef::wrap(p));
        }

        path.reverse();
        format!("/{}", path.join("/"))
    }

    /// Clean HTML by removing script, style, etc.
    pub fn clean_html(html: &str) -> Result<Html> {
        let document = Html::parse_document(html);

        // Create cleaned HTML string (simplified - proper cleaning would modify DOM)
        let mut cleaned = html.to_string();

        // Remove script tags
        let script_selector = Selector::parse("script").unwrap();
        for element in document.select(&script_selector) {
            if let Some(html) = element.html().get(0..100) {
                cleaned = cleaned.replace(html, "");
            }
        }

        // Remove style tags
        let style_selector = Selector::parse("style").unwrap();
        for element in document.select(&style_selector) {
            if let Some(html) = element.html().get(0..100) {
                cleaned = cleaned.replace(html, "");
            }
        }

        Ok(Html::parse_document(&cleaned))
    }

    /// Get candidate article nodes from document
    pub fn get_candidate_nodes(document: &Html, top_k: usize) -> Vec<ElementRef> {
        let mut candidates = Vec::new();

        // Try article tags first
        let article_selector = Selector::parse("article").unwrap();
        for element in document.select(&article_selector) {
            candidates.push(element);
        }

        // Try divs
        let div_selector = Selector::parse("div").unwrap();
        for element in document.select(&div_selector) {
            candidates.push(element);
        }

        // Try sections
        let section_selector = Selector::parse("section").unwrap();
        for element in document.select(&section_selector) {
            candidates.push(element);
        }

        candidates.truncate(top_k);
        candidates
    }

    /// Extract paragraphs from element
    pub fn extract_paragraphs(element: ElementRef) -> Vec<String> {
        let p_selector = Selector::parse("p").unwrap();

        element.select(&p_selector)
            .map(|p| Self::extract_text(p).trim().to_string())
            .filter(|text| !text.is_empty())
            .collect()
    }

    /// Get parent element
    pub fn get_parent(element: ElementRef) -> Option<ElementRef> {
        element.parent().and_then(ElementRef::wrap)
    }

    /// Get previous sibling element
    pub fn get_prev_sibling(element: ElementRef) -> Option<ElementRef> {
        element.prev_sibling_element()
    }

    /// Get next sibling element
    pub fn get_next_sibling(element: ElementRef) -> Option<ElementRef> {
        element.next_sibling_element()
    }

    /// Count child elements
    pub fn count_children(element: ElementRef) -> usize {
        element.children().filter(|n| n.value().is_element()).count()
    }

    /// Get tree depth
    pub fn get_tree_depth(document: &Html) -> usize {
        fn depth_recursive(element: ElementRef) -> usize {
            let children: Vec<_> = element.children()
                .filter_map(ElementRef::wrap)
                .collect();

            if children.is_empty() {
                1
            } else {
                1 + children.into_iter()
                    .map(depth_recursive)
                    .max()
                    .unwrap_or(0)
            }
        }

        document.root_element()
            .children()
            .filter_map(ElementRef::wrap)
            .map(depth_recursive)
            .max()
            .unwrap_or(0)
    }

    /// Get node depth in tree
    pub fn get_node_depth(element: ElementRef) -> usize {
        let mut depth = 0;
        let mut current = Some(element);

        while let Some(elem) = current {
            depth += 1;
            current = elem.parent().and_then(ElementRef::wrap);
        }

        depth
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_html() {
        let html = r#"<html><body><p>Hello World</p></body></html>"#;
        let doc = HtmlParser::parse(html).unwrap();
        assert!(doc.root_element().html().contains("Hello World"));
    }

    #[test]
    fn test_extract_paragraphs() {
        let html = r#"
            <article>
                <p>First paragraph.</p>
                <p>Second paragraph.</p>
            </article>
        "#;
        let doc = HtmlParser::parse(html).unwrap();
        let article = doc.select(&Selector::parse("article").unwrap()).next().unwrap();
        let paragraphs = HtmlParser::extract_paragraphs(article);
        assert_eq!(paragraphs.len(), 2);
    }
}
