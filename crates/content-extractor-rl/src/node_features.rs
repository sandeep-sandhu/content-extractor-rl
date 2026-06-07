// ============================================================================
// FILE: crates/content-extractor-rl/src/node_features.rs
// ============================================================================
//! Real, content-aware features for candidate DOM nodes.
//!
//! These replace the placeholder constant state vector that previously made the
//! RL agent blind to the document. Every feature is derived from the actual DOM
//! subtree of a candidate node, so two different candidates produce two
//! different feature vectors — a precondition for the agent to learn anything.
//!
//! The same features power the supervised node classifier (hybrid mode), so the
//! representation is shared in one place.

use scraper::{ElementRef, Selector};
use std::collections::HashSet;

/// Continuous extraction parameters that the RL policy tunes. They actually
/// affect which text blocks are kept, so the policy's continuous head has a
/// real effect on the extracted text (and therefore on the reward).
#[derive(Debug, Clone, Copy)]
pub struct ExtractionParams {
    /// Minimum words for a block (<p>/text node) to be kept.
    pub min_block_words: usize,
    /// Drop blocks whose link density exceeds this threshold (0..=1).
    pub max_block_link_density: f32,
}

impl Default for ExtractionParams {
    fn default() -> Self {
        Self { min_block_words: 5, max_block_link_density: 0.5 }
    }
}

impl ExtractionParams {
    /// Map a policy's normalized continuous params (each roughly in [-1, 1]) to
    /// concrete extraction settings. Only the first two params are used today;
    /// extra params are accepted and ignored so the action space can stay wide.
    pub fn from_normalized(params: &[f32]) -> Self {
        let p0 = params.first().copied().unwrap_or(0.0).clamp(-1.0, 1.0);
        let p1 = params.get(1).copied().unwrap_or(0.0).clamp(-1.0, 1.0);
        // min_block_words in [1, 40]
        let min_block_words = (1.0 + (p0 + 1.0) * 19.5).round().clamp(1.0, 40.0) as usize;
        // max_block_link_density in [0.1, 0.9]
        let max_block_link_density = (0.1 + (p1 + 1.0) * 0.4).clamp(0.1, 0.9);
        Self { min_block_words, max_block_link_density }
    }
}

/// Structural / textual features for a single candidate node.
///
/// All fields are pre-normalized to roughly [0, 1] so they can be fed directly
/// to the network without further scaling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodeFeatures {
    pub word_count_norm: f32,
    pub char_count_norm: f32,
    pub link_density: f32,
    pub stopword_ratio: f32,
    pub p_count_norm: f32,
    pub text_tag_ratio: f32,
    pub depth_norm: f32,
    pub comma_density: f32,
    pub tag_article: f32,
    pub tag_main: f32,
    pub tag_section: f32,
    pub tag_div: f32,
    pub tag_other: f32,
    pub class_positive: f32,
    pub class_negative: f32,
    pub unique_word_ratio: f32,
}

impl NodeFeatures {
    /// Number of features per node. Kept in sync with [`Self::to_vec`].
    pub const DIM: usize = 16;

    /// Zeroed features (used to pad unused candidate slots).
    pub fn zeros() -> Self {
        Self {
            word_count_norm: 0.0,
            char_count_norm: 0.0,
            link_density: 0.0,
            stopword_ratio: 0.0,
            p_count_norm: 0.0,
            text_tag_ratio: 0.0,
            depth_norm: 0.0,
            comma_density: 0.0,
            tag_article: 0.0,
            tag_main: 0.0,
            tag_section: 0.0,
            tag_div: 0.0,
            tag_other: 0.0,
            class_positive: 0.0,
            class_negative: 0.0,
            unique_word_ratio: 0.0,
        }
    }

    /// Flatten to a fixed-length vector (length == [`Self::DIM`]).
    pub fn to_vec(&self) -> Vec<f32> {
        vec![
            self.word_count_norm,
            self.char_count_norm,
            self.link_density,
            self.stopword_ratio,
            self.p_count_norm,
            self.text_tag_ratio,
            self.depth_norm,
            self.comma_density,
            self.tag_article,
            self.tag_main,
            self.tag_section,
            self.tag_div,
            self.tag_other,
            self.class_positive,
            self.class_negative,
            self.unique_word_ratio,
        ]
    }

    /// A heuristic "is this the article body?" score in [0, 1], derived purely
    /// from the features. Used as a warm-start prior and as a baseline for the
    /// supervised classifier's tests. Not learned — just a sane linear combo of
    /// Readability-style signals.
    pub fn heuristic_content_score(&self) -> f32 {
        let mut s = 0.0;
        s += 0.40 * self.word_count_norm;
        s += 0.20 * self.p_count_norm;
        s += 0.15 * (1.0 - self.link_density);
        s += 0.10 * self.class_positive;
        s += 0.10 * self.tag_article;
        s += 0.05 * self.tag_main;
        s -= 0.40 * self.class_negative;
        s -= 0.20 * self.link_density;
        s.clamp(0.0, 1.0)
    }
}

/// Positive (content-ish) class/id substrings, à la Readability/arc90.
const POSITIVE_HINTS: &[&str] = &[
    "article", "content", "post", "story", "body", "entry", "main", "text",
    "blog", "page",
];

/// Negative (boilerplate) class/id substrings.
const NEGATIVE_HINTS: &[&str] = &[
    "comment", "sidebar", "footer", "header", "nav", "menu", "ad", "advert",
    "promo", "share", "social", "related", "widget", "banner", "popup",
    "cookie", "newsletter", "subscribe", "breadcrumb",
];

fn p_selector() -> Selector {
    Selector::parse("p").unwrap()
}

fn a_selector() -> Selector {
    Selector::parse("a").unwrap()
}

/// Concatenate the text content of an element, collapsing whitespace.
pub fn node_text(el: &ElementRef) -> String {
    let raw: String = el.text().collect::<Vec<_>>().join(" ");
    raw.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// A single block of text (one `<p>`) with the stats needed to filter it.
#[derive(Debug, Clone)]
pub struct TextBlock {
    pub text: String,
    pub words: usize,
    pub link_density: f32,
}

/// Self-contained, owned snapshot of a candidate node's extractable text.
///
/// It holds the per-paragraph blocks plus a whole-node fallback, so the
/// environment can re-extract under *different* policy params on later steps
/// without keeping borrowed `ElementRef`s across calls (which Rust's borrow
/// checker forbids when the document is owned by the same struct).
#[derive(Debug, Clone)]
pub struct CandidateContent {
    pub blocks: Vec<TextBlock>,
    pub full_text: String,
    pub full_link_density: f32,
}

impl CandidateContent {
    /// Apply extraction params to produce the article text. Blocks are kept
    /// only if they meet the minimum word count and stay below the link-density
    /// threshold; if none survive we fall back to the whole-node text when it is
    /// not link-dominated.
    pub fn extract(&self, params: &ExtractionParams) -> String {
        let kept: Vec<&str> = self
            .blocks
            .iter()
            .filter(|b| b.words >= params.min_block_words && b.link_density <= params.max_block_link_density)
            .map(|b| b.text.as_str())
            .collect();

        if kept.is_empty() {
            if self.full_link_density <= params.max_block_link_density
                && self.full_text.split_whitespace().count() >= params.min_block_words
            {
                return self.full_text.clone();
            }
            return String::new();
        }

        kept.join("\n\n")
    }
}

/// Build the owned [`CandidateContent`] snapshot for a node.
pub fn node_content(el: &ElementRef) -> CandidateContent {
    let p_sel = p_selector();
    let blocks: Vec<TextBlock> = el
        .select(&p_sel)
        .map(|p| {
            let text = node_text(&p);
            let words = text.split_whitespace().count();
            TextBlock { words, link_density: link_density(&p), text }
        })
        .collect();

    CandidateContent {
        blocks,
        full_text: node_text(el),
        full_link_density: link_density(el),
    }
}

/// Extract article text from a node, honoring the policy's extraction params.
///
/// Convenience wrapper over [`node_content`] + [`CandidateContent::extract`].
pub fn extract_node_text(el: &ElementRef, params: &ExtractionParams) -> String {
    node_content(el).extract(params)
}

/// Fraction of characters inside `<a>` descendants relative to total text.
pub fn link_density(el: &ElementRef) -> f32 {
    let total = node_text(el).chars().count();
    if total == 0 {
        return 0.0;
    }
    let a_sel = a_selector();
    let link_chars: usize = el
        .select(&a_sel)
        .map(|a| node_text(&a).chars().count())
        .sum();
    (link_chars as f32 / total as f32).clamp(0.0, 1.0)
}

fn class_id_hint_scores(el: &ElementRef) -> (f32, f32) {
    let mut haystack = String::new();
    if let Some(c) = el.value().attr("class") {
        haystack.push_str(&c.to_lowercase());
        haystack.push(' ');
    }
    if let Some(id) = el.value().attr("id") {
        haystack.push_str(&id.to_lowercase());
    }
    if haystack.is_empty() {
        return (0.0, 0.0);
    }
    let pos = POSITIVE_HINTS.iter().filter(|h| haystack.contains(**h)).count();
    let neg = NEGATIVE_HINTS.iter().filter(|h| haystack.contains(**h)).count();
    // Squash counts into [0, 1] — presence matters more than exact count.
    let pos_score = (pos as f32 / 2.0).clamp(0.0, 1.0);
    let neg_score = (neg as f32 / 2.0).clamp(0.0, 1.0);
    (pos_score, neg_score)
}

fn node_depth(el: &ElementRef) -> usize {
    let mut depth = 0;
    let mut current = Some(*el);
    while let Some(e) = current {
        depth += 1;
        current = e.parent().and_then(ElementRef::wrap);
    }
    depth
}

/// Compute the full feature set for a candidate node.
pub fn extract_features(el: &ElementRef, stopwords: &HashSet<String>) -> NodeFeatures {
    let text = node_text(el);
    let tokens: Vec<&str> = text.split_whitespace().collect();
    let word_count = tokens.len();
    let char_count = text.chars().count();

    let word_count_norm = if word_count == 0 {
        0.0
    } else {
        ((word_count as f32 + 1.0).ln() / (5000f32).ln()).clamp(0.0, 1.0)
    };
    let char_count_norm = ((char_count as f32 + 1.0).ln() / (40000f32).ln()).clamp(0.0, 1.0);

    let link_density = link_density(el);

    let stopword_ratio = if word_count == 0 {
        0.0
    } else {
        let sw = tokens
            .iter()
            .filter(|t| stopwords.contains(&t.to_lowercase()))
            .count();
        (sw as f32 / word_count as f32).clamp(0.0, 1.0)
    };

    let p_count = el.select(&p_selector()).count();
    let p_count_norm = (p_count as f32 / 30.0).clamp(0.0, 1.0);

    let element_count = el
        .descendants()
        .filter(|n| n.value().is_element())
        .count()
        .max(1);
    let text_tag_ratio = ((word_count as f32 / element_count as f32) / 50.0).clamp(0.0, 1.0);

    let depth_norm = (node_depth(el) as f32 / 30.0).clamp(0.0, 1.0);

    let comma_count = text.chars().filter(|c| *c == ',').count();
    let comma_density = if word_count == 0 {
        0.0
    } else {
        ((comma_count as f32 / word_count as f32) / 0.2).clamp(0.0, 1.0)
    };

    let tag = el.value().name().to_lowercase();
    let (tag_article, tag_main, tag_section, tag_div, tag_other) = match tag.as_str() {
        "article" => (1.0, 0.0, 0.0, 0.0, 0.0),
        "main" => (0.0, 1.0, 0.0, 0.0, 0.0),
        "section" => (0.0, 0.0, 1.0, 0.0, 0.0),
        "div" => (0.0, 0.0, 0.0, 1.0, 0.0),
        _ => (0.0, 0.0, 0.0, 0.0, 1.0),
    };

    let (class_positive, class_negative) = class_id_hint_scores(el);

    let unique_word_ratio = if word_count == 0 {
        0.0
    } else {
        let unique: HashSet<String> = tokens.iter().map(|t| t.to_lowercase()).collect();
        (unique.len() as f32 / word_count as f32).clamp(0.0, 1.0)
    };

    NodeFeatures {
        word_count_norm,
        char_count_norm,
        link_density,
        stopword_ratio,
        p_count_norm,
        text_tag_ratio,
        depth_norm,
        comma_density,
        tag_article,
        tag_main,
        tag_section,
        tag_div,
        tag_other,
        class_positive,
        class_negative,
        unique_word_ratio,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scraper::Html;

    fn stopwords() -> HashSet<String> {
        ["the", "a", "is", "of", "and", "to", "in", "with", "this", "for"]
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    }

    fn first_matching<'a>(doc: &'a Html, sel: &str) -> ElementRef<'a> {
        doc.select(&Selector::parse(sel).unwrap()).next().unwrap()
    }

    #[test]
    fn to_vec_len_matches_dim() {
        assert_eq!(NodeFeatures::zeros().to_vec().len(), NodeFeatures::DIM);
    }

    #[test]
    fn content_node_scores_higher_than_boilerplate() {
        let html = r#"
            <html><body>
                <article class="post-content">
                    <p>This is the real article body with a lot of meaningful text content.</p>
                    <p>It has multiple paragraphs describing important information in detail here.</p>
                    <p>Readers expect substantial prose and varied vocabulary throughout the piece.</p>
                </article>
                <div class="sidebar-ads">
                    <a href="/1">Link one</a> <a href="/2">Link two</a> <a href="/3">Link three</a>
                </div>
            </body></html>
        "#;
        let doc = Html::parse_document(html);
        let sw = stopwords();

        let article = first_matching(&doc, "article");
        let sidebar = first_matching(&doc, "div");

        let art_feat = extract_features(&article, &sw);
        let side_feat = extract_features(&sidebar, &sw);

        // Article has real prose; sidebar is link-dominated boilerplate.
        assert!(art_feat.link_density < side_feat.link_density);
        assert!(art_feat.word_count_norm > side_feat.word_count_norm);
        assert!(art_feat.class_positive > 0.0);
        assert!(side_feat.class_negative > 0.0);
        assert!(
            art_feat.heuristic_content_score() > side_feat.heuristic_content_score(),
            "article {} should beat sidebar {}",
            art_feat.heuristic_content_score(),
            side_feat.heuristic_content_score()
        );
    }

    #[test]
    fn different_nodes_yield_different_features() {
        let html = r#"
            <html><body>
                <article><p>Alpha beta gamma delta epsilon zeta eta theta iota kappa.</p></article>
                <div class="nav"><a href="/x">x</a></div>
            </body></html>
        "#;
        let doc = Html::parse_document(html);
        let sw = stopwords();
        let a = extract_features(&first_matching(&doc, "article"), &sw);
        let d = extract_features(&first_matching(&doc, "div"), &sw);
        assert_ne!(a.to_vec(), d.to_vec(), "distinct nodes must have distinct features");
    }

    #[test]
    fn extraction_params_change_extracted_text() {
        let html = r#"
            <html><body>
                <article>
                    <p>Short one.</p>
                    <p>This paragraph is clearly long enough to survive a high minimum word threshold filter.</p>
                </article>
            </body></html>
        "#;
        let doc = Html::parse_document(html);
        let article = first_matching(&doc, "article");

        let lenient = ExtractionParams { min_block_words: 1, max_block_link_density: 0.9 };
        let strict = ExtractionParams { min_block_words: 6, max_block_link_density: 0.9 };

        let lenient_text = extract_node_text(&article, &lenient);
        let strict_text = extract_node_text(&article, &strict);

        // The strict filter must drop the short paragraph -> different output.
        assert!(lenient_text.contains("Short one"));
        assert!(!strict_text.contains("Short one"));
        assert_ne!(lenient_text, strict_text);
    }

    #[test]
    fn normalized_params_map_into_range() {
        let lo = ExtractionParams::from_normalized(&[-1.0, -1.0]);
        let hi = ExtractionParams::from_normalized(&[1.0, 1.0]);
        assert!(lo.min_block_words >= 1);
        assert!(hi.min_block_words <= 40);
        assert!(lo.max_block_link_density >= 0.1);
        assert!(hi.max_block_link_density <= 0.9);
        assert!(hi.min_block_words > lo.min_block_words);
    }
}
