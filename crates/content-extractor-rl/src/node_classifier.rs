// ============================================================================
// FILE: crates/content-extractor-rl/src/node_classifier.rs
// ============================================================================
//! Supervised content-node classifier (the "hybrid" half of the system).
//!
//! Content extraction is, at heart, a supervised node-classification problem:
//! given the candidate DOM nodes of a page and labelled ground-truth article
//! text, learn which candidate is the article body. This is far more
//! sample-efficient and stable than asking RL to discover node selection from a
//! sparse reward. The division of labour is therefore:
//!
//!   * this classifier picks **which node** is the content root (supervised);
//!   * the RL policy tunes the **continuous extraction params** within it.
//!
//! Labels come for free from the data: the candidate whose extracted text has
//! the highest token-F1 against the ground-truth article is the positive
//! example, the rest are negatives ([`label_from_f1`]).

use candle_core::{DType, Device, Tensor};
use candle_nn::{linear, AdamW, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use candle_nn::ops::sigmoid;

use crate::html_parser::HtmlParser;
use crate::node_features::{self, CandidateContent, ExtractionParams, NodeFeatures};
use crate::text_utils::TextUtils;
use crate::training::TrainingSample;
use crate::{Config, ExtractionError, Result};
use std::collections::HashSet;
use std::path::Path;

/// A small MLP that maps [`NodeFeatures`] to a content-probability.
pub struct NodeClassifier {
    fc1: Linear,
    fc2: Linear,
    out: Linear,
    varmap: VarMap,
    optimizer: AdamW,
    device: Device,
}

impl NodeClassifier {
    /// Create a new, randomly-initialized classifier.
    pub fn new(device: &Device, lr: f64) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

        let hidden1 = 32;
        let hidden2 = 16;
        let fc1 = linear(NodeFeatures::DIM, hidden1, vb.pp("fc1"))?;
        let fc2 = linear(hidden1, hidden2, vb.pp("fc2"))?;
        let out = linear(hidden2, 1, vb.pp("out"))?;

        let params = ParamsAdamW { lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 1e-5 };
        let optimizer = AdamW::new(varmap.all_vars(), params)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        Ok(Self { fc1, fc2, out, varmap, optimizer, device: device.clone() })
    }

    /// Forward pass returning raw logits of shape `[batch]`.
    fn logits(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.fc1.forward(x)?.relu()?;
        let x = self.fc2.forward(&x)?.relu()?;
        self.out.forward(&x)?.squeeze(1)
    }

    fn features_to_tensor(&self, features: &[NodeFeatures]) -> candle_core::Result<Tensor> {
        let flat: Vec<f32> = features.iter().flat_map(|f| f.to_vec()).collect();
        Tensor::from_vec(flat, &[features.len(), NodeFeatures::DIM], &self.device)
    }

    /// Content probability in [0, 1] for each candidate.
    pub fn score_batch(&self, features: &[NodeFeatures]) -> Result<Vec<f32>> {
        if features.is_empty() {
            return Ok(Vec::new());
        }
        let x = self.features_to_tensor(features)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;
        let probs = sigmoid(&self.logits(&x).map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;
        probs.to_vec1::<f32>()
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))
    }

    /// Index of the candidate most likely to be the content root.
    pub fn select_best(&self, features: &[NodeFeatures]) -> Result<Option<usize>> {
        let scores = self.score_batch(features)?;
        Ok(scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i))
    }

    /// One supervised gradient step over a batch of `(features, label)` pairs,
    /// where `label` is 1.0 for the content node and 0.0 otherwise. Returns the
    /// binary cross-entropy loss.
    pub fn train_batch(&mut self, features: &[NodeFeatures], labels: &[f32]) -> Result<f32> {
        assert_eq!(features.len(), labels.len());
        if features.is_empty() {
            return Ok(0.0);
        }

        let x = self.features_to_tensor(features)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;
        let targets = Tensor::from_vec(labels.to_vec(), &[labels.len()], &self.device)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        let loss = self
            .bce_loss(&x, &targets)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;
        let loss_val = loss.to_scalar::<f32>()
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;
        if loss_val.is_nan() || loss_val.is_infinite() {
            return Ok(f32::NAN);
        }

        let grads = loss.backward()
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;
        self.optimizer.step(&grads)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;
        Ok(loss_val)
    }

    fn bce_loss(&self, x: &Tensor, targets: &Tensor) -> candle_core::Result<Tensor> {
        let p = sigmoid(&self.logits(x)?)?;
        // Clamp to avoid log(0).
        let p = p.clamp(1e-7f32, 1.0f32 - 1e-7f32)?;
        let log_p = p.log()?;
        let log_1mp = p.affine(-1.0, 1.0)?.log()?; // log(1 - p)
        let pos = targets.mul(&log_p)?;
        let neg = targets.affine(-1.0, 1.0)?.mul(&log_1mp)?; // (1 - y) * log(1 - p)
        (pos + neg)?.neg()?.mean_all()
    }

    /// Number of trainable parameters (for logging / metadata).
    pub fn num_parameters(&self) -> usize {
        self.varmap
            .all_vars()
            .iter()
            .map(|v| v.as_tensor().elem_count())
            .sum()
    }

    /// Serialize the classifier weights to a `.safetensors` file.
    pub fn save(&self, path: &Path) -> Result<()> {
        self.varmap
            .save(path)
            .map_err(|e| ExtractionError::ModelError(format!("classifier save failed: {e}")))
    }

    /// Load a classifier previously written with [`Self::save`].
    ///
    /// The architecture is fixed (so `new` recreates the same variables) and the
    /// stored tensors are copied into them; `lr` only affects subsequent
    /// training and is irrelevant for inference.
    pub fn load(path: &Path, device: &Device, lr: f64) -> Result<Self> {
        let mut clf = Self::new(device, lr)?;
        clf.varmap
            .load(path)
            .map_err(|e| ExtractionError::ModelError(format!("classifier load failed: {e}")))?;
        Ok(clf)
    }
}

/// Build a pointwise training set for the classifier from labelled samples.
///
/// For every sample that has ground-truth text, each candidate node becomes one
/// `(features, label)` example, where `label = 1.0` for the best-F1 candidate
/// (via [`label_from_f1`]) and `0.0` otherwise. Samples without ground truth, or
/// where no candidate matches, are skipped.
pub fn build_classifier_dataset(
    samples: &[TrainingSample],
    num_candidates: usize,
    stopwords: &HashSet<String>,
) -> (Vec<NodeFeatures>, Vec<f32>) {
    let mut features = Vec::new();
    let mut labels = Vec::new();

    for sample in samples {
        let Some(gt) = sample.ground_truth_text.as_deref() else { continue };
        let Ok(document) = HtmlParser::clean_html(&sample.html) else { continue };
        let candidates = HtmlParser::get_candidate_nodes(&document, num_candidates);
        if candidates.is_empty() {
            continue;
        }
        let contents: Vec<CandidateContent> =
            candidates.iter().map(node_features::node_content).collect();
        let Some(sample_labels) = label_from_f1(&contents, gt, stopwords) else { continue };

        for (candidate, label) in candidates.iter().zip(sample_labels) {
            features.push(node_features::extract_features(candidate, stopwords));
            labels.push(label);
        }
    }

    (features, labels)
}

/// Train a [`NodeClassifier`] on labelled samples for `epochs` full-batch steps.
///
/// Returns the trained classifier and the final BCE loss. Errors if no sample
/// carried usable ground truth.
pub fn train_classifier(
    samples: &[TrainingSample],
    config: &Config,
    epochs: usize,
    lr: f64,
    device: &Device,
) -> Result<(NodeClassifier, f32)> {
    let (features, labels) =
        build_classifier_dataset(samples, config.num_candidate_nodes, &config.stopwords);

    if features.is_empty() {
        return Err(ExtractionError::ModelError(
            "no labelled training examples for the classifier (samples need ground-truth text)"
                .to_string(),
        ));
    }

    let mut classifier = NodeClassifier::new(device, lr)?;
    let mut last_loss = f32::NAN;
    for _ in 0..epochs.max(1) {
        last_loss = classifier.train_batch(&features, &labels)?;
        if last_loss.is_nan() {
            return Err(ExtractionError::ModelError(
                "classifier training diverged (NaN loss)".to_string(),
            ));
        }
    }

    Ok((classifier, last_loss))
}

/// Result of a hybrid extraction.
#[derive(Debug, Clone)]
pub struct HybridExtraction {
    /// The extracted article text.
    pub text: String,
    /// XPath of the selected content node.
    pub xpath: String,
    /// Index of the selected candidate.
    pub candidate_index: usize,
    /// Content score of the selected candidate (classifier prob or heuristic).
    pub score: f32,
}

/// End-to-end hybrid extractor: the classifier (supervised) picks the content
/// node, then the RL-tuned [`ExtractionParams`] drive block-level extraction
/// within it. When no trained classifier is supplied it falls back to the
/// Readability-style [`NodeFeatures::heuristic_content_score`], so it is useful
/// even before any training has happened.
pub struct HybridExtractor {
    classifier: Option<NodeClassifier>,
    stopwords: HashSet<String>,
}

impl HybridExtractor {
    /// Heuristic-only extractor (no learned model).
    pub fn heuristic(stopwords: HashSet<String>) -> Self {
        Self { classifier: None, stopwords }
    }

    /// Extractor backed by a trained node classifier.
    pub fn with_classifier(classifier: NodeClassifier, stopwords: HashSet<String>) -> Self {
        Self { classifier: Some(classifier), stopwords }
    }

    /// Extract article content from a page. Returns `None` when the document
    /// exposes no candidate nodes.
    pub fn extract(
        &self,
        html: &str,
        num_candidates: usize,
        params: &ExtractionParams,
    ) -> Result<Option<HybridExtraction>> {
        let document = HtmlParser::clean_html(html)?;
        let candidates = HtmlParser::get_candidate_nodes(&document, num_candidates);
        if candidates.is_empty() {
            return Ok(None);
        }

        let features: Vec<NodeFeatures> = candidates
            .iter()
            .map(|e| node_features::extract_features(e, &self.stopwords))
            .collect();

        let scores: Vec<f32> = match &self.classifier {
            Some(c) => c.score_batch(&features)?,
            None => features.iter().map(|f| f.heuristic_content_score()).collect(),
        };

        let best = scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        let content = node_features::node_content(&candidates[best]);
        Ok(Some(HybridExtraction {
            text: content.extract(params),
            xpath: HtmlParser::get_element_path(candidates[best]),
            candidate_index: best,
            score: scores[best],
        }))
    }
}

/// Derive the supervised label vector for one page: 1.0 for the candidate whose
/// extracted text best matches the ground truth (token F1), 0.0 for the rest.
///
/// Returns `None` when there is no ground truth or no candidate scores above
/// zero (nothing to learn from for this page).
pub fn label_from_f1(
    contents: &[CandidateContent],
    ground_truth: &str,
    stopwords: &HashSet<String>,
) -> Option<Vec<f32>> {
    if contents.is_empty() || ground_truth.trim().is_empty() {
        return None;
    }

    let params = ExtractionParams::default();
    let f1s: Vec<f32> = contents
        .iter()
        .map(|c| TextUtils::token_f1(&c.extract(&params), ground_truth, stopwords))
        .collect();

    let (best_idx, best_f1) = f1s
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, v)| (i, *v))?;

    if best_f1 <= 0.0 {
        return None;
    }

    Some((0..contents.len()).map(|i| if i == best_idx { 1.0 } else { 0.0 }).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A clearly content-like feature vector.
    fn content_features(seed: f32) -> NodeFeatures {
        let mut f = NodeFeatures::zeros();
        f.word_count_norm = 0.8 + 0.1 * seed;
        f.char_count_norm = 0.8;
        f.p_count_norm = 0.7;
        f.link_density = 0.05;
        f.stopword_ratio = 0.45;
        f.tag_article = 1.0;
        f.class_positive = 1.0;
        f.unique_word_ratio = 0.6;
        f
    }

    /// A clearly boilerplate-like feature vector.
    fn boilerplate_features(seed: f32) -> NodeFeatures {
        let mut f = NodeFeatures::zeros();
        f.word_count_norm = 0.1 + 0.05 * seed;
        f.char_count_norm = 0.1;
        f.p_count_norm = 0.0;
        f.link_density = 0.9;
        f.stopword_ratio = 0.1;
        f.tag_div = 1.0;
        f.class_negative = 1.0;
        f.unique_word_ratio = 0.95;
        f
    }

    #[test]
    fn classifier_learns_to_rank_content_above_boilerplate() {
        let device = Device::Cpu;
        let mut clf = NodeClassifier::new(&device, 1e-2).unwrap();

        // Build a synthetic dataset: 4 candidates per page, the first is content.
        let mut features = Vec::new();
        let mut labels = Vec::new();
        for k in 0..40 {
            let s = (k % 5) as f32 / 5.0;
            features.push(content_features(s));
            labels.push(1.0);
            for _ in 0..3 {
                features.push(boilerplate_features(s));
                labels.push(0.0);
            }
        }

        let first_loss = clf.train_batch(&features, &labels).unwrap();
        let mut last_loss = first_loss;
        for _ in 0..200 {
            last_loss = clf.train_batch(&features, &labels).unwrap();
            assert!(!last_loss.is_nan(), "loss went NaN");
        }
        assert!(last_loss < first_loss, "loss should decrease: {first_loss} -> {last_loss}");

        // The content node must score above boilerplate, and select_best must
        // pick it out of a fresh page.
        let page = vec![
            boilerplate_features(0.2),
            content_features(0.3),
            boilerplate_features(0.4),
        ];
        let scores = clf.score_batch(&page).unwrap();
        assert!(scores[1] > scores[0] && scores[1] > scores[2], "scores: {scores:?}");
        assert_eq!(clf.select_best(&page).unwrap(), Some(1));
    }

    #[test]
    fn hybrid_extractor_heuristic_picks_article() {
        let stopwords: HashSet<String> = ["the", "a", "is", "to", "of", "and", "for", "in"]
            .into_iter()
            .map(|s| s.to_string())
            .collect();

        let html = r#"
            <html><body>
                <nav class="site-nav"><a href="/a">Home</a> <a href="/b">Sports</a> <a href="/c">World</a></nav>
                <article class="article-body">
                    <p>The mission successfully entered orbit after a seven month journey through deep space.</p>
                    <p>Engineers celebrated as telemetry confirmed every subsystem performed within tolerances.</p>
                    <p>The spacecraft will now begin its primary science campaign mapping the surface below.</p>
                </article>
                <div class="footer-links"><a href="/x">Privacy</a> <a href="/y">Terms</a> <a href="/z">Contact</a></div>
            </body></html>
        "#;

        let extractor = HybridExtractor::heuristic(stopwords);
        let result = extractor
            .extract(html, 10, &ExtractionParams::default())
            .unwrap()
            .expect("should extract something");

        // The heuristic must select the article body, not the nav or footer.
        assert!(result.text.contains("entered orbit"), "got: {}", result.text);
        assert!(!result.text.contains("Privacy"));
        assert!(result.score > 0.0);
    }

    #[test]
    fn save_load_round_trip_preserves_scores() {
        let device = Device::Cpu;
        let mut clf = NodeClassifier::new(&device, 1e-2).unwrap();

        // Train briefly so the weights are non-trivial.
        let feats = vec![content_features(0.1), boilerplate_features(0.2)];
        let labels = vec![1.0, 0.0];
        for _ in 0..50 {
            clf.train_batch(&feats, &labels).unwrap();
        }

        let page = vec![boilerplate_features(0.3), content_features(0.4)];
        let before = clf.score_batch(&page).unwrap();

        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("clf.safetensors");
        clf.save(&path).unwrap();

        let loaded = NodeClassifier::load(&path, &device, 1e-2).unwrap();
        let after = loaded.score_batch(&page).unwrap();

        for (b, a) in before.iter().zip(after.iter()) {
            assert!((b - a).abs() < 1e-5, "score changed after reload: {b} vs {a}");
        }
    }

    #[test]
    fn train_classifier_fits_labelled_pages() {
        use crate::Config;

        let page = r#"
            <html><body>
                <nav class="site-nav"><a href="/a">Home</a> <a href="/b">News</a> <a href="/c">Sport</a></nav>
                <article class="article-body">
                    <p>The committee approved the new budget after a lengthy debate on public spending.</p>
                    <p>Officials said the additional funds would be directed toward infrastructure projects.</p>
                    <p>Opposition members requested further review of the long term fiscal projections.</p>
                </article>
                <div class="footer-links"><a href="/p">Privacy</a> <a href="/t">Terms</a> <a href="/s">Subscribe</a></div>
            </body></html>
        "#;
        let gt = "The committee approved the new budget after a lengthy debate on public spending. \
                  Officials said the additional funds would be directed toward infrastructure projects. \
                  Opposition members requested further review of the long term fiscal projections.";

        let config = Config::default();
        let device = Device::Cpu;

        let samples = vec![TrainingSample::with_ground_truth(
            page.to_string(),
            "https://example.com/budget".to_string(),
            gt.to_string(),
        )];

        let (clf, loss) = train_classifier(&samples, &config, 150, 1e-2, &device).unwrap();
        assert!(loss.is_finite() && loss < 0.5, "classifier did not fit (loss {loss})");

        // After training, select_best must pick the candidate the labels marked
        // as content (the article body).
        let document = HtmlParser::clean_html(page).unwrap();
        let candidates =
            HtmlParser::get_candidate_nodes(&document, config.num_candidate_nodes);
        let contents: Vec<_> =
            candidates.iter().map(node_features::node_content).collect();
        let labels = label_from_f1(&contents, gt, &config.stopwords).unwrap();
        let expected = labels.iter().position(|&l| l == 1.0).unwrap();

        let features: Vec<_> = candidates
            .iter()
            .map(|c| node_features::extract_features(c, &config.stopwords))
            .collect();
        assert_eq!(clf.select_best(&features).unwrap(), Some(expected));
    }

    #[test]
    fn train_classifier_errors_without_ground_truth() {
        use crate::Config;
        let config = Config::default();
        let samples = vec![TrainingSample::from((
            "<html><body><article><p>no ground truth here</p></article></body></html>".to_string(),
            "https://example.com/x".to_string(),
        ))];
        assert!(train_classifier(&samples, &config, 10, 1e-2, &Device::Cpu).is_err());
    }

    #[test]
    fn label_from_f1_picks_best_matching_candidate() {
        use crate::node_features::node_content;
        use scraper::{Html, Selector};

        let html = r#"
            <html><body>
                <div class="nav"><a href="/a">Home</a> <a href="/b">News</a></div>
                <article>
                    <p>The central bank raised interest rates today citing persistent inflation pressures.</p>
                    <p>Economists expect further tightening over the coming quarters as growth slows.</p>
                </article>
            </body></html>
        "#;
        let doc = Html::parse_document(html);
        let gt = "The central bank raised interest rates today citing persistent inflation \
                  pressures. Economists expect further tightening over the coming quarters as \
                  growth slows.";

        let stopwords: HashSet<String> = ["the", "a", "as", "today", "over"]
            .into_iter()
            .map(|s| s.to_string())
            .collect();

        let nav = doc.select(&Selector::parse("div").unwrap()).next().unwrap();
        let article = doc.select(&Selector::parse("article").unwrap()).next().unwrap();
        let contents = vec![node_content(&nav), node_content(&article)];

        let labels = label_from_f1(&contents, gt, &stopwords).unwrap();
        // The article (index 1) must be the positive label.
        assert_eq!(labels, vec![0.0, 1.0]);

        // No ground truth -> no labels.
        assert!(label_from_f1(&contents, "", &stopwords).is_none());
    }
}
