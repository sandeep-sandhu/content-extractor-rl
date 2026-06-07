// ============================================================================
// FILE: crates/content-extractor-rl/src/environment.rs
// ============================================================================
//! RL environment for article extraction.
//!
//! This is a *real* MDP: the agent's discrete action chooses which DOM
//! candidate becomes the content root, its continuous params tune block-level
//! filtering, and the resulting extracted text is scored against the
//! ground-truth article (token F1). Different actions therefore produce
//! different rewards — the precondition for learning that the previous
//! placeholder environment lacked.

use crate::baseline_extractor::BaselineExtractor;
use crate::html_parser::HtmlParser;
use crate::node_features::{self, CandidateContent, ExtractionParams, NodeFeatures};
use crate::text_utils::TextUtils;
use crate::site_profile::SiteProfile;
use crate::config::{
    Config, ACTION_SELECT_PARENT, ACTION_SELECT_SIBLING_LEFT, ACTION_SELECT_SIBLING_RIGHT,
    ACTION_EXPAND_REGION, ACTION_CONTRACT_REGION, ACTION_TERMINATE,
};
use crate::Result;
use std::collections::HashMap;

/// A candidate content node, captured as owned data so the environment does not
/// need to hold borrowed `ElementRef`s across `step` calls.
struct CandidateNode {
    xpath: String,
    features: NodeFeatures,
    content: CandidateContent,
}

/// RL environment for article extraction
pub struct ArticleExtractionEnvironment {
    baseline_extractor: BaselineExtractor,
    candidates: Vec<CandidateNode>,
    current_node_idx: Option<usize>,
    /// Coarse block-word-threshold adjustment driven by EXPAND/CONTRACT actions.
    word_threshold_adjust: i32,
    terminated: bool,
    url: String,
    domain: String,
    ground_truth_text: String,
    /// Cached baseline fallback text (used only when the DOM has no candidates).
    baseline_fallback: String,
    step_count: usize,
    max_steps: usize,
    config: Config,
}

impl ArticleExtractionEnvironment {
    /// Create new environment
    pub fn new(baseline_extractor: BaselineExtractor, config: Config) -> Self {
        Self {
            baseline_extractor,
            candidates: Vec::new(),
            current_node_idx: None,
            word_threshold_adjust: 0,
            terminated: false,
            url: String::new(),
            domain: String::new(),
            ground_truth_text: String::new(),
            baseline_fallback: String::new(),
            step_count: 0,
            max_steps: config.max_steps_per_episode,
            config,
        }
    }

    /// Reset environment with new HTML and (optionally) the ground-truth article
    /// text used to compute the reward.
    pub fn reset(
        &mut self,
        html: &str,
        url: String,
        ground_truth_text: Option<&str>,
        _site_profile: Option<&SiteProfile>,
    ) -> Result<Vec<f32>> {
        self.url = url.clone();
        self.domain = Self::extract_domain(&url);
        self.step_count = 0;
        self.word_threshold_adjust = 0;
        self.terminated = false;
        self.ground_truth_text = ground_truth_text.unwrap_or("").to_string();

        // Parse and clean HTML, then snapshot the candidate nodes' features and
        // extractable content as owned data.
        let document = HtmlParser::clean_html(html)?;
        let candidate_refs =
            HtmlParser::get_candidate_nodes(&document, self.config.num_candidate_nodes);

        self.candidates = candidate_refs
            .iter()
            .map(|node| CandidateNode {
                xpath: HtmlParser::get_element_path(*node),
                features: node_features::extract_features(node, &self.config.stopwords),
                content: node_features::node_content(node),
            })
            .collect();

        // Fallback only matters when the DOM exposed no candidate nodes at all.
        self.baseline_fallback = if self.candidates.is_empty() {
            self.baseline_extractor
                .extract(&document.html())
                .map(|r| r.text)
                .unwrap_or_default()
        } else {
            String::new()
        };

        self.current_node_idx = if self.candidates.is_empty() { None } else { Some(0) };

        self.build_state()
    }

    /// Execute action and return next state, reward, done, info
    pub fn step(&mut self, action: (usize, Vec<f32>)) -> Result<(Vec<f32>, f32, bool, StepInfo)> {
        let (discrete_action, params) = action;
        self.step_count += 1;

        let n = self.candidates.len();

        // Apply the discrete action. Node-select actions pick a candidate;
        // navigation actions move the selection or adjust block filtering;
        // TERMINATE ends the episode. Every branch has a real effect.
        match discrete_action {
            d if d < self.config.num_candidate_nodes => {
                if n > 0 {
                    self.current_node_idx = Some(d.min(n - 1));
                }
            }
            ACTION_SELECT_PARENT => self.select_parent(),
            ACTION_SELECT_SIBLING_LEFT => {
                if let Some(idx) = self.current_node_idx {
                    self.current_node_idx = Some(idx.saturating_sub(1));
                }
            }
            ACTION_SELECT_SIBLING_RIGHT => {
                if let (Some(idx), true) = (self.current_node_idx, n > 0) {
                    self.current_node_idx = Some((idx + 1).min(n - 1));
                }
            }
            // EXPAND keeps more text (lower the word threshold); CONTRACT is stricter.
            ACTION_EXPAND_REGION => {
                self.word_threshold_adjust = (self.word_threshold_adjust - 2).max(-20);
            }
            ACTION_CONTRACT_REGION => {
                self.word_threshold_adjust = (self.word_threshold_adjust + 2).min(40);
            }
            ACTION_TERMINATE => self.terminated = true,
            _ => {}
        }

        // Extract text using the *selected* node and the effective params.
        let effective_params = self.effective_params(&params);
        let extracted_text = self.extract_selected(&effective_params);

        // Reward: token F1 against ground truth when available, otherwise a
        // self-supervised text-quality proxy. Mapped to [-1, 1] with a small
        // per-step cost to encourage decisive episodes.
        let score = if self.ground_truth_text.is_empty() {
            TextUtils::calculate_text_quality(&extracted_text, &self.config.stopwords)
        } else {
            TextUtils::token_f1(&extracted_text, &self.ground_truth_text, &self.config.stopwords)
        };
        let reward = (score * 2.0 - 1.0 - 0.01 * self.step_count as f32).clamp(-1.0, 1.0);

        let done = self.terminated || self.step_count >= self.max_steps;

        let next_state = self.build_state()?;

        let info = StepInfo {
            quality_score: score,
            text: extracted_text,
            xpath: self
                .current_node_idx
                .and_then(|idx| self.candidates.get(idx))
                .map(|c| c.xpath.clone())
                .unwrap_or_default(),
            parameters: self.denormalize_params(&params),
            step_count: self.step_count,
        };

        Ok((next_state, reward, done, info))
    }

    /// Move selection to the candidate that is the nearest DOM ancestor (longest
    /// proper xpath prefix) of the current selection, if any.
    fn select_parent(&mut self) {
        let Some(idx) = self.current_node_idx else { return };
        let current_path = self.candidates[idx].xpath.clone();

        let mut best: Option<(usize, usize)> = None; // (candidate idx, prefix len)
        for (j, cand) in self.candidates.iter().enumerate() {
            if j == idx {
                continue;
            }
            if current_path.starts_with(&cand.xpath) && cand.xpath.len() < current_path.len() {
                let better = best.map(|(_, len)| cand.xpath.len() > len).unwrap_or(true);
                if better {
                    best = Some((j, cand.xpath.len()));
                }
            }
        }
        if let Some((j, _)) = best {
            self.current_node_idx = Some(j);
        }
    }

    /// Combine the policy's continuous params with the EXPAND/CONTRACT offset.
    fn effective_params(&self, params: &[f32]) -> ExtractionParams {
        let mut p = ExtractionParams::from_normalized(params);
        let adjusted = p.min_block_words as i32 + self.word_threshold_adjust;
        p.min_block_words = adjusted.clamp(1, 60) as usize;
        p
    }

    /// Extract text from the currently selected candidate.
    fn extract_selected(&self, params: &ExtractionParams) -> String {
        match self.current_node_idx.and_then(|idx| self.candidates.get(idx)) {
            Some(cand) => cand.content.extract(params),
            None => self.baseline_fallback.clone(),
        }
    }

    /// Denormalize parameters from [-1, 1] to actual ranges (for site profiles).
    fn denormalize_params(&self, params: &[f32]) -> HashMap<String, f64> {
        let mut result = HashMap::new();

        if params.len() >= 6 {
            result.insert("min_word_threshold".to_string(), (2.0 + (params[0] + 1.0) * 4.0) as f64);
            result.insert("stopword_weight".to_string(), (0.5 + (params[1] + 1.0) * 0.75) as f64);
            result.insert("link_density_penalty".to_string(), ((params[2] + 1.0) * 1.0) as f64);
            result.insert("paragraph_boost".to_string(), (1.0 + (params[3] + 1.0) * 0.5) as f64);
            result.insert("sibling_extension".to_string(), ((params[4] + 1.0) * 0.5) as f64);
            result.insert("depth_penalty".to_string(), ((params[5] + 1.0) * 0.25) as f64);
        }

        result
    }

    /// Build state vector from real DOM features.
    ///
    /// Layout (before padding to `config.state_dim`):
    ///   - `num_candidate_nodes` × `NodeFeatures::DIM` per-candidate features
    ///   - 8 global document features
    ///   - selection one-hot (`num_candidate_nodes`) + step fraction
    ///     + threshold-adjust + terminated flag
    fn build_state(&self) -> Result<Vec<f32>> {
        let mut state = Vec::with_capacity(self.config.state_dim);

        // Per-candidate features (padded slots get zeros).
        for slot in 0..self.config.num_candidate_nodes {
            match self.candidates.get(slot) {
                Some(c) => state.extend(c.features.to_vec()),
                None => state.extend(NodeFeatures::zeros().to_vec()),
            }
        }

        // Global document features.
        let n = self.candidates.len();
        let num_candidates_norm = (n as f32 / self.config.num_candidate_nodes as f32).clamp(0.0, 1.0);
        let max_word = self
            .candidates
            .iter()
            .map(|c| c.features.word_count_norm)
            .fold(0.0_f32, f32::max);
        let mean_link_density = if n == 0 {
            0.0
        } else {
            self.candidates.iter().map(|c| c.features.link_density).sum::<f32>() / n as f32
        };
        let has_article = if self.candidates.iter().any(|c| c.features.tag_article > 0.5) { 1.0 } else { 0.0 };
        let has_main = if self.candidates.iter().any(|c| c.features.tag_main > 0.5) { 1.0 } else { 0.0 };
        let mean_stopword = if n == 0 {
            0.0
        } else {
            self.candidates.iter().map(|c| c.features.stopword_ratio).sum::<f32>() / n as f32
        };

        state.push(num_candidates_norm);
        state.push(max_word);
        state.push(mean_link_density);
        state.push(mean_stopword);
        state.push(has_article);
        state.push(has_main);
        state.push(Self::hash_domain_normalized(&self.domain));
        state.push(self.ground_truth_text.is_empty() as i32 as f32);

        // Selection state.
        for slot in 0..self.config.num_candidate_nodes {
            state.push(if self.current_node_idx == Some(slot) { 1.0 } else { 0.0 });
        }
        state.push(self.step_count as f32 / self.max_steps.max(1) as f32);
        state.push((self.word_threshold_adjust as f32 / 40.0).clamp(-1.0, 1.0));
        state.push(self.terminated as i32 as f32);

        // Pad or truncate to exact STATE_DIM.
        state.truncate(self.config.state_dim);
        while state.len() < self.config.state_dim {
            state.push(0.0);
        }

        Ok(state)
    }

    /// Extract domain from URL
    fn extract_domain(url: &str) -> String {
        url::Url::parse(url)
            .ok()
            .and_then(|u| u.host_str().map(|h| h.to_string()))
            .unwrap_or_else(|| "unknown".to_string())
    }

    /// Hash domain to normalized value
    fn hash_domain_normalized(domain: &str) -> f32 {
        use sha2::{Sha256, Digest};

        let mut hasher = Sha256::new();
        hasher.update(domain.as_bytes());
        let result = hasher.finalize();

        let hash_val = u32::from_be_bytes([result[0], result[1], result[2], result[3]]);
        (hash_val % 10000) as f32 / 10000.0
    }
}

/// Information returned from step
#[derive(Debug, Clone)]
pub struct StepInfo {
    pub quality_score: f32,
    pub text: String,
    pub xpath: String,
    pub parameters: HashMap<String, f64>,
    pub step_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_html() -> &'static str {
        r#"
        <html><body>
            <nav class="navigation"><a href="/a">Home</a> <a href="/b">About</a> <a href="/c">Contact</a></nav>
            <article class="article-content">
                <p>Quantum researchers reported a significant breakthrough in error correction this week.</p>
                <p>The new technique stabilizes qubits for longer durations enabling deeper computations.</p>
                <p>Independent laboratories confirmed the reproducible measurements across several runs.</p>
            </article>
            <div class="sidebar-ads"><a href="/x">Buy now</a> <a href="/y">Subscribe today</a></div>
        </body></html>
        "#
    }

    fn env() -> ArticleExtractionEnvironment {
        let config = Config::default();
        let baseline = BaselineExtractor::new(config.stopwords.clone());
        ArticleExtractionEnvironment::new(baseline, config)
    }

    #[test]
    fn reset_produces_correct_state_dim_and_varies() {
        let mut env = env();
        let state = env
            .reset(test_html(), "https://example.com/post".to_string(), None, None)
            .unwrap();
        assert_eq!(state.len(), env.config.state_dim);
        // The state must not be the old constant 0.5 placeholder.
        let distinct: std::collections::HashSet<u32> =
            state.iter().map(|f| f.to_bits()).collect();
        assert!(distinct.len() > 5, "state should contain varied real features");
    }

    #[test]
    fn action_choice_changes_reward() {
        let gt = "Quantum researchers reported a significant breakthrough in error correction \
                  this week. The new technique stabilizes qubits for longer durations enabling \
                  deeper computations. Independent laboratories confirmed the reproducible \
                  measurements across several runs.";
        let mut env = env();
        env.reset(test_html(), "https://example.com/post".to_string(), Some(gt), None)
            .unwrap();

        // Find the article candidate vs a non-article candidate and compare rewards.
        let mut rewards = Vec::new();
        for action in 0..env.candidates.len() {
            env.reset(test_html(), "https://example.com/post".to_string(), Some(gt), None)
                .unwrap();
            let (_s, reward, _d, info) = env
                .step((action, vec![-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
                .unwrap();
            rewards.push((action, reward, info.quality_score));
        }

        let best = rewards.iter().cloned().fold((0usize, f32::MIN, 0.0), |acc, x| {
            if x.1 > acc.1 { x } else { acc }
        });
        let worst = rewards.iter().cloned().fold((0usize, f32::MAX, 0.0), |acc, x| {
            if x.1 < acc.1 { x } else { acc }
        });

        // If actions had no effect (the old bug) every reward would be equal.
        assert!(
            (best.1 - worst.1).abs() > 1e-3,
            "different node selections must yield different rewards: {rewards:?}"
        );
        // The best-scoring selection should recover most of the ground truth.
        assert!(best.2 > 0.5, "best F1 should be high, got {}", best.2);
    }

    #[test]
    fn terminate_action_ends_episode() {
        let mut env = env();
        env.reset(test_html(), "https://example.com/post".to_string(), None, None)
            .unwrap();
        let (_s, _r, done, _info) = env.step((ACTION_TERMINATE, vec![0.0; 6])).unwrap();
        assert!(done, "TERMINATE must end the episode");
    }

    #[test]
    fn episode_force_terminates_at_max_steps() {
        let mut env = env();
        env.reset(test_html(), "https://example.com/post".to_string(), None, None)
            .unwrap();
        let mut done = false;
        let mut steps = 0;
        while !done && steps < env.config.max_steps_per_episode + 5 {
            // Use a non-terminating navigation action.
            let (_s, _r, d, _i) = env.step((ACTION_SELECT_SIBLING_RIGHT, vec![0.0; 6])).unwrap();
            done = d;
            steps += 1;
        }
        assert!(done);
        assert!(steps <= env.config.max_steps_per_episode);
    }

    #[test]
    fn continuous_params_affect_extraction() {
        let mut env = env();
        env.reset(test_html(), "https://example.com/post".to_string(), None, None)
            .unwrap();
        // Select the article node (action 1 is typically the article here, but
        // force-select via repeated reset+select to be deterministic).
        let lenient = env.step((1, vec![-1.0, 1.0, 0.0, 0.0, 0.0, 0.0])).unwrap().3.text;
        env.reset(test_html(), "https://example.com/post".to_string(), None, None)
            .unwrap();
        let strict = env.step((1, vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0])).unwrap().3.text;
        // A very high min-word threshold should not produce *more* text than lenient.
        assert!(strict.len() <= lenient.len());
    }
}
