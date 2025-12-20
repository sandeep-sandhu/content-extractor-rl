use scraper::{Html, ElementRef};
use crate::baseline_extractor::BaselineExtractor;
use crate::html_parser::HtmlParser;
use crate::text_utils::TextUtils;
use crate::site_profile::SiteProfile;
use crate::config::{Config, ACTION_SELECT_PARENT, ACTION_SELECT_SIBLING_LEFT, ACTION_SELECT_SIBLING_RIGHT, ACTION_TERMINATE};
use crate::Result;
use std::collections::HashMap;

/// RL environment for article extraction
pub struct ArticleExtractionEnvironment {
    baseline_extractor: BaselineExtractor,
    document: Option<Html>,
    current_node_idx: Option<usize>,
    candidates: Vec<String>, // Store node identifiers
    url: String,
    domain: String,
    step_count: usize,
    max_steps: usize,
    config: Config,
}

impl ArticleExtractionEnvironment {
    /// Create new environment
    pub fn new(baseline_extractor: BaselineExtractor, config: Config) -> Self {
        Self {
            baseline_extractor,
            document: None,
            current_node_idx: None,
            candidates: Vec::new(),
            url: String::new(),
            domain: String::new(),
            step_count: 0,
            max_steps: config.max_steps_per_episode,
            config,
        }
    }

    /// Reset environment with new HTML
    pub fn reset(&mut self, html: &str, url: String, _site_profile: Option<&SiteProfile>) -> Result<Vec<f32>> {
        self.url = url.clone();
        self.domain = Self::extract_domain(&url);
        self.step_count = 0;

        // Parse and clean HTML
        let document = HtmlParser::clean_html(html)?;
        let candidates = HtmlParser::get_candidate_nodes(&document, self.config.num_candidate_nodes);

        // Store candidate identifiers
        self.candidates = candidates.iter()
            .map(|node| HtmlParser::get_element_path(*node))
            .collect();

        self.document = Some(document);
        self.current_node_idx = if !self.candidates.is_empty() { Some(0) } else { None };

        // Build initial state
        self.build_state()
    }

    /// Execute action and return next state, reward, done, info
    pub fn step(&mut self, action: (usize, Vec<f32>)) -> Result<(Vec<f32>, f32, bool, StepInfo)> {
        let (discrete_action, params) = action;
        self.step_count += 1;

        let mut done = false;
        let mut info = StepInfo {
            quality_score: 0.0,
            text: String::new(),
            xpath: String::new(),
            parameters: HashMap::new(),
            step_count: self.step_count,
        };

        // Execute discrete action
        match discrete_action {
            0..=9 => {
                // Select candidate node
                let idx = discrete_action.min(self.candidates.len().saturating_sub(1));
                self.current_node_idx = Some(idx);
            }
            ACTION_SELECT_PARENT => {
                // Move to parent (simplified)
            }
            ACTION_SELECT_SIBLING_LEFT => {
                // Move to left sibling (simplified)
            }
            ACTION_SELECT_SIBLING_RIGHT => {
                // Move to right sibling (simplified)
            }
            ACTION_TERMINATE => {
                done = true;
            }
            _ => {}
        }

        // Extract text with parameters
        let extracted_text = self.extract_with_params(&params)?;

        // Calculate reward
        let quality_score = TextUtils::calculate_text_quality(&extracted_text, &self.config.stopwords);
        let reward = quality_score * 2.0 - 1.0 - 0.01 * self.step_count as f32;

        // Force termination
        if self.step_count >= self.max_steps {
            done = true;
        }

        // Build next state
        let next_state = self.build_state()?;

        info.quality_score = quality_score;
        info.text = extracted_text;
        info.xpath = self.current_node_idx
            .and_then(|idx| self.candidates.get(idx))
            .cloned()
            .unwrap_or_default();
        info.parameters = self.denormalize_params(&params);

        Ok((next_state, reward, done, info))
    }

    /// Extract text using parameters
    fn extract_with_params(&self, params: &[f32]) -> Result<String> {
        // Simplified extraction using parameters
        if let Some(document) = &self.document {
            if let Some(idx) = self.current_node_idx {
                if let Some(_xpath) = self.candidates.get(idx) {
                    // In a real implementation, we would use the parameters
                    // to customize the extraction
                    let result = self.baseline_extractor.extract(&document.html())?;
                    return Ok(result.text);
                }
            }
        }

        Ok(String::new())
    }

    /// Denormalize parameters from [-1, 1] to actual ranges
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

    /// Build state vector
    fn build_state(&self) -> Result<Vec<f32>> {
        let mut state = Vec::with_capacity(self.config.state_dim);

        // Global document features (12 dims)
        if let Some(document) = &self.document {
            let all_text = document.root_element().text().collect::<String>();

            state.push(0.5); // Normalized features
            state.push(0.5);
            state.push(0.5);
            state.push(0.5);
            state.push(0.5);
            state.push(0.5);
            state.push(0.5);
            state.push(0.5);
            state.push(0.0);
            state.push(0.0);
            state.push(0.5);
            state.push(Self::hash_domain_normalized(&self.domain));
        } else {
            state.extend(vec![0.0; 12]);
        }

        // Candidate node features (20 dims * 10 nodes = 200 dims)
        for _ in 0..self.config.num_candidate_nodes {
            state.extend(vec![0.5; 20]); // Simplified features
        }

        // Historical features (8 dims)
        state.extend(vec![0.0; 8]);

        // Current extraction state (6 dims)
        state.push(self.step_count as f32 / self.max_steps as f32);
        state.extend(vec![0.5; 5]);

        // Pad or truncate to exact STATE_DIM
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
