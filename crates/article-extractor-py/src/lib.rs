// ============================================================================
// FILE: crates/article-extractor-py/src/lib.rs
// ============================================================================


use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::{PyDict, PyModule};
use article_extractor::{
    Config, BaselineExtractor, SiteProfileMemory,
    ExtractedArticle, BatchExtractionResult, cuda_is_available,
};
use std::path::PathBuf;
use article_extractor::agents::dqn_agent::DQNAgent;

/// Python wrapper for the article extractor
#[pyclass]
struct RustArticleExtractor {
    baseline_extractor: BaselineExtractor,
    agent: Option<DQNAgent>,
    site_memory: SiteProfileMemory,
    config: Config,
}

#[pymethods]
impl RustArticleExtractor {
    /// Create new extractor

    /// Create new extractor
    ///
    /// Args:
    ///     site_profile: Path to site profile JSON file (optional)
    ///     model: Path to ONNX model file (optional)
    ///
    /// Returns:
    ///     RustArticleExtractor instance
    #[new]
    #[pyo3(signature = (_site_profile=None, model=None))]
    fn new(_site_profile: Option<String>, model: Option<String>) -> PyResult<Self> {
        // Print device info immediately
        let device = article_extractor::device::get_device();
        let device_info = article_extractor::device::get_device_info(&device);

        // Use println! to ensure it reaches Python's stdout
        println!("╔═══════════════════════════════════════╗");
        println!("║  Article Extractor - Initialization    ║");
        println!("╠═══════════════════════════════════════╣");
        println!("║ Device: {:<31} ║", device_info);
        println!("╚═══════════════════════════════════════╝");

        let config = Config::from_env()
            .map_err(|e| PyRuntimeError::new_err(format!("Config error: {}", e)))?;

        config.setup_directories()
            .map_err(|e| PyRuntimeError::new_err(format!("Setup error: {}", e)))?;

        let baseline_extractor = BaselineExtractor::new(config.stopwords.clone());

        let site_memory = SiteProfileMemory::new(&config.site_profiles_dir)
            .map_err(|e| PyRuntimeError::new_err(format!("Site memory error: {}", e)))?;

        let agent = if let Some(model_path) = model {
            let path = PathBuf::from(model_path);
            Some(
                DQNAgent::load(
                    &path,
                    config.state_dim,
                    config.num_discrete_actions,
                    config.num_continuous_params,
                ).map_err(|e| PyRuntimeError::new_err(format!("Model load error: {}", e)))?
            )
        } else {
            None
        };

        Ok(Self {
            baseline_extractor,
            agent,
            site_memory,
            config,
        })
    }

    /// Check if CUDA is available
    ///
    /// Returns:
    ///     bool: True if CUDA is available, False otherwise
    #[pyo3(signature = ())]
    fn check_cuda_available(&self) -> PyResult<bool> {
        Ok(cuda_is_available())
    }

    /// Extract article from HTML
    ///
    /// Args:
    ///     website_page_html: HTML content as string
    ///     url: URL of the page
    ///
    /// Returns:
    ///     Dictionary containing extracted article data
    #[pyo3(signature = (website_page_html, url))]
    fn extract(&mut self, website_page_html: String, url: String) -> PyResult<Py<PyAny>> {
        // Extract using baseline or RL model
        let result = if self.agent.is_some() {
            // TODO: Use RL agent for extraction
            self.baseline_extractor.extract(&website_page_html)
                .map_err(|e| PyRuntimeError::new_err(format!("Extraction error: {}", e)))?
        } else {
            self.baseline_extractor.extract(&website_page_html)
                .map_err(|e| PyRuntimeError::new_err(format!("Extraction error: {}", e)))?
        };

        // Extract domain and update site profile
        let domain = url::Url::parse(&url)
            .ok()
            .and_then(|u: url::Url| u.host_str().map(|h: &str| h.to_string()))
            .unwrap_or_else(|| "unknown".to_string());

        let profile = self.site_memory.get_profile(&domain);
        profile.add_extraction(result.clone());

        // Save profile
        self.site_memory.save_profile(&domain)
            .map_err(|e| PyRuntimeError::new_err(format!("Profile save error: {}", e)))?;

        // Create article result
        let article = ExtractedArticle {
            url: url.clone(),
            title: None,
            date: None,
            content: result.text,
            quality_score: result.quality_score,
            method: if self.agent.is_some() { "rl".to_string() } else { "baseline".to_string() },
            xpath: Some(result.xpath),
        };

        // Convert to Python dict - FIXED: Use Python::with_gil correctly
        Python::attach(|py| {
            let dict = PyDict::new(py);
            dict.set_item("url", article.url)?;
            dict.set_item("title", article.title)?;
            dict.set_item("date", article.date)?;
            dict.set_item("content", article.content)?;
            dict.set_item("quality_score", article.quality_score)?;
            dict.set_item("method", article.method)?;
            dict.set_item("xpath", article.xpath)?;
            Ok(dict.into())
        })
    }

    /// Extract multiple articles
    ///
    /// Args:
    ///     html_url_pairs: List of tuples (html, url)
    ///
    /// Returns:
    ///     Dictionary with "articles" key containing list of extracted articles
    #[pyo3(signature = (html_url_pairs))]
    fn extract_batch(&mut self, html_url_pairs: Vec<(String, String)>) -> PyResult<Py<PyAny>> {
        let mut articles = Vec::new();

        for (html, url) in html_url_pairs {
            let result = self.baseline_extractor.extract(&html)
                .map_err(|e| PyRuntimeError::new_err(format!("Extraction error: {}", e)))?;

            let article = ExtractedArticle {
                url: url.clone(),
                title: None,
                date: None,
                content: result.text,
                quality_score: result.quality_score,
                method: "baseline".to_string(),
                xpath: Some(result.xpath),
            };

            articles.push(article);
        }

        let batch_result = BatchExtractionResult { articles };

        // Convert to Python dict - FIXED: Use Python::with_gil
        Python::attach(|py| {
            let json_str = serde_json::to_string(&batch_result)
                .map_err(|e| PyRuntimeError::new_err(format!("JSON error: {}", e)))?;

            let json_module = py.import("json")?;
            let loads = json_module.getattr("loads")?;
            let result = loads.call1((json_str,))?;

            Ok(result.into())
        })
    }

    /// Train the model
    ///
    /// Args:
    ///     html_samples: List of tuples (html, url) for training
    ///     episodes: Number of training episodes
    ///     improved: Use improved training (curriculum learning, etc.)
    ///
    /// Returns:
    ///     Dictionary with training metrics
    #[pyo3(signature = (html_samples, episodes=1000, improved=false))]
    fn train(
        &mut self,
        html_samples: Vec<(String, String)>,
        episodes: usize,
        improved: bool,
    ) -> PyResult<Py<PyAny>> {
        // Log device info before training
        let device = article_extractor::device::get_device();
        let device_info = article_extractor::device::get_device_info(&device);
        println!("╔═══════════════════════════════════════╗");
        println!("║  Starting Training                     ║");
        println!("╠═══════════════════════════════════════╣");
        println!("║ Device: {:<31} ║", device_info);
        println!("║ Episodes: {:<29} ║", episodes);
        println!("║ Mode: {:<33} ║", if improved { "Improved" } else { "Standard" });
        println!("║ Samples: {:<30} ║", html_samples.len());
        println!("╚═══════════════════════════════════════╝");

        let mut config = self.config.clone();
        config.num_episodes = episodes;

        let (_agent, metrics) = if improved {
            article_extractor::train_with_improvements(&config, html_samples)
                .map_err(|e| PyRuntimeError::new_err(format!("Training error: {}", e)))?
        } else {
            article_extractor::train_standard(&config, html_samples)
                .map_err(|e| PyRuntimeError::new_err(format!("Training error: {}", e)))?
        };

        // Convert metrics to Python dict
        Python::attach(|py| {
            let dict = PyDict::new(py);
            dict.set_item("episode_rewards", metrics.episode_rewards)?;
            dict.set_item("episode_qualities", metrics.episode_qualities)?;
            dict.set_item("best_avg_quality", metrics.best_avg_quality)?;
            Ok(dict.into())
        })
    }

    /// Get statistics about extractions
    #[pyo3(signature = ())]
    fn get_stats(&self) -> PyResult<Py<PyAny>> {
        Python::attach(|py| {
            let dict = PyDict::new(py);
            dict.set_item("has_model", self.agent.is_some())?;
            dict.set_item("num_profiles", 0)?; // Simplified
            Ok(dict.into())
        })
    }
}

/// Check if CUDA is available (module-level function)
#[pyfunction]
fn check_cuda_available() -> PyResult<bool> {
    Ok(cuda_is_available())
}

/// Python module definition
#[pymodule]
fn article_extractor_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustArticleExtractor>()?;
    m.add_function(wrap_pyfunction!(check_cuda_available, m)?)?;
    Ok(())
}