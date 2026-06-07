//! High-level command interface for CLI
//! This module contains the main logic for each CLI command
// ============================================================================
// FILE: crates/content-extractor-rl/src/cli_utils.rs
// ============================================================================

use crate::*;
use std::path::{Path, PathBuf};
use bzip2::read::BzDecoder;
use std::io::Read;
use indicatif::{ProgressBar, ProgressStyle};
use url::Url;
use crate::node_classifier::{HybridExtractor, NodeClassifier};
use crate::node_features::ExtractionParams;
use crate::text_utils::TextUtils;

/// Outcome of a content-only extraction (text + the node it came from).
struct ContentExtraction {
    text: String,
    xpath: String,
    method: &'static str,
}

/// Run a trained RL agent greedily through the environment for one page and
/// return the highest-quality extraction it produced. This is the real RL
/// inference path: the agent observes the page's DOM features, picks a content
/// node and tunes the extraction params, and we keep the best step.
fn rl_agent_extract(
    html: &str,
    url: &str,
    config: &Config,
    agent: &dyn RLAgent,
) -> Result<ContentExtraction> {
    let baseline = BaselineExtractor::new(config.stopwords.clone());
    let mut env = ArticleExtractionEnvironment::new(baseline, config.clone());
    let mut state = env.reset(html, url.to_string(), None, None)?;

    let mut best = ContentExtraction { text: String::new(), xpath: String::new(), method: "rl" };
    let mut best_quality = f32::MIN;
    let mut done = false;
    let mut steps = 0;

    while !done && steps < config.max_steps_per_episode {
        let action = agent.select_action(&state, 0.0)?; // greedy (epsilon = 0)
        let (next_state, _reward, is_done, info) = env.step(action)?;
        if info.quality_score > best_quality && !info.text.trim().is_empty() {
            best_quality = info.quality_score;
            best.text = info.text.clone();
            best.xpath = info.xpath.clone();
        }
        state = next_state;
        done = is_done;
        steps += 1;
    }

    Ok(best)
}

/// Pick the content node with the supervised/heuristic [`HybridExtractor`] and
/// extract its text. Used when no trained RL model is supplied.
fn hybrid_extract(html: &str, config: &Config) -> Result<ContentExtraction> {
    let extractor = HybridExtractor::heuristic(config.stopwords.clone());
    match extractor.extract(html, config.num_candidate_nodes, &ExtractionParams::default())? {
        Some(e) => Ok(ContentExtraction { text: e.text, xpath: e.xpath, method: "hybrid" }),
        None => Ok(ContentExtraction { text: String::new(), xpath: String::new(), method: "hybrid" }),
    }
}

/// Extract a complete article (title/date metadata + body) from one page.
///
/// Body selection prefers, in order: the trained RL `agent` if supplied, then
/// the hybrid/heuristic node selector, then the plain baseline. Title and date
/// always come from the baseline metadata extractor. This is the single shared
/// entry point used by the CLI and the Python bindings.
///
/// ```no_run
/// use content_extractor_rl::{Config, extract_article};
/// let config = Config::default();
/// let html = std::fs::read_to_string("page.html").unwrap();
/// // No model -> hybrid heuristic selection (no training required):
/// let article = extract_article(&html, "https://example.com/post", &config, None).unwrap();
/// println!("{}", article.content);
/// ```
pub fn extract_article(
    html: &str,
    url: &str,
    config: &Config,
    agent: Option<&dyn RLAgent>,
) -> Result<ExtractedArticle> {
    let baseline_extractor = BaselineExtractor::new(config.stopwords.clone());
    let baseline_result = baseline_extractor.extract(html)?;
    let content = extract_content(html, url, config, agent, None, &baseline_result)?;
    let quality_score = TextUtils::calculate_text_quality(&content.text, &config.stopwords);

    Ok(ExtractedArticle {
        url: url.to_string(),
        title: baseline_result.title,
        date: baseline_result.date,
        content: content.text,
        quality_score,
        method: content.method.to_string(),
        xpath: Some(content.xpath),
    })
}

/// Extract a complete article using a prepared [`HybridExtractor`] (which may be
/// backed by a trained `NodeClassifier` or the heuristic). Title/date come from
/// the baseline metadata extractor; falls back to the baseline body if the
/// hybrid selector returns nothing.
pub fn extract_article_hybrid(
    html: &str,
    url: &str,
    config: &Config,
    hybrid: &HybridExtractor,
) -> Result<ExtractedArticle> {
    let baseline_extractor = BaselineExtractor::new(config.stopwords.clone());
    let baseline_result = baseline_extractor.extract(html)?;

    let (text, xpath, method) =
        match hybrid.extract(html, config.num_candidate_nodes, &ExtractionParams::default())? {
            Some(e) if !e.text.trim().is_empty() => (e.text, e.xpath, "classifier"),
            _ => (
                baseline_result.text.clone(),
                baseline_result.xpath.clone(),
                "baseline",
            ),
        };

    let quality_score = TextUtils::calculate_text_quality(&text, &config.stopwords);

    Ok(ExtractedArticle {
        url: url.to_string(),
        title: baseline_result.title,
        date: baseline_result.date,
        content: text,
        quality_score,
        method: method.to_string(),
        xpath: Some(xpath),
    })
}

/// Extract the article body for one page, preferring (in order): the RL agent if
/// supplied, then the hybrid/heuristic node selector, then the plain baseline.
/// Title/date always come from the baseline metadata extractor.
fn extract_content(
    html: &str,
    url: &str,
    config: &Config,
    agent: Option<&dyn RLAgent>,
    hybrid: Option<&HybridExtractor>,
    baseline_result: &crate::site_profile::ExtractionResult,
) -> Result<ContentExtraction> {
    let primary = if let Some(agent) = agent {
        rl_agent_extract(html, url, config, agent)?
    } else if let Some(hybrid) = hybrid {
        match hybrid.extract(html, config.num_candidate_nodes, &ExtractionParams::default())? {
            Some(e) => ContentExtraction { text: e.text, xpath: e.xpath, method: "classifier" },
            None => ContentExtraction { text: String::new(), xpath: String::new(), method: "classifier" },
        }
    } else {
        hybrid_extract(html, config)?
    };

    if primary.text.trim().is_empty() {
        // Fall back to the baseline body so we never return nothing.
        Ok(ContentExtraction {
            text: baseline_result.text.clone(),
            xpath: baseline_result.xpath.clone(),
            method: "baseline",
        })
    } else {
        Ok(primary)
    }
}

/// Extract article from single HTML file.
///
/// Selection priority: RL `model_path` (if any) → trained `classifier_path`
/// (if any) → hybrid heuristic.
pub fn extract_single(
    html_file: &Path,
    url: String,
    model_path: Option<&Path>,
    classifier_path: Option<&Path>,
    output: Option<&Path>,
    config: &Config,
) -> Result<ExtractedArticle> {
    let html_content = read_html_file(html_file)?;

    let article = if let Some(model_path) = model_path {
        // RL agent (any algorithm — auto-detected).
        let device = get_device();
        let agent = AgentFactory::load(
            model_path,
            config.state_dim,
            config.num_discrete_actions,
            config.num_continuous_params,
            &device,
        )?;
        extract_article(&html_content, &url, config, Some(agent.as_ref()))?
    } else if let Some(classifier_path) = classifier_path {
        // Supervised node classifier.
        let device = get_device();
        let classifier = NodeClassifier::load(classifier_path, &device, config.learning_rate)?;
        let hybrid = HybridExtractor::with_classifier(classifier, config.stopwords.clone());
        extract_article_hybrid(&html_content, &url, config, &hybrid)?
    } else {
        // Hybrid heuristic (no model).
        extract_article(&html_content, &url, config, None)?
    };

    if let Some(output_path) = output {
        let batch_result = BatchExtractionResult {
            articles: vec![article.clone()],
        };
        let json = serde_json::to_string_pretty(&batch_result)?;
        std::fs::write(output_path, json)?;
    }

    Ok(article)
}

/// Extract batch of HTML files with site profile support
pub fn extract_batch(
    archive_dir: &Path,
    model_path: Option<&Path>,
    classifier_path: Option<&Path>,
    output_dir: &Path,
    max_files: Option<usize>,
    _batch_size: usize,
    config: &Config,
) -> Result<BatchExtractionResult> {
    std::fs::create_dir_all(output_dir)?;

    let file_pairs = load_html_files_recursive(archive_dir, max_files)?;

    if file_pairs.is_empty() {
        return Err(ExtractionError::ExtractionFailed(
            "No HTML files found".to_string()
        ));
    }
    let count_of_files: usize = file_pairs.len();

    tracing::info!("Found {} HTML/JSON file pairs", count_of_files);

    let baseline_extractor = BaselineExtractor::new(config.stopwords.clone());
    let device = get_device();
    let agent = if let Some(path) = model_path {
        Some(AgentFactory::load(
            path,
            config.state_dim,
            config.num_discrete_actions,
            config.num_continuous_params,
            &device,
        )?)
    } else {
        None
    };

    // A trained classifier is used only when no RL model is supplied.
    let hybrid = if agent.is_none() {
        if let Some(path) = classifier_path {
            let classifier = NodeClassifier::load(path, &device, config.learning_rate)?;
            Some(HybridExtractor::with_classifier(classifier, config.stopwords.clone()))
        } else {
            None
        }
    } else {
        None
    };

    // Initialize site profile memory
    let mut site_memory = SiteProfileMemory::new(&config.site_profiles_dir)?;

    let pb = ProgressBar::new(count_of_files as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("=>-"),
    );

    let mut all_articles = Vec::new();
    let mut failed = Vec::new();
    let mut site_profile_used_count = 0;

    for (html_path, json_path) in file_pairs {
        let url = read_url_from_json(&json_path);
        let domain = extract_domain_from_url(&url);

        let html_content = match read_html_file(&html_path) {
            Ok(content) => content,
            Err(e) => {
                failed.push((url, e.to_string()));
                pb.inc(1);
                continue;
            }
        };

        // Get site profile for this domain
        let site_profile = site_memory.get_profile(&domain);
        let has_profile = site_profile.extractions.len() > 5;

        if has_profile {
            site_profile_used_count += 1;
        }

        match baseline_extractor.extract(&html_content) {
            Ok(result) => {
                // Select the content node with the RL agent (if loaded) or the
                // hybrid heuristic; metadata comes from the baseline result.
                let content = match extract_content(
                    &html_content, &url, config, agent.as_deref(), hybrid.as_ref(), &result,
                ) {
                    Ok(c) => c,
                    Err(e) => {
                        failed.push((url, e.to_string()));
                        pb.inc(1);
                        continue;
                    }
                };

                let method = if has_profile {
                    format!("{}+profile", content.method)
                } else {
                    content.method.to_string()
                };

                let quality_score =
                    TextUtils::calculate_text_quality(&content.text, &config.stopwords);

                let article = ExtractedArticle {
                    url: url.clone(),
                    title: result.title.clone(),
                    date: result.date.clone(),
                    content: content.text.clone(),
                    quality_score,
                    method,
                    xpath: Some(content.xpath.clone()),
                };

                // Update site profile with this extraction
                let extraction_result = site_profile::ExtractionResult {
                    text: content.text,
                    xpath: content.xpath,
                    quality_score,
                    parameters: result.parameters,
                    title: result.title,
                    date: result.date,
                };
                site_profile.add_extraction(extraction_result);

                all_articles.push(article);
            }
            Err(e) => {
                failed.push((url, e.to_string()));
            }
        }
        pb.inc(1);
    }

    pb.finish_with_message("Batch extraction complete");

    // Save site profiles
    site_memory.save_all()?;
    tracing::info!("Site profiles saved ({} domains used profiles)", site_profile_used_count);

    // Save results
    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let results_path = output_dir.join(format!("batch_results_{}.json", timestamp));
    let batch_result = BatchExtractionResult { articles: all_articles.clone() };
    let json = serde_json::to_string_pretty(&batch_result)?;
    std::fs::write(&results_path, json)?;

    // Save failed extractions
    if !failed.is_empty() {
        let failed_path = output_dir.join(format!("failed_{}.json", timestamp));
        let failed_json = serde_json::to_string_pretty(&failed)?;
        std::fs::write(&failed_path, failed_json)?;
        tracing::warn!("Failed extractions saved to: {}", failed_path.display());
    }

    tracing::info!("Batch extraction: {}/{} successful, {} with site profiles",
                   all_articles.len(), count_of_files, site_profile_used_count);

    Ok(batch_result)
}

/// Extract domain from URL
pub fn extract_domain_from_url(url: &str) -> String {
    match Url::parse(url) {
        Ok(parsed_url) => {
            parsed_url.host_str()
                .map(|h| h.to_string())
                .unwrap_or_else(|| "unknown".to_string())
        }
        Err(_) => {
            let url = url.trim();
            let without_protocol = url.strip_prefix("https://")
                .or_else(|| url.strip_prefix("http://"))
                .unwrap_or(url);

            let host_part = without_protocol.split('/').next().unwrap_or("");
            let domain = host_part.split(':').next().unwrap_or("");

            if domain.is_empty() {
                "unknown".to_string()
            } else {
                domain.to_string()
            }
        }
    }
}

/// Load HTML files recursively
pub fn load_html_files_recursive(
    dir: &Path,
    max_files: Option<usize>,
) -> Result<Vec<(PathBuf, PathBuf)>> {
    use walkdir::WalkDir;

    let mut files = Vec::new();

    for entry in WalkDir::new(dir).into_iter().filter_map(|e| e.ok()) {
        if let Some(max) = max_files {
            if files.len() >= max {
                break;
            }
        }

        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path.extension() {
                if ext == "bz2" && path.to_string_lossy().contains(".html.") {
                    let json_path = path.with_extension("").with_extension("json");
                    if json_path.exists() {
                        files.push((path.to_path_buf(), json_path));
                    }
                } else if ext == "html" || ext == "htm" {
                    let json_path = path.with_extension("json");
                    if json_path.exists() {
                        files.push((path.to_path_buf(), json_path));
                    }
                }
            }
        }
    }

    Ok(files)
}

/// Read HTML file with UTF-8 error handling
pub fn read_html_file(path: &Path) -> Result<String> {
    if path.extension().and_then(|s| s.to_str()) == Some("bz2") {
        let file = std::fs::File::open(path)?;
        let mut decoder = BzDecoder::new(file);
        let mut bytes = Vec::new();
        decoder.read_to_end(&mut bytes)?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    } else {
        let bytes = std::fs::read(path)?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
}

/// Read URL from JSON file
pub fn read_url_from_json(json_path: &Path) -> String {
    match std::fs::read_to_string(json_path) {
        Ok(json_content) => {
            if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(&json_content) {
                json_value.get("URL")
                    .and_then(|u| u.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "https://example.com/unknown".to_string())
            } else {
                "https://example.com/invalid-json".to_string()
            }
        }
        Err(_) => "https://example.com/no-json".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_extract_domain_from_url() {
        assert_eq!(
            extract_domain_from_url("https://www.example.com/article"),
            "www.example.com"
        );

        assert_eq!(
            extract_domain_from_url("http://subdomain.example.org:8080/path"),
            "subdomain.example.org"
        );
    }

    #[test]
    fn test_read_url_from_json() {
        let temp_dir = TempDir::new().unwrap();
        let json_path = temp_dir.path().join("test.json");

        let json_content = r#"{"URL": "https://example.com/article"}"#;
        std::fs::write(&json_path, json_content).unwrap();

        let url = read_url_from_json(&json_path);
        assert_eq!(url, "https://example.com/article");
    }

    #[test]
    fn test_extract_single_uses_hybrid_without_model() {
        let temp_dir = TempDir::new().unwrap();
        let html_path = temp_dir.path().join("page.html");
        std::fs::write(
            &html_path,
            r#"
            <html><head><title>Mission Update</title></head><body>
                <nav class="site-nav"><a href="/a">Home</a> <a href="/b">News</a> <a href="/c">More</a></nav>
                <article class="article-body">
                    <p>The spacecraft entered orbit today after a long interplanetary cruise phase.</p>
                    <p>Mission controllers confirmed every instrument survived the journey intact.</p>
                    <p>Scientific observations of the planet will begin within the next two weeks.</p>
                </article>
                <div class="footer-links"><a href="/p">Privacy</a> <a href="/t">Terms</a></div>
            </body></html>
            "#,
        )
        .unwrap();

        let config = Config::default();
        let article = extract_single(
            &html_path,
            "https://example.com/mission".to_string(),
            None, // no RL model
            None, // no classifier -> hybrid heuristic path
            None, // no output file
            &config,
        )
        .unwrap();

        // The hybrid selector must return the article body, not nav/footer noise.
        assert_eq!(article.method, "hybrid");
        assert!(article.content.contains("entered orbit"), "content: {}", article.content);
        assert!(!article.content.contains("Privacy"), "footer leaked: {}", article.content);
    }
}