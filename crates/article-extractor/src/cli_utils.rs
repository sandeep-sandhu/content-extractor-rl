//! High-level command interface for CLI
//! This module contains the main logic for each CLI command

use crate::*;
use std::path::{Path, PathBuf};
use bzip2::read::BzDecoder;
use std::io::Read;
use indicatif::{ProgressBar, ProgressStyle};
use url::Url;

/// Extract article from single HTML file
pub fn extract_single(
    html_file: &Path,
    url: String,
    model_path: Option<&Path>,
    output: Option<&Path>,
    config: &Config,
) -> Result<ExtractedArticle> {
    let html_content = read_html_file(html_file)?;
    let baseline_extractor = BaselineExtractor::new(config.stopwords.clone());

    // Extract domain for site profile
    let domain = extract_domain_from_url(&url);

    // Try to load site profile
    let mut site_memory = SiteProfileMemory::new(&config.site_profiles_dir)?;
    let site_profile = site_memory.get_profile(&domain);

    let result = if let Some(model_path) = model_path {
        let device = get_device();
        let _agent = DQNAgent::load_with_device(
            model_path,
            config.state_dim,
            config.num_discrete_actions,
            config.num_continuous_params,
            &device,
        )?;

        // Use site profile if available for better extraction
        if site_profile.extractions.len() > 5 {
            tracing::debug!("Using site profile for {} (has {} past extractions)",
                          domain, site_profile.extractions.len());
        }

        baseline_extractor.extract(&html_content)?
    } else {
        baseline_extractor.extract(&html_content)?
    };

    let article = ExtractedArticle {
        url: url.clone(),
        title: result.title,
        date: result.date,
        content: result.text,
        quality_score: result.quality_score,
        method: if model_path.is_some() { "rl" } else { "baseline" }.to_string(),
        xpath: Some(result.xpath),
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
        Some(DQNAgent::load_with_device(
            path,
            config.state_dim,
            config.num_discrete_actions,
            config.num_continuous_params,
            &device,
        )?)
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
                let method = if agent.is_some() {
                    if has_profile { "rl+profile" } else { "rl" }
                } else {
                    if has_profile { "baseline+profile" } else { "baseline" }
                };

                let article = ExtractedArticle {
                    url: url.clone(),
                    title: result.title.clone(),
                    date: result.date.clone(),
                    content: result.text.clone(),
                    quality_score: result.quality_score,
                    method: method.to_string(),
                    xpath: Some(result.xpath.clone()),
                };

                // Update site profile with this extraction
                let extraction_result = site_profile::ExtractionResult {
                    text: result.text,
                    xpath: result.xpath,
                    quality_score: result.quality_score,
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
            let without_protocol = if url.starts_with("https://") {
                &url[8..]
            } else if url.starts_with("http://") {
                &url[7..]
            } else {
                url
            };

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
}