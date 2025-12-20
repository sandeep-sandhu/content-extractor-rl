use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use crate::Result;
use sha2::{Sha256, Digest};

/// Site profile storing historical extraction patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiteProfile {
    pub domain: String,
    pub extractions: Vec<ExtractionRecord>,
    pub successful_xpaths: HashMap<String, usize>,
    pub avg_parameters: HashMap<String, Vec<f64>>,
    pub quality_scores: Vec<f32>,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionRecord {
    pub timestamp: DateTime<Utc>,
    pub quality_score: f32,
    pub xpath: String,
    pub parameters: HashMap<String, f64>,
    pub text_length: usize,
}

impl SiteProfile {
    /// Create new site profile
    pub fn new(domain: String) -> Self {
        Self {
            domain,
            extractions: Vec::new(),
            successful_xpaths: HashMap::new(),
            avg_parameters: HashMap::new(),
            quality_scores: Vec::new(),
            last_updated: Utc::now(),
        }
    }

    /// Add extraction result to profile
    pub fn add_extraction(&mut self, result: ExtractionResult) {
        let record = ExtractionRecord {
            timestamp: Utc::now(),
            quality_score: result.quality_score,
            xpath: result.xpath.clone(),
            parameters: result.parameters.clone(),
            text_length: result.text.len(),
        };

        self.extractions.push(record);

        // Update statistics for successful extractions
        if result.quality_score > 0.7 {
            if !result.xpath.is_empty() {
                *self.successful_xpaths.entry(result.xpath.clone()).or_insert(0) += 1;
            }

            for (key, value) in result.parameters.iter() {
                self.avg_parameters.entry(key.clone())
                    .or_insert_with(Vec::new)
                    .push(*value);
            }
        }

        self.quality_scores.push(result.quality_score);
        self.last_updated = Utc::now();

        // Keep only recent extractions (last 1000)
        if self.extractions.len() > 1000 {
            self.extractions = self.extractions.split_off(self.extractions.len() - 1000);
        }
    }

    /// Get most successful XPath pattern
    pub fn get_best_xpath(&self) -> Option<&String> {
        self.successful_xpaths.iter()
            .max_by_key(|(_, count)| *count)
            .map(|(xpath, _)| xpath)
    }

    /// Get recommended parameters (median values)
    pub fn get_recommended_parameters(&self) -> HashMap<String, f64> {
        let mut recommended = HashMap::new();

        for (param, values) in self.avg_parameters.iter() {
            if !values.is_empty() {
                let mut sorted = values.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median = sorted[sorted.len() / 2];
                recommended.insert(param.clone(), median);
            }
        }

        recommended
    }

    /// Calculate success rate
    pub fn get_success_rate(&self) -> f32 {
        if self.quality_scores.is_empty() {
            return 0.0;
        }

        let successful = self.quality_scores.iter()
            .filter(|&&score| score > 0.7)
            .count();

        successful as f32 / self.quality_scores.len() as f32
    }

    /// Save profile to file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load profile from file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let profile = serde_json::from_str(&json)?;
        Ok(profile)
    }
}

/// Result of an extraction operation
#[derive(Debug, Clone)]
pub struct ExtractionResult {
    pub text: String,
    pub xpath: String,
    pub quality_score: f32,
    pub parameters: HashMap<String, f64>,
}

/// Site profile memory manager
pub struct SiteProfileMemory {
    storage_dir: PathBuf,
    cache: HashMap<String, SiteProfile>,
}

impl SiteProfileMemory {
    /// Create new site profile memory
    pub fn new<P: AsRef<Path>>(storage_dir: P) -> Result<Self> {
        let storage_dir = storage_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&storage_dir)?;

        Ok(Self {
            storage_dir,
            cache: HashMap::new(),
        })
    }

    /// Get or create profile for domain
    pub fn get_profile(&mut self, domain: &str) -> &mut SiteProfile {
        if !self.cache.contains_key(domain) {
            let profile_path = self.get_profile_path(domain);

            let profile = if profile_path.exists() {
                SiteProfile::load(&profile_path).unwrap_or_else(|_| SiteProfile::new(domain.to_string()))
            } else {
                SiteProfile::new(domain.to_string())
            };

            self.cache.insert(domain.to_string(), profile);
        }

        self.cache.get_mut(domain).unwrap()
    }

    /// Save profile to disk
    pub fn save_profile(&self, domain: &str) -> Result<()> {
        if let Some(profile) = self.cache.get(domain) {
            let profile_path = self.get_profile_path(domain);
            profile.save(profile_path)?;
        }
        Ok(())
    }

    /// Save all cached profiles
    pub fn save_all(&self) -> Result<()> {
        for (domain, profile) in self.cache.iter() {
            let profile_path = self.get_profile_path(domain);
            profile.save(profile_path)?;
        }
        Ok(())
    }

    /// Get profile file path
    fn get_profile_path(&self, domain: &str) -> PathBuf {
        let hash = hash_domain(domain);
        self.storage_dir.join(format!("{}.json", hash))
    }
}

/// Hash domain name to create filename
fn hash_domain(domain: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain.as_bytes());
    let result = hasher.finalize();
    format!("{:x}", result)[..16].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_site_profile() {
        let mut profile = SiteProfile::new("example.com".to_string());

        let result = ExtractionResult {
            text: "Test content".to_string(),
            xpath: "//article[1]".to_string(),
            quality_score: 0.8,
            parameters: HashMap::new(),
        };

        profile.add_extraction(result);

        assert_eq!(profile.quality_scores.len(), 1);
        assert_eq!(profile.extractions.len(), 1);
    }

    #[test]
    fn test_profile_memory() {
        let temp_dir = TempDir::new().unwrap();
        let mut memory = SiteProfileMemory::new(temp_dir.path()).unwrap();

        let profile = memory.get_profile("example.com");
        assert_eq!(profile.domain, "example.com");

        memory.save_all().unwrap();
    }
}
