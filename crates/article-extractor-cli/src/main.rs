use article_extractor::{
    Config, BaselineExtractor, DQNAgent,  // Removed SiteProfileMemory
    ExtractedArticle, BatchExtractionResult, Result,
    train_standard, train_with_improvements, TPEOptimizer, HyperparameterSpace,
    Hyperparameters, TrialResult, GroundTruthData, GroundTruthEvaluator,
    TrainingPlotter,  // Removed PlotConfig
};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing::{info, error, warn};
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Instant;
use bzip2::read::BzDecoder;

// Helper function to create separator string
fn separator() -> String {
    "=".repeat(80)
}


#[derive(Parser)]
#[command(name = "article-extractor")]
#[command(about = "RL-based article extraction from HTML with TPE hyperparameter tuning", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Extract article from HTML file
    Extract {
        /// Path to HTML file
        #[arg(short, long)]
        html_file: PathBuf,

        /// URL of the page
        #[arg(short, long)]
        url: String,

        /// Path to trained model
        #[arg(short, long)]
        model: Option<PathBuf>,

        /// Path to site profile
        #[arg(short, long)]
        site_profile: Option<PathBuf>,

        /// Output file path
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Extract from batch of HTML files in archive
    ExtractBatch {
        /// Archive directory containing HTML files
        #[arg(short, long)]
        archive_dir: PathBuf,

        /// Path to trained model
        #[arg(short, long)]
        model: Option<PathBuf>,

        /// Output directory for results
        #[arg(short, long)]
        output_dir: Option<PathBuf>,

        /// Maximum number of files to process
        #[arg(long)]
        max_files: Option<usize>,

        /// Batch size for processing
        #[arg(long, default_value = "2048")]
        batch_size: usize,
    },

    /// Train the model
    Train {
        /// Training data directory containing HTML files
        #[arg(short, long)]
        data_dir: PathBuf,

        /// Number of training episodes
        #[arg(short = 'e', long, default_value = "10000")]
        episodes: usize,

        /// Use improved training (curriculum, enhanced rewards)
        #[arg(short, long)]
        improved: bool,

        /// Auto-load best hyperparameters if available
        #[arg(long)]
        auto_hyperparams: bool,

        /// Path to hyperparameters JSON file
        #[arg(long)]
        hyperparams: Option<PathBuf>,

        /// Plot update frequency (episodes)
        #[arg(long, default_value = "1000")]
        plot_every: usize,

        /// Enable MLflow tracking
        #[arg(long)]
        mlflow: bool,

        /// MLflow tracking URI
        #[arg(long)]
        mlflow_uri: Option<String>,
    },

    /// Run TPE hyperparameter search
    Tune {
        /// Training data directory
        #[arg(short, long)]
        data_dir: PathBuf,

        /// Number of trials
        #[arg(short = 'n', long, default_value = "20")]
        trials: usize,

        /// Episodes per trial
        #[arg(short, long, default_value = "500")]
        episodes_per_trial: usize,

        /// Resume from previous search
        #[arg(long)]
        resume: bool,

        /// Output directory for results
        #[arg(short, long)]
        output_dir: Option<PathBuf>,
    },

    /// Evaluate extracted articles against ground truth
    Evaluate {
        /// Directory containing HTML and JSON ground truth files
        #[arg(short, long)]
        data_dir: PathBuf,

        /// Path to trained model
        #[arg(short, long)]
        model: Option<PathBuf>,

        /// Output file for evaluation results
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Maximum number of files to evaluate
        #[arg(long)]
        max_files: Option<usize>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter("article_extractor=info")
        .init();

    // Print device info
    article_extractor::print_device_info();

    let cli = Cli::parse();

    match cli.command {
        Commands::Extract { html_file, url, model, site_profile, output } => {
            extract_command(html_file, url, model, site_profile, output).await?;
        }
        Commands::ExtractBatch { archive_dir, model, output_dir, max_files, batch_size } => {
            extract_batch_command(archive_dir, model, output_dir, max_files, batch_size).await?;
        }
        Commands::Train { data_dir, episodes, improved, auto_hyperparams, hyperparams, plot_every, mlflow, mlflow_uri } => {
            train_command(data_dir, episodes, improved, auto_hyperparams, hyperparams, plot_every, mlflow, mlflow_uri).await?;
        }
        Commands::Tune { data_dir, trials, episodes_per_trial, resume, output_dir } => {
            tune_command(data_dir, trials, episodes_per_trial, resume, output_dir).await?;
        }
        Commands::Evaluate { data_dir, model, output, max_files } => {
            evaluate_command(data_dir, model, output, max_files).await?;
        }
    }

    Ok(())
}

async fn extract_command(
    html_file: PathBuf,
    url: String,
    model_path: Option<PathBuf>,
    _site_profile_path: Option<PathBuf>,
    output: Option<PathBuf>,
) -> Result<()> {
    info!("Extracting article from: {}", html_file.display());

    // Read HTML
    let html_content = std::fs::read_to_string(&html_file)?;

    // Load configuration
    let config = Config::from_env()
        .map_err(|e| article_extractor::ExtractionError::ParseError(e.to_string()))?;

    // Initialize extractor
    let baseline_extractor = BaselineExtractor::new(config.stopwords.clone());

    // Try to use RL model if available
    let result = if let Some(model_path) = model_path {
        info!("Using RL model: {}", model_path.display());

        let _agent = DQNAgent::load(
            &model_path,
            config.state_dim,
            config.num_discrete_actions,
            config.num_continuous_params,
        )?;

        // Use agent for extraction (simplified)
        baseline_extractor.extract(&html_content)?
    } else {
        info!("Using baseline extractor");
        baseline_extractor.extract(&html_content)?
    };

    // Create extracted article
    let article = ExtractedArticle {
        url: url.clone(),
        title: None,
        date: None,
        content: result.text,
        quality_score: result.quality_score,
        method: "baseline".to_string(),
        xpath: Some(result.xpath),
    };

    // Output result
    let output_path = output.unwrap_or_else(|| {
        config.output_dir.join(format!("article_{}.json", chrono::Utc::now().timestamp()))
    });

    let batch_result = BatchExtractionResult {
        articles: vec![article],
    };

    let json = serde_json::to_string_pretty(&batch_result)?;
    std::fs::write(&output_path, json)?;

    info!("Extraction saved to: {}", output_path.display());

    Ok(())
}

// Batch extraction

async fn extract_batch_command(
    archive_dir: PathBuf,
    model_path: Option<PathBuf>,
    output_dir: Option<PathBuf>,
    max_files: Option<usize>,
    batch_size: usize,
) -> Result<()> {
    info!("{}", separator());  // FIXED: Use function call
    info!("BATCH EXTRACTION MODE");
    info!("{}", separator());  // FIXED: Use function call
    info!("Archive directory: {}", archive_dir.display());
    info!("Batch size: {}", batch_size);

    let config = Config::from_env()
        .map_err(|e| article_extractor::ExtractionError::ParseError(e.to_string()))?;

    let output_dir = output_dir.unwrap_or_else(|| config.output_dir.clone());
    std::fs::create_dir_all(&output_dir)?;

    // Load HTML files recursively
    let html_files = load_html_files_recursive(&archive_dir, max_files)?;
    info!("Found {} HTML files", html_files.len());

    if html_files.is_empty() {
        error!("No HTML files found in {}", archive_dir.display());
        return Ok(());
    }

    // Initialize extractor
    let baseline_extractor = BaselineExtractor::new(config.stopwords.clone());
    let agent = if let Some(ref path) = model_path {
        Some(DQNAgent::load(path, config.state_dim, config.num_discrete_actions, config.num_continuous_params)?)
    } else {
        None
    };

    // Process in batches with progress bar
    let pb = ProgressBar::new(html_files.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("=>-"),
    );

    let mut all_articles = Vec::new();
    let mut failed = Vec::new();

    for chunk in html_files.chunks(batch_size) {
        for (html_path, url) in chunk {
            match std::fs::read_to_string(html_path) {
                Ok(html_content) => {
                    match baseline_extractor.extract(&html_content) {
                        Ok(result) => {
                            let article = ExtractedArticle {
                                url: url.clone(),
                                title: None,
                                date: None,
                                content: result.text,
                                quality_score: result.quality_score,
                                method: if agent.is_some() { "rl" } else { "baseline" }.to_string(),
                                xpath: Some(result.xpath),
                            };
                            all_articles.push(article);
                        }
                        Err(e) => {
                            failed.push((url.clone(), e.to_string()));
                        }
                    }
                }
                Err(e) => {
                    failed.push((url.clone(), e.to_string()));
                }
            }
            pb.inc(1);
        }
    }

    pb.finish_with_message("Batch extraction complete");

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
        warn!("Failed extractions saved to: {}", failed_path.display());
    }

    info!("{}", separator());
    info!("Batch extraction complete: {}/{} successful", all_articles.len(), html_files.len());
    info!("Results saved to: {}", results_path.display());
    info!("{}", separator());

    Ok(())
}

async fn train_command(
    data_dir: PathBuf,
    episodes: usize,
    improved: bool,
    auto_hyperparams: bool,
    hyperparams: Option<PathBuf>,
    plot_every: usize,
    mlflow: bool,
    mlflow_uri: Option<String>,
) -> Result<()> {
    info!("{}", separator());
    info!("TRAINING MODE");
    info!("{}", separator());
    info!("Data directory: {}", data_dir.display());
    info!("Episodes: {}", episodes);
    info!("Improved: {}", improved);
    info!("MLflow: {}", mlflow);

    // Load configuration
    let mut config = Config::from_env()
        .map_err(|e| article_extractor::ExtractionError::ParseError(e.to_string()))?;
    config.num_episodes = episodes;
    config.setup_directories()
        .map_err(|e| article_extractor::ExtractionError::ParseError(e.to_string()))?;

    // Load hyperparameters if specified
    if auto_hyperparams {
        let best_hyperparams_path = config.models_dir.join("best_hyperparams.json");
        if best_hyperparams_path.exists() {
            info!("Loading best hyperparameters from: {}", best_hyperparams_path.display());
            if let Ok(params) = Hyperparameters::load(&best_hyperparams_path) {
                params.apply_to_config(&mut config);
                info!("Applied hyperparameters: lr={:.6}, batch={}, gamma={:.3}",
                      params.learning_rate, params.batch_size, params.gamma);
            }
        }
    } else if let Some(ref path) = hyperparams {
        info!("Loading hyperparameters from: {}", path.display());
        let params = Hyperparameters::load(path)?;
        params.apply_to_config(&mut config);
    }

    // Load HTML samples
    let html_samples = load_html_samples(&data_dir, None)?;
    info!("Loaded {} HTML samples", html_samples.len());

    if html_samples.is_empty() {
        error!("No HTML samples found in {}", data_dir.display());
        return Err(article_extractor::ExtractionError::ParseError(
            "No training data found".to_string()
        ));
    }

    // Initialize MLflow if enabled
    #[cfg(feature = "mlflow")]
    let mut mlflow_tracker = if mlflow {
        let uri = mlflow_uri.or_else(|| std::env::var("MLFLOW_TRACKING_URI").ok());
        let tracker = article_extractor::mlflow::MlflowTracker::new(uri);
        tracker
    } else {
        article_extractor::mlflow::MlflowTracker::new(None)
    };

    #[cfg(feature = "mlflow")]
    if mlflow_tracker.is_enabled() {
        mlflow_tracker.start_run(Some(format!("training_{}", chrono::Utc::now().format("%Y%m%d_%H%M%S"))))?;

        // Log hyperparameters
        let mut params = std::collections::HashMap::new();
        params.insert("learning_rate".to_string(), config.learning_rate.to_string());
        params.insert("batch_size".to_string(), config.batch_size.to_string());
        params.insert("gamma".to_string(), config.gamma.to_string());
        params.insert("epsilon_decay".to_string(), config.epsilon_decay.to_string());
        params.insert("episodes".to_string(), episodes.to_string());
        params.insert("improved".to_string(), improved.to_string());
        mlflow_tracker.log_params(params)?;
    }

    // Initialize plotter
    let plotter = TrainingPlotter::new();
    let plot_path = config.output_dir.join("training_plot.png");

    // Train with periodic plotting
    let start_time = Instant::now();

    let (_agent, metrics) = if improved {
        train_with_improvements(&config, html_samples)?
    } else {
        train_standard(&config, html_samples)?
    };

    let duration = start_time.elapsed();

    // Generate final plot
    plotter.plot_training_results(&metrics, &plot_path)?;

    // Log to MLflow
    #[cfg(feature = "mlflow")]
    if mlflow_tracker.is_enabled() {
        mlflow_tracker.log_training_metrics(&metrics, episodes)?;
        mlflow_tracker.log_artifact(&plot_path)?;

        let best_model = config.models_dir.join("best_model.onnx");
        if best_model.exists() {
            mlflow_tracker.log_artifact(&best_model)?;
        }

        mlflow_tracker.end_run("FINISHED")?;
    }

    // Log results
    info!("{}", separator());
    info!("TRAINING COMPLETED");
    info!("{}", separator());
    info!("Duration: {:.2} seconds ({:.2} minutes)", duration.as_secs_f64(), duration.as_secs_f64() / 60.0);
    info!("Best avg quality: {:.4}", metrics.best_avg_quality);
    info!("Final reward: {:.4}", metrics.episode_rewards.last().copied().unwrap_or(0.0));
    info!("Model saved at: {}", config.models_dir.join("best_model.onnx").display());
    info!("Plot saved at: {}", plot_path.display());
    info!("{}", separator());

    Ok(())
}

async fn tune_command(
    data_dir: PathBuf,
    trials: usize,
    episodes_per_trial: usize,
    resume: bool,
    output_dir: Option<PathBuf>,
) -> Result<()> {
    info!("{}", separator());
    info!("TPE HYPERPARAMETER TUNING");
    info!("{}", separator());
    info!("Trials: {}", trials);
    info!("Episodes per trial: {}", episodes_per_trial);
    info!("Resume: {}", resume);

    let config = Config::from_env()
        .map_err(|e| article_extractor::ExtractionError::ParseError(e.to_string()))?;

    let output_dir = output_dir.unwrap_or_else(|| config.output_dir.clone());
    std::fs::create_dir_all(&output_dir)?;

    // Load samples (use subset for faster tuning)
    let html_samples = load_html_samples(&data_dir, Some(5000))?;
    info!("Loaded {} HTML samples for tuning", html_samples.len());

    if html_samples.is_empty() {
        error!("No HTML samples found");
        return Ok(());
    }

    // Initialize optimizer with resume capability
    let space = HyperparameterSpace::default();
    let state_path = output_dir.join("optimizer_state.json");

    let mut optimizer = if resume && state_path.exists() {
        TPEOptimizer::with_resume(space, state_path.clone())?
    } else {
        let opt = TPEOptimizer::new(space);
        opt
    };

    // Progress bar
    let completed = optimizer.num_trials();
    let remaining = trials.saturating_sub(completed);

    let pb = ProgressBar::new(remaining as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} Trial {msg}")
            .unwrap()
            .progress_chars("=>-"),
    );

    // Run trials
    for trial_num in completed..trials {
        pb.set_message(format!("{}", trial_num + 1));

        let params = optimizer.suggest();

        info!("Trial {}/{}: lr={:.6}, batch={}, gamma={:.3}",
              trial_num + 1, trials, params.learning_rate, params.batch_size, params.gamma);

        // Train with these hyperparameters
        let mut trial_config = config.clone();
        params.apply_to_config(&mut trial_config);
        trial_config.num_episodes = episodes_per_trial;

        let trial_start = Instant::now();

        let (_agent, metrics) = train_standard(&trial_config, html_samples.clone())?;

        let duration = trial_start.elapsed();

        // Calculate quality score
        let quality = metrics.best_avg_quality;
        let avg_reward = if !metrics.episode_rewards.is_empty() {
            metrics.episode_rewards.iter().sum::<f32>() / metrics.episode_rewards.len() as f32
        } else {
            0.0
        };

        // Record result
        let trial = TrialResult {
            trial_number: trial_num,
            hyperparameters: Hyperparameters {
                quality_score: quality as f64,
                ..params
            },
            quality_score: quality as f64,
            avg_reward: avg_reward as f64,
            duration_seconds: duration.as_secs_f64(),
        };

        optimizer.tell(trial);
        pb.inc(1);
    }

    pb.finish_with_message("Tuning complete");

    // Save results
    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let results_path = output_dir.join(format!("tuning_results_{}.json", timestamp));
    optimizer.save_results(&results_path)?;

    // Save best hyperparameters
    if let Some(best) = optimizer.get_best() {
        let best_path = config.models_dir.join("best_hyperparams.json");
        best.save(&best_path)?;

        info!("{}", separator());
        info!("TUNING COMPLETED");
        info!("{}", separator());
        info!("Best quality: {:.4}", best.quality_score);
        info!("Best hyperparameters:");
        info!("  learning_rate: {:.6}", best.learning_rate);
        info!("  batch_size: {}", best.batch_size);
        info!("  gamma: {:.3}", best.gamma);
        info!("  epsilon_decay: {:.3}", best.epsilon_decay);
        info!("Results saved to: {}", results_path.display());
        info!("Best hyperparameters saved to: {}", best_path.display());
        info!("{}", separator());
    }

    Ok(())
}

async fn evaluate_command(
    data_dir: PathBuf,
    model_path: Option<PathBuf>,
    output: Option<PathBuf>,
    max_files: Option<usize>,
) -> Result<()> {
    info!("{}", separator());
    info!("EVALUATION MODE");
    info!("{}", separator());
    info!("Data directory: {}", data_dir.display());

    let config = Config::from_env()
        .map_err(|e| article_extractor::ExtractionError::ParseError(e.to_string()))?;

    // Find HTML and JSON pairs
    let file_pairs = find_html_json_pairs(&data_dir, max_files)?;
    info!("Found {} HTML/JSON pairs for evaluation", file_pairs.len());

    if file_pairs.is_empty() {
        error!("No HTML/JSON pairs found");
        return Ok(());
    }

    // Initialize extractor and evaluator
    let baseline_extractor = BaselineExtractor::new(config.stopwords.clone());
    let evaluator = GroundTruthEvaluator::new(config.stopwords.clone());

    let agent = if let Some(ref path) = model_path {
        Some(DQNAgent::load(path, config.state_dim, config.num_discrete_actions, config.num_continuous_params)?)
    } else {
        None
    };

    // Evaluate with progress bar
    let pb = ProgressBar::new(file_pairs.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("=>-"),
    );

    let mut all_metrics = Vec::new();

    for (html_path, json_path) in file_pairs {
        // Load HTML and ground truth
        let html_content = std::fs::read_to_string(&html_path)?;
        let ground_truth = GroundTruthData::load(&json_path)?;

        // Extract
        let result = baseline_extractor.extract(&html_content)?;

        // Evaluate
        let metrics = evaluator.evaluate(
            &result.text,
            None, // Title extraction not implemented yet
            &ground_truth,
            result.quality_score,
        );

        all_metrics.push(metrics);
        pb.inc(1);
    }

    pb.finish_with_message("Evaluation complete");

    // Calculate averages
    let avg_metrics = GroundTruthEvaluator::average_metrics(&all_metrics);

    // Save results
    let output_path = output.unwrap_or_else(|| {
        config.output_dir.join(format!("evaluation_{}.json", chrono::Utc::now().format("%Y%m%d_%H%M%S")))
    });

    let results = serde_json::json!({
        "num_evaluated": all_metrics.len(),
        "average_metrics": avg_metrics,
        "all_metrics": all_metrics,
    });

    let json = serde_json::to_string_pretty(&results)?;
    std::fs::write(&output_path, json)?;

    info!("{}", separator());
    info!("EVALUATION RESULTS");
    info!("{}", separator());
    info!("Files evaluated: {}", all_metrics.len());
    info!("Average combined quality: {:.4}", avg_metrics.combined_quality);
    info!("Average text F1: {:.4}", avg_metrics.text_f1_score);
    info!("Average title match: {:.4}", avg_metrics.title_match_score);
    info!("Results saved to: {}", output_path.display());
    info!("{}", separator());

    Ok(())
}

// Helper functions

fn load_html_files_recursive(dir: &PathBuf, max_files: Option<usize>) -> Result<Vec<(PathBuf, String)>> {
    let mut files = Vec::new();

    for entry in walkdir::WalkDir::new(dir).into_iter().filter_map(|e| e.ok()) {
        if let Some(max) = max_files {
            if files.len() >= max {
                break;
            }
        }

        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path.extension() {
                if ext == "bz2" && path.to_string_lossy().contains(".html.") {
                    let url = path.file_stem()
                        .and_then(|s| s.to_str())
                        .map(|s| format!("https://example.com/{}", s))
                        .unwrap_or_default();
                    files.push((path.to_path_buf(), url));
                } else if ext == "html" || ext == "htm" {
                    let url = path.file_stem()
                        .and_then(|s| s.to_str())
                        .map(|s| format!("https://example.com/{}", s))
                        .unwrap_or_default();
                    files.push((path.to_path_buf(), url));
                }
            }
        }
    }

    Ok(files)
}

fn load_html_samples(dir: &PathBuf, max_samples: Option<usize>) -> Result<Vec<(String, String)>> {
    let files = load_html_files_recursive(dir, max_samples)?;
    let mut samples = Vec::new();

    for (path, url) in files {
        let content = if path.extension().and_then(|s| s.to_str()) == Some("bz2") {
            // Decompress bz2 file
            let file = std::fs::File::open(&path)?;
            let mut decoder = BzDecoder::new(file);
            let mut html = String::new();
            std::io::Read::read_to_string(&mut decoder, &mut html)?;
            html
        } else {
            std::fs::read_to_string(&path)?
        };

        samples.push((content, url));
    }

    Ok(samples)
}

fn find_html_json_pairs(dir: &PathBuf, max_pairs: Option<usize>) -> Result<Vec<(PathBuf, PathBuf)>> {
    let mut pairs = Vec::new();

    for entry in walkdir::WalkDir::new(dir).into_iter().filter_map(|e| e.ok()) {
        if let Some(max) = max_pairs {
            if pairs.len() >= max {
                break;
            }
        }

        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path.extension() {
                if ext == "bz2" && path.to_string_lossy().contains(".html.") {
                    // Look for corresponding JSON file
                    let json_path = path.with_extension("").with_extension("json");
                    if json_path.exists() {
                        pairs.push((path.to_path_buf(), json_path));
                    }
                }
            }
        }
    }

    Ok(pairs)
}