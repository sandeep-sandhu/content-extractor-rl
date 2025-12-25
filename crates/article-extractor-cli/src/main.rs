use article_extractor::{
    Config, BaselineExtractor, DQNAgent,
    ExtractedArticle, BatchExtractionResult, Result,
    train_standard, train_with_improvements, TPEOptimizer, HyperparameterSpace,
    Hyperparameters, TrialResult, GroundTruthData, GroundTruthEvaluator,
    TrainingPlotter,
};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use indicatif::{ProgressBar, ProgressStyle};
use bzip2::read::BzDecoder;
use std::env;
use std::time::Instant;
use tracing::{info, error, warn};
use tracing_subscriber::{fmt, prelude::*};
use tracing_subscriber::EnvFilter;
use tracing_appender::{non_blocking, rolling};
use chrono::{Local};
use std::error::Error;

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

        /// Performance mode: "default", "fast", "gpu", "rtx3080"
        #[arg(long, default_value = "default")]
        perf_mode: String,

        /// Maximum dataset samples (CRITICAL for performance)
        #[arg(long, default_value = "5000")]
        max_samples: usize,

        /// Custom batch size override
        #[arg(long)]
        batch_size: Option<usize>,

        /// Training frequency (train every N steps)
        #[arg(long)]
        train_freq: Option<usize>,

        /// Gradient updates per episode
        #[arg(long)]
        train_steps_per_episode: Option<usize>,

        /// Metrics window size
        #[arg(long)]
        metrics_window: Option<usize>,

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

        /// Maximum samples for tuning (smaller = faster)
        #[arg(long, default_value = "3000")]
        max_samples: usize,

        /// Use CPU for tuning (avoid GPU memory issues)
        #[arg(long)]
        use_cpu: bool,
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


// Helper function to create separator string
fn separator() -> String {
    "=".repeat(80)
}


fn setup_logging(command_type: &str) -> std::result::Result<non_blocking::WorkerGuard, Box<dyn Error>> {

    // Create log directory
    let log_dir = env::current_dir()?.join("logs");
    std::fs::create_dir_all(&log_dir)?;

    // Create timestamp for log file (using local time for filename)
    let timestamp = Local::now().format("%Y%m%d_%H%M%S");
    // Use consistent naming based on command type
    let log_file = match command_type {
        "train" => format!("training_{}.log", timestamp),
        "tune" => format!("tuning_{}.log", timestamp),
        "extract" => format!("extraction_{}.log", timestamp),
        "extract_batch" => format!("batch_extraction_{}.log", timestamp),
        "evaluate" => format!("evaluation_{}.log", timestamp),
        _ => format!("article_extractor_{}.log", timestamp),
    };
    // Set up file appender
    let file_appender = rolling::never(&log_dir, log_file);
    let (non_blocking_file, file_guard) = non_blocking(file_appender);

    // Set up console appender
    let (non_blocking_console, _console_guard) = non_blocking(std::io::stdout());

    // Configure file logging layer with UTC time but readable format
    let file_layer = fmt::layer()
        .with_writer(non_blocking_file)
        .with_ansi(false)
        .with_target(true)
        .with_level(true)
        //.with_line_number(true)
        //.with_file(true)
        .with_thread_ids(false)
        .with_timer(fmt::time::UtcTime::rfc_3339()); // Use RFC3339 format for UTC

    // Configure console logging layer with local time-like format
    let console_layer = fmt::layer()
        .with_writer(non_blocking_console)
        .with_ansi(true)
        .with_target(false)  // Hide target for cleaner console output
        .with_level(true)
        //.with_line_number(false)
        //.with_file(false)
        .with_thread_ids(false)
        .with_timer(fmt::time::UtcTime::rfc_3339());

    // Initialize tracing with custom filter to suppress html5ever warnings
    tracing_subscriber::registry()
        .with(file_layer)
        .with(console_layer)
        .with(tracing_subscriber::EnvFilter::from_default_env()
            // Suppress html5ever warnings
            .add_directive("html5ever=error".parse().unwrap())
            .add_directive("article_extractor=info".parse()?)
            .add_directive("warn".parse().unwrap())
            .add_directive("info".parse()?))
        .init();

    Ok(file_guard)
}


#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let command_type = match &cli.command {
        Commands::Train { .. } => "train",
        Commands::Tune { .. } => "tune",
        Commands::Extract { .. } => "extract",
        Commands::ExtractBatch { .. } => "extract_batch",
        Commands::Evaluate { .. } => "evaluate",
    };

    // Set up logging - keep guard to prevent early drop
    let _log_guard = setup_logging(command_type).map_err(|e| {
        article_extractor::ExtractionError::ParseError(format!("Failed to setup logging: {}", e))
    })?;

    // Print device info
    article_extractor::print_device_info();

    match cli.command {
        Commands::Extract { html_file, url, model, site_profile, output } => {
            extract_command(html_file, url, model, site_profile, output).await?;
        }
        Commands::ExtractBatch { archive_dir, model, output_dir, max_files, batch_size } => {
            extract_batch_command(archive_dir, model, output_dir, max_files, batch_size).await?;
        }
        Commands::Train {
            data_dir, episodes, improved, auto_hyperparams, hyperparams, plot_every,
            perf_mode, max_samples, batch_size, train_freq, train_steps_per_episode,
            metrics_window, mlflow, mlflow_uri
        } => {
            train_command(
                data_dir, episodes, improved, auto_hyperparams, hyperparams, plot_every,
                perf_mode, max_samples, batch_size, train_freq, train_steps_per_episode,
                metrics_window, mlflow, mlflow_uri
            ).await?;
        }
        Commands::Tune { data_dir, trials, episodes_per_trial, resume, output_dir, max_samples, use_cpu } => {
            tune_command(data_dir, trials, episodes_per_trial, resume, output_dir, max_samples, use_cpu).await?;
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
    let config = Config::from_env()
        .map_err(|e| article_extractor::ExtractionError::ParseError(e.to_string()))?;

    let article = article_extractor::extract_single(
        &html_file,
        url,
        model_path.as_deref(),
        output.as_deref(),
        &config,
    )?;

    info!("Extracted article with quality: {:.3}", article.quality_score);
    Ok(())
}

async fn extract_batch_command(
    archive_dir: PathBuf,
    model_path: Option<PathBuf>,
    output_dir: Option<PathBuf>,
    max_files: Option<usize>,
    batch_size: usize,
) -> Result<()> {
    let config = Config::from_env()
        .map_err(|e| article_extractor::ExtractionError::ParseError(e.to_string()))?;

    let output_dir = output_dir.unwrap_or_else(|| config.output_dir.clone());

    let result = article_extractor::extract_batch(
        &archive_dir,
        model_path.as_deref(),
        &output_dir,
        max_files,
        batch_size,
        &config,
    )?;

    info!("Extracted {} articles", result.articles.len());
    Ok(())
}

// Helper function to read URL from JSON file
fn get_url_from_json(json_path: &PathBuf) -> String {
    match std::fs::read_to_string(json_path) {
        Ok(json_content) => {
            if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(&json_content) {
                json_value.get("URL")
                    .and_then(|u| u.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "https://unknown/unknown".to_string())
            } else {
                "https://unknown/invalid-json".to_string()
            }
        }
        Err(_) => {
            "https://unknown/no-json".to_string()
        }
    }
}

async fn train_command(
    data_dir: PathBuf,
    episodes: usize,
    improved: bool,
    auto_hyperparams: bool,
    hyperparams: Option<PathBuf>,
    _plot_every: usize,
    perf_mode: String,
    max_samples: usize,
    batch_size_override: Option<usize>,
    train_freq_override: Option<usize>,
    train_steps_override: Option<usize>,
    metrics_window_override: Option<usize>,
    mlflow: bool,
    _mlflow_uri: Option<String>,
) -> Result<()> {
    info!("{}", separator());
    info!("TRAINING MODE");
    info!("{}", separator());
    info!("Data directory: {}", data_dir.display());
    info!("Episodes: {}", episodes);
    info!("Improved: {}", improved);
    info!("Performance mode: {}", perf_mode);
    info!("Max samples: {}", max_samples);
    info!("MLflow: {}", mlflow);

    // Select config based on performance mode
    let mut config = match perf_mode.as_str() {
        "fast" | "gpu" => {
            info!("Using GPU-optimized configuration");
            Config::gpu_optimized()
        }
        _ => {
            info!("Using default configuration");
            Config::default()
        }
    };

    // Apply episode count and max samples
    config.num_episodes = episodes;
    config.max_html_samples = max_samples;

    config.setup_directories()
        .map_err(|e| article_extractor::ExtractionError::ParseError(e.to_string()))?;

    // Load hyperparameters if specified
    let hyperparams_loaded = if auto_hyperparams {
        let best_hyperparams_path = config.models_dir.join("best_hyperparams.json");
        if best_hyperparams_path.exists() {
            info!("Loading best hyperparameters from: {}", best_hyperparams_path.display());
            if let Ok(params) = Hyperparameters::load(&best_hyperparams_path) {
                params.apply_to_config(&mut config);
                info!("Applied hyperparameters: lr={:.6}, batch={}, gamma={:.3}",
                      params.learning_rate, params.batch_size, params.gamma);
                true
            } else {
                false
            }
        } else {
            false
        }
    } else if let Some(ref path) = hyperparams {
        info!("Loading hyperparameters from: {}", path.display());
        let params = Hyperparameters::load(path)?;
        params.apply_to_config(&mut config);
        info!("Applied hyperparameters: lr={:.6}, batch={}, gamma={:.3}",
              params.learning_rate, params.batch_size, params.gamma);
        true
    } else {
        false
    };

    // Apply CLI overrides (these take precedence)
    if let Some(batch_size) = batch_size_override {
        info!("Overriding batch size to: {}", batch_size);
        config.batch_size = batch_size;
    } else if hyperparams_loaded && perf_mode != "default" {
        // Restore optimized batch size if hyperparams overwrote it
        let optimized_batch = match perf_mode.as_str() {
            "rtx3080" => 2048,
            "gpu" | "fast" => 1024,
            _ => config.batch_size,
        };
        if config.batch_size > 4096 || config.batch_size < 256 {
            warn!("Hyperparams batch size {} seems wrong for perf_mode={}, using {}",
                  config.batch_size, perf_mode, optimized_batch);
            config.batch_size = optimized_batch;
        }
    }

    if let Some(train_freq) = train_freq_override {
        info!("Overriding train frequency to: {}", train_freq);
        config.train_freq = train_freq;
    }

    if let Some(train_steps) = train_steps_override {
        info!("Overriding train steps per episode to: {}", train_steps);
        config.num_train_steps_per_episode = train_steps;
    }

    if let Some(window) = metrics_window_override {
        info!("Overriding metrics window to: {}", window);
        config.metrics_window = window;
    }

    // Log final performance configuration
    info!("{}", separator());
    info!("PERFORMANCE CONFIGURATION");
    info!("{}", separator());
    info!("Batch size: {}", config.batch_size);
    info!("Train frequency: every {} steps", config.train_freq);
    info!("Gradient updates per episode: {}", config.num_train_steps_per_episode);
    info!("Min replay size: {}", config.min_replay_size);
    info!("Metrics window: {}", config.metrics_window);
    info!("Max HTML samples: {}", config.max_html_samples);
    info!("{}", separator());

    // Load HTML samples with optimization
    info!("Loading HTML samples...");
    let load_start = Instant::now();
    let html_samples = load_html_samples(&data_dir, Some(config.max_html_samples))?;
    let load_duration = load_start.elapsed();
    info!("Loaded {} HTML samples in {:.2}s", html_samples.len(), load_duration.as_secs_f64());

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
        article_extractor::mlflow::MlflowTracker::new(uri)
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
        params.insert("perf_mode".to_string(), perf_mode.clone());
        params.insert("train_freq".to_string(), config.train_freq.to_string());
        params.insert("train_steps_per_episode".to_string(), config.num_train_steps_per_episode.to_string());
        mlflow_tracker.log_params(params)?;
    }

    // Initialize plotter
    let plotter = TrainingPlotter::new();
    let plot_path = config.output_dir.join("training_plot.png");

    // Train
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
    info!("Duration: {:.2} seconds ({:.2} minutes, {:.2} hours)",
          duration.as_secs_f64(),
          duration.as_secs_f64() / 60.0,
          duration.as_secs_f64() / 3600.0);
    info!("Episodes per second: {:.2}", episodes as f64 / duration.as_secs_f64());
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
    max_samples: usize,
    use_cpu: bool,
) -> Result<()> {
    info!("{}", separator());
    info!("TPE HYPERPARAMETER TUNING");
    info!("{}", separator());
    info!("Trials: {}", trials);
    info!("Episodes per trial: {}", episodes_per_trial);
    info!("Max samples: {}", max_samples);
    info!("Resume: {}", resume);
    info!("Use CPU: {}", use_cpu);

    let mut config = Config::default();
    config.use_cpu_for_tuning = use_cpu;

    let output_dir = output_dir.unwrap_or_else(|| config.output_dir.clone());
    std::fs::create_dir_all(&output_dir)?;

    // Load samples with limit for faster tuning
    info!("Loading HTML samples for tuning...");
    let html_samples = load_html_samples(&data_dir, Some(max_samples))?;
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
        TPEOptimizer::new(space)
    };

    // Progress bar
    let completed = optimizer.num_trials();
    let remaining = trials.saturating_sub(completed);

    if completed > 0 {
        info!("Resuming from trial {}/{}", completed, trials);
    }

    let pb = ProgressBar::new(remaining as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} Trial {msg}")
            .unwrap()
            .progress_chars("█▓▒░"),
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
        trial_config.max_html_samples = max_samples;

        let trial_start = Instant::now();

        let (_agent, metrics) = train_standard(&trial_config, html_samples.clone())?;

        let duration = trial_start.elapsed();

        // Calculate quality score (use smaller window for faster feedback)
        let window = metrics.episode_qualities.len().min(50);
        let quality = if metrics.episode_qualities.len() >= window {
            metrics.episode_qualities[metrics.episode_qualities.len() - window..]
                .iter()
                .sum::<f32>() / window as f32
        } else if !metrics.episode_qualities.is_empty() {
            metrics.episode_qualities.iter().sum::<f32>() / metrics.episode_qualities.len() as f32
        } else {
            0.0
        };

        let avg_reward = if !metrics.episode_rewards.is_empty() {
            let window = metrics.episode_rewards.len().min(50);
            if metrics.episode_rewards.len() >= window {
                metrics.episode_rewards[metrics.episode_rewards.len() - window..]
                    .iter()
                    .sum::<f32>() / window as f32
            } else {
                metrics.episode_rewards.iter().sum::<f32>() / metrics.episode_rewards.len() as f32
            }
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
        info!("  priority_alpha: {:.3}", best.priority_alpha);
        info!("  priority_beta: {:.3}", best.priority_beta);
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

    let _agent = if let Some(ref path) = model_path {
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

fn load_html_files_recursive(dir: &PathBuf, max_files: Option<usize>) -> Result<Vec<(PathBuf, PathBuf)>> {
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
                    // Look for corresponding JSON file
                    let json_path = path.with_extension("").with_extension("json");
                    if json_path.exists() {
                        files.push((path.to_path_buf(), json_path));
                    }
                } else if ext == "html" || ext == "htm" {
                    // Look for corresponding JSON file
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

fn load_html_samples(dir: &PathBuf, max_samples: Option<usize>) -> Result<Vec<(String, String)>> {
    let files = load_html_files_recursive(dir, max_samples)?;
    let mut samples = Vec::new();

    for (html_path, json_path) in files {
        // Read HTML content
        let content = if html_path.extension().and_then(|s| s.to_str()) == Some("bz2") {
            // Decompress bz2 file
            let file = std::fs::File::open(&html_path)?;
            let mut decoder = BzDecoder::new(file);
            let mut html = String::new();
            std::io::Read::read_to_string(&mut decoder, &mut html)?;
            html
        } else {
            std::fs::read_to_string(&html_path)?
        };

        // Read URL from JSON file
        let url = if let Ok(json_content) = std::fs::read_to_string(&json_path) {
            if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(&json_content) {
                json_value.get("URL")
                    .and_then(|u| u.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "https://example.com/unknown".to_string())
            } else {
                "https://example.com/invalid-json".to_string()
            }
        } else {
            "https://example.com/no-json".to_string()
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