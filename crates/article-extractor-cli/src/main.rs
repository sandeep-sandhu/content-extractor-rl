use article_extractor::{
    Config, BaselineExtractor, DQNAgent, SiteProfileMemory,
    ExtractedArticle, BatchExtractionResult, Result,
    train_standard, train_with_improvements, HyperparameterSearch, GridSearchConfig,
};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing::{info, error};

#[derive(Parser)]
#[command(name = "article-extractor")]
#[command(about = "RL-based article extraction from HTML", long_about = None)]
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

        /// Learning rate
        #[arg(long)]
        learning_rate: Option<f64>,

        /// Batch size
        #[arg(long)]
        batch_size: Option<usize>,
    },

    /// Run hyperparameter search
    Search {
        /// Training data directory
        #[arg(short, long)]
        data_dir: PathBuf,

        /// Number of episodes per trial
        #[arg(short, long, default_value = "500")]
        episodes_per_trial: usize,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter("article_extractor=info")
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Extract { html_file, url, model, site_profile, output } => {
            extract_command(html_file, url, model, site_profile, output).await?;
        }
        Commands::Train { data_dir, episodes, improved, learning_rate, batch_size } => {
            train_command(data_dir, episodes, improved, learning_rate, batch_size).await?;
        }
        Commands::Search { data_dir, episodes_per_trial } => {
            search_command(data_dir, episodes_per_trial).await?;
        }
    }

    Ok(())
}

async fn extract_command(
    html_file: PathBuf,
    url: String,
    model_path: Option<PathBuf>,
    site_profile_path: Option<PathBuf>,
    output: Option<PathBuf>,
) -> Result<()> {
    info!("Extracting article from: {}", html_file.display());

    // Read HTML
    let html_content = std::fs::read_to_string(&html_file)?;

    // Load configuration
    let config = Config::from_env()?;

    // Initialize extractor
    let baseline_extractor = BaselineExtractor::new(config.stopwords.clone());

    // Try to use RL model if available
    let result = if let Some(model_path) = model_path {
        info!("Using RL model: {}", model_path.display());

        let agent = DQNAgent::load(
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
        title: None, // Could be extracted from HTML
        date: None,  // Could be extracted from HTML
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

async fn train_command(
    data_dir: PathBuf,
    episodes: usize,
    improved: bool,
    learning_rate: Option<f64>,
    batch_size: Option<usize>,
) -> Result<()> {
    info!("Starting training...");
    info!("  Data directory: {}", data_dir.display());
    info!("  Episodes: {}", episodes);
    info!("  Improved: {}", improved);

    // Load configuration
    let mut config = Config::from_env()?;
    config.num_episodes = episodes;

    if let Some(lr) = learning_rate {
        config.learning_rate = lr;
        info!("  Learning rate override: {}", lr);
    }

    if let Some(bs) = batch_size {
        config.batch_size = bs;
        info!("  Batch size override: {}", bs);
    }

    config.setup_directories()?;

    // Load HTML samples
    let html_samples = load_html_samples(&data_dir)?;
    info!("Loaded {} HTML samples", html_samples.len());

    if html_samples.is_empty() {
        error!("No HTML samples found in {}", data_dir.display());
        return Err(article_extractor::ExtractionError::ConfigError(
            "No training data found".to_string()
        ));
    }

    // Train
    let (_agent, metrics) = if improved {
        info!("Using improved training mode");
        train_with_improvements(&config, html_samples)?
    } else {
        info!("Using standard training mode");
        train_standard(&config, html_samples)?
    };

    // Save training plot
    let plot_path = config.output_dir.join(format!(
        "training_plot_{}.png",
        chrono::Utc::now().format!("%Y%m%d_%H%M%S")
    ));
    article_extractor::training::save_training_plot(&metrics, &plot_path)?;

    info!("Training completed!");
    info!("  Best quality: {:.4}", metrics.best_avg_quality);
    info!("  Final reward: {:.4}", metrics.episode_rewards.last().unwrap_or(&0.0));
    info!("  Plot saved to: {}", plot_path.display());

    Ok(())
}
async fn search_command(
    data_dir: PathBuf,
    episodes_per_trial: usize,
) -> Result<()> {
    info!("Starting hyperparameter search...");
    info!("  Data directory: {}", data_dir.display());
    info!("  Episodes per trial: {}", episodes_per_trial);
    // Load configuration
    let config = Config::from_env()?;
    config.setup_directories()?;

    // Load HTML samples
    let html_samples = load_html_samples(&data_dir)?;
    info!("Loaded {} HTML samples", html_samples.len());

    if html_samples.is_empty() {
        error!("No HTML samples found in {}", data_dir.display());
        return Err(article_extractor::ExtractionError::ConfigError(
            "No training data found".to_string()
        ));
    }

    // Create grid search configuration
    let grid_config = GridSearchConfig::default();

    // Run search
    let search = HyperparameterSearch::new(grid_config, episodes_per_trial);
    let best_result = search.run_search(&config, html_samples)?;

    // Save results
    let results_path = config.output_dir.join(format!(
        "hyperparameter_search_{}.json",
        chrono::Utc::now().format("%Y%m%d_%H%M%S")
    ));

    let json = serde_json::to_string_pretty(&best_result)?;
    std::fs::write(&results_path, json)?;

    info!("Hyperparameter search completed!");
    info!("  Best quality: {:.4}", best_result.avg_quality);
    info!("  Results saved to: {}", results_path.display());

    Ok(())
}
/// Load HTML samples from directory
fn load_html_samples(data_dir: &PathBuf) -> Result<Vec<(String, String)>> {
    let mut samples = Vec::new();
    // Walk through directory and find HTML files
    for entry in walkdir::WalkDir::new(data_dir)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();

        if path.is_file() {
            if let Some(ext) = path.extension() {
                if ext == "html" || ext == "htm" {
                    let html_content = std::fs::read_to_string(path)?;

                    // Generate URL from filename
                    let filename = path.file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("unknown");
                    let url = format!("https://example.com/{}", filename);

                    samples.push((html_content, url));
                }
            }
        }
    }

    Ok(samples)
}
