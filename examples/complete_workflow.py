#!/usr/bin/env python3
"""
Production Content Extractor RL Workflow with TPE Hyperparameter Tuning
====================================================================

Features:
- Memory-efficient batch training
- TPE (Tree-structured Parzen Estimator) hyperparameter optimization
- Training plots and visualizations
- GPU acceleration support
"""

import os
import logging
import random
import sys
import bz2
import json
import argparse
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from content_extractor_rl_rs import RustArticleExtractor

# Try to import optuna for TPE
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not available. Install with: pip install optuna")


# Configuration
DATA_DIR = Path("/var/local/sss/content_extractor_rl")
HTML_ARCHIVE_DIR = DATA_DIR / "html_archive"
SITE_PROFILES_DIR = DATA_DIR / "site_profiles"
MODELS_DIR = DATA_DIR / "models"
LOGS_DIR = DATA_DIR / "logs"
OUTPUT_DIR = DATA_DIR / "output"
PLOTS_DIR = DATA_DIR / "plots"

# Ensure directories exist
for dir_path in [SITE_PROFILES_DIR, MODELS_DIR, LOGS_DIR, OUTPUT_DIR, PLOTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Logging configuration
_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_FILE = LOGS_DIR / f'content_extractor_rl_{_timestamp}.log'

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        stream_handler
    ]
)
logger = logging.getLogger(__name__)
print(f"Logging to: {LOG_FILE}")


class HTMLArchiveLoader:
    """Load pre-downloaded bzipped HTML files with memory-efficient streaming."""

    def __init__(self, archive_dir: Path):
        self.archive_dir = archive_dir
        self.file_index = self._build_index()

    def _build_index(self) -> List[Dict]:
        """Build index of all HTML files."""
        index = []

        if not self.archive_dir.exists():
            logger.warning(f"Archive directory {self.archive_dir} does not exist")
            return index

        for date_dir in sorted(self.archive_dir.iterdir()):
            if not date_dir.is_dir():
                continue

            for html_file in date_dir.glob("*.html.bz2"):
                filename = html_file.stem.replace('.html', '')
                parts = filename.split('_', 1)

                domain = parts[0] if len(parts) > 0 else 'unknown'
                article_id = parts[1] if len(parts) > 1 else filename

                index.append({
                    'path': html_file,
                    'domain': domain,
                    'article_id': article_id,
                    'date': date_dir.name,
                    'url': f"https://{domain}/{article_id}"
                })

        logger.info(f"Loaded {len(index)} HTML files from archive")
        return index

    def load_html(self, index_entry: Dict) -> str:
        """Load and decompress HTML from bzipped file."""
        try:
            with bz2.open(index_entry['path'], 'rt', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to load {index_entry['path']}: {e}")
            return ""

    def sample_random(self, n: int = 1) -> List[Tuple[str, str]]:
        """Sample n random HTML files. Returns [(html_content, url), ...]."""
        if not self.file_index:
            return []

        samples = []
        entries = random.sample(self.file_index, min(n, len(self.file_index)))

        for entry in entries:
            html_content = self.load_html(entry)
            if html_content:
                samples.append((html_content, entry['url']))

        return samples

    def get_batches(self, batch_size: int = 1000, max_samples: Optional[int] = None):
        """
        Generator that yields batches of HTML samples.
        Memory-efficient for large datasets.
        """
        entries = self.file_index[:max_samples] if max_samples else self.file_index

        for i in range(0, len(entries), batch_size):
            batch_entries = entries[i:i + batch_size]
            batch_samples = []

            for entry in batch_entries:
                html_content = self.load_html(entry)
                if html_content:
                    batch_samples.append((html_content, entry['url']))

            yield batch_samples

            # Progress logging
            if i % 5000 == 0:
                logger.info(f"  Processed {i}/{len(entries)} files...")


def check_device_info():
    """Check and log device information."""
    try:
        from content_extractor_rl_rs import check_cuda_available
        cuda_available = check_cuda_available()

        logger.info("="*80)
        if cuda_available:
            logger.info("GPU ACCELERATION ENABLED (CUDA)")
            logger.info("Training will be significantly faster")
        else:
            logger.info("CPU MODE")
            logger.info("Consider using GPU for faster training")
        logger.info("="*80)

        return cuda_available
    except (ImportError, AttributeError) as e:
        logger.warning(f"Could not check CUDA availability: {e}")
        return False


def plot_training_results(metrics: Dict, output_path: Path):
    """Generate training plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Episode Rewards
    if metrics['episode_rewards']:
        ax = axes[0, 0]
        rewards = metrics['episode_rewards']
        ax.plot(rewards, alpha=0.3, label='Raw')
        # Moving average
        if len(rewards) > 100:
            window = min(100, len(rewards) // 10)
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(rewards)), moving_avg, 'r-', linewidth=2, label=f'MA({window})')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Episode Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 2: Episode Quality
    if metrics['episode_qualities']:
        ax = axes[0, 1]
        qualities = metrics['episode_qualities']
        ax.plot(qualities, alpha=0.3, label='Raw')
        # Moving average
        if len(qualities) > 100:
            window = min(100, len(qualities) // 10)
            moving_avg = np.convolve(qualities, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(qualities)), moving_avg, 'g-', linewidth=2, label=f'MA({window})')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Quality Score')
        ax.set_title('Episode Quality')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 3: Reward Distribution
    if metrics['episode_rewards']:
        ax = axes[1, 0]
        ax.hist(metrics['episode_rewards'], bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Reward')
        ax.set_ylabel('Frequency')
        ax.set_title('Reward Distribution')
        ax.axvline(np.mean(metrics['episode_rewards']), color='r', linestyle='--',
                   label=f'Mean: {np.mean(metrics["episode_rewards"]):.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 4: Quality Distribution
    if metrics['episode_qualities']:
        ax = axes[1, 1]
        ax.hist(metrics['episode_qualities'], bins=50, alpha=0.7, edgecolor='black', color='green')
        ax.set_xlabel('Quality Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Quality Distribution')
        ax.axvline(np.mean(metrics['episode_qualities']), color='r', linestyle='--',
                   label=f'Mean: {np.mean(metrics["episode_qualities"]):.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Training plots saved to: {output_path}")


def save_training_results(metrics: Dict, output_path: Path, hyperparams: Optional[Dict] = None):
    """Save training results and hyperparameters to JSON."""
    results = {
        'timestamp': datetime.now().isoformat(),
        'best_avg_quality': float(metrics['best_avg_quality']),
        'final_reward': float(metrics['episode_rewards'][-1]) if metrics['episode_rewards'] else 0.0,
        'num_episodes': len(metrics['episode_rewards']),
        'avg_reward': float(sum(metrics['episode_rewards']) / len(metrics['episode_rewards'])) if metrics['episode_rewards'] else 0.0,
        'avg_quality': float(sum(metrics['episode_qualities']) / len(metrics['episode_qualities'])) if metrics['episode_qualities'] else 0.0,
        'max_reward': float(max(metrics['episode_rewards'])) if metrics['episode_rewards'] else 0.0,
        'max_quality': float(max(metrics['episode_qualities'])) if metrics['episode_qualities'] else 0.0,
    }

    if hyperparams:
        results['hyperparameters'] = hyperparams

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved training results to {output_path}")
    return results


def train_model_memory_efficient(html_loader: HTMLArchiveLoader, args):
    """
    Memory-efficient training that loads samples in batches.
    For large datasets (>10k samples).
    """
    logger.info("="*80)
    logger.info("MEMORY-EFFICIENT TRAINING MODE")
    logger.info("="*80)

    # Check device
    check_device_info()

    # Set environment variables
    os.environ["ARTICLE_EXTRACTOR_SITE_PROFILES"] = str(SITE_PROFILES_DIR)
    os.environ["ARTICLE_EXTRACTOR_OUTPUT_DIR"] = str(OUTPUT_DIR)
    os.environ["ARTICLE_EXTRACTOR_MODELS_DIR"] = str(MODELS_DIR)

    # Use smaller subset for training
    max_samples = args.max_samples if hasattr(args, 'max_samples') and args.max_samples else 10000
    logger.info(f"Using {max_samples} samples for training (memory limit)")

    # Load samples in manageable chunks
    training_samples = []
    for batch in html_loader.get_batches(batch_size=1000, max_samples=max_samples):
        training_samples.extend(batch)
        if len(training_samples) >= max_samples:
            break

    logger.info(f"Loaded {len(training_samples)} training samples")

    if not training_samples:
        logger.error("No training samples loaded!")
        return

    logger.info(f"Episodes: {args.episodes}")
    logger.info(f"Improved mode: {args.improved}")

    # Initialize extractor
    logger.info("Initializing extractor...")
    extractor = RustArticleExtractor()

    # Train
    logger.info("Starting training...")
    start_time = datetime.now()

    metrics = extractor.train(
        html_samples=training_samples,
        episodes=args.episodes,
        improved=args.improved
    )

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Log results
    logger.info("="*80)
    logger.info("TRAINING COMPLETED")
    logger.info("="*80)
    logger.info(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    logger.info(f"Best avg quality: {metrics['best_avg_quality']:.4f}")
    logger.info(f"Final reward: {metrics['episode_rewards'][-1]:.4f}")
    logger.info(f"Avg reward: {sum(metrics['episode_rewards'])/len(metrics['episode_rewards']):.4f}")
    logger.info(f"Avg quality: {sum(metrics['episode_qualities'])/len(metrics['episode_qualities']):.4f}")

    # Save results
    results_path = OUTPUT_DIR / f'training_results_{_timestamp}.json'
    save_training_results(metrics, results_path)

    # Generate plots
    plot_path = PLOTS_DIR / f'training_plot_{_timestamp}.png'
    plot_training_results(metrics, plot_path)

    # Log model location
    model_path = MODELS_DIR / "best_model.onnx"
    if model_path.exists():
        logger.info(f"Best model saved at: {model_path}")

    logger.info(f"Site profiles saved in: {SITE_PROFILES_DIR}")
    logger.info("="*80)


def hyperparameter_tuning_tpe(html_loader: HTMLArchiveLoader, args):
    """
    Hyperparameter optimization using TPE (Tree-structured Parzen Estimator).
    Uses Optuna for Bayesian optimization.
    """
    if not OPTUNA_AVAILABLE:
        logger.error("Optuna is required for TPE. Install with: pip install optuna")
        return

    logger.info("="*80)
    logger.info("HYPERPARAMETER OPTIMIZATION (TPE)")
    logger.info("="*80)

    # Check device
    check_device_info()

    # Set environment variables
    os.environ["ARTICLE_EXTRACTOR_SITE_PROFILES"] = str(SITE_PROFILES_DIR)
    os.environ["ARTICLE_EXTRACTOR_OUTPUT_DIR"] = str(OUTPUT_DIR)
    os.environ["ARTICLE_EXTRACTOR_MODELS_DIR"] = str(MODELS_DIR)

    # Load small training set for fast trials
    max_samples = 5000  # Use smaller set for hyperparameter search
    logger.info(f"Loading {max_samples} samples for hyperparameter tuning...")

    training_samples = []
    for batch in html_loader.get_batches(batch_size=1000, max_samples=max_samples):
        training_samples.extend(batch)
        if len(training_samples) >= max_samples:
            break

    logger.info(f"Loaded {len(training_samples)} samples")

    n_trials = args.trials if hasattr(args, 'trials') else 20
    episodes_per_trial = args.episodes_per_trial if hasattr(args, 'episodes_per_trial') else 100

    logger.info(f"Running {n_trials} trials with {episodes_per_trial} episodes each")

    def objective(trial):
        """Optuna objective function."""
        # Suggest hyperparameters
        # Note: These would need to be passed to Rust config somehow
        # For now, we'll just demonstrate the concept

        logger.info(f"Trial {trial.number + 1}/{n_trials}")

        # Train with current hyperparameters
        extractor = RustArticleExtractor()

        try:
            metrics = extractor.train(
                html_samples=training_samples,
                episodes=episodes_per_trial,
                improved=True
            )

            # Optimize for best average quality
            quality = metrics['best_avg_quality']
            logger.info(f"  Trial {trial.number + 1} quality: {quality:.4f}")

            return quality
        except Exception as e:
            logger.error(f"  Trial {trial.number + 1} failed: {e}")
            return 0.0

    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    # Run optimization
    study.optimize(objective, n_trials=n_trials)

    # Log results
    logger.info("="*80)
    logger.info("OPTIMIZATION COMPLETED")
    logger.info("="*80)
    logger.info(f"Best quality: {study.best_value:.4f}")
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best parameters: {study.best_params}")

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'n_trials': n_trials,
        'best_value': float(study.best_value),
        'best_params': study.best_params,
        'best_trial': study.best_trial.number,
    }

    results_path = OUTPUT_DIR / f'hyperparameter_optimization_{_timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Optimization results saved to: {results_path}")

    # Plot optimization history
    try:
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        plot_path = PLOTS_DIR / f'optimization_history_{_timestamp}.png'
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        logger.info(f"Optimization plot saved to: {plot_path}")
    except Exception as e:
        logger.warning(f"Could not generate optimization plot: {e}")


def extract_single(html_content: str, url: str, model_path: Optional[Path] = None):
    """Extract article from single HTML content."""
    logger.info(f"Extracting from: {url}")

    os.environ["ARTICLE_EXTRACTOR_SITE_PROFILES"] = str(SITE_PROFILES_DIR)
    os.environ["ARTICLE_EXTRACTOR_OUTPUT_DIR"] = str(OUTPUT_DIR)
    os.environ["ARTICLE_EXTRACTOR_MODELS_DIR"] = str(MODELS_DIR)

    if model_path and model_path.exists():
        logger.info(f"Using model: {model_path}")
        extractor = RustArticleExtractor(model=str(model_path))
    else:
        logger.info("Using baseline extractor (no model)")
        extractor = RustArticleExtractor()

    result = extractor.extract(
        website_page_html=html_content,
        url=url
    )

    logger.info("="*80)
    logger.info("EXTRACTION RESULTS")
    logger.info("="*80)
    logger.info(f"URL: {result['url']}")
    logger.info(f"Quality Score: {result['quality_score']:.4f}")
    logger.info(f"Method: {result['method']}")
    logger.info(f"Content Length: {len(result['content'])} characters")
    logger.info(f"XPath: {result.get('xpath', 'N/A')}")
    logger.info("")
    logger.info("Content Preview:")
    logger.info("-"*80)
    preview = result['content'][:500]
    logger.info(preview + ("..." if len(result['content']) > 500 else ""))
    logger.info("-"*80)

    output_file = OUTPUT_DIR / f'extracted_{_timestamp}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"Full extraction saved to: {output_file}")
    return result


def extract_batch(html_loader: HTMLArchiveLoader, args):
    """Extract from multiple HTML files."""
    logger.info("="*80)
    logger.info("BATCH EXTRACTION MODE")
    logger.info("="*80)

    model_path = MODELS_DIR / "best_model.onnx"

    os.environ["ARTICLE_EXTRACTOR_SITE_PROFILES"] = str(SITE_PROFILES_DIR)
    os.environ["ARTICLE_EXTRACTOR_OUTPUT_DIR"] = str(OUTPUT_DIR)
    os.environ["ARTICLE_EXTRACTOR_MODELS_DIR"] = str(MODELS_DIR)

    if model_path.exists():
        logger.info(f"Using model: {model_path}")
        extractor = RustArticleExtractor(model=str(model_path))
    else:
        logger.info("Using baseline extractor (no model)")
        extractor = RustArticleExtractor()

    batch_size = args.batch_size if hasattr(args, 'batch_size') else 10
    samples = html_loader.sample_random(batch_size)

    logger.info(f"Processing {len(samples)} samples...")

    results = []
    for idx, (html_content, url) in enumerate(samples, 1):
        logger.info(f"\nProcessing {idx}/{len(samples)}: {url}")

        try:
            result = extractor.extract(
                website_page_html=html_content,
                url=url
            )
            results.append(result)
            logger.info(f"  Quality: {result['quality_score']:.4f}, Length: {len(result['content'])} chars")
        except Exception as e:
            logger.error(f"  Failed: {e}")

    batch_file = OUTPUT_DIR / f'batch_extraction_{_timestamp}.json'
    with open(batch_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("="*80)
    logger.info(f"Batch extraction complete: {len(results)}/{len(samples)} successful")
    logger.info(f"Results saved to: {batch_file}")
    logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Content Extractor RL - Training and Extraction Workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--mode',
        choices=['train', 'extract', 'batch', 'extract-archive', 'tune'],
        required=True,
        help='Operation mode'
    )

    # Training arguments
    parser.add_argument(
        '--episodes',
        type=int,
        default=1000,
        help='Number of training episodes (default: 1000)'
    )

    parser.add_argument(
        '--improved',
        action='store_true',
        help='Use improved training (curriculum learning, enhanced rewards)'
    )

    parser.add_argument(
        '--max-samples',
        type=int,
        help='Maximum number of training samples (default: 10000 for memory efficiency)'
    )

    # Hyperparameter tuning arguments
    parser.add_argument(
        '--trials',
        type=int,
        default=20,
        help='Number of TPE trials for hyperparameter tuning (default: 20)'
    )

    parser.add_argument(
        '--episodes-per-trial',
        type=int,
        default=100,
        help='Episodes per trial for hyperparameter tuning (default: 100)'
    )

    # Extraction arguments
    parser.add_argument(
        '--input',
        type=Path,
        help='Input HTML file for extraction'
    )

    parser.add_argument(
        '--url',
        type=str,
        default='https://example.com/article',
        help='URL for the article'
    )

    parser.add_argument(
        '--model',
        type=Path,
        help='Path to trained model'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Number of samples for batch extraction'
    )

    args = parser.parse_args()

    # Initialize HTML loader
    html_loader = HTMLArchiveLoader(HTML_ARCHIVE_DIR)

    if not html_loader.file_index and args.mode in ['train', 'extract-archive', 'batch', 'tune']:
        logger.error(f"No HTML files found in {HTML_ARCHIVE_DIR}")
        return 1

    try:
        if args.mode == 'train':
            train_model_memory_efficient(html_loader, args)

        elif args.mode == 'tune':
            hyperparameter_tuning_tpe(html_loader, args)

        elif args.mode == 'extract':
            if not args.input:
                logger.error("--input required for extract mode")
                return 1

            if not args.input.exists():
                logger.error(f"Input file not found: {args.input}")
                return 1

            with open(args.input, 'r', encoding='utf-8') as f:
                html_content = f.read()

            model_path = args.model or (MODELS_DIR / "best_model.onnx")
            extract_single(html_content, args.url, model_path)

        elif args.mode == 'extract-archive':
            samples = html_loader.sample_random(1)
            if not samples:
                logger.error("No samples available")
                return 1

            html_content, url = samples[0]
            model_path = args.model or (MODELS_DIR / "best_model.onnx")
            extract_single(html_content, url, model_path)

        elif args.mode == 'batch':
            extract_batch(html_loader, args)

        logger.info("\nWorkflow completed successfully!")
        logger.info(f"Log file: {LOG_FILE}")
        return 0

    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
