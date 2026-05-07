//! Training visualization and plotting using plotters library
// ============================================================================
// FILE: crates/content-extractor-rl/src/plotting.rs
// ============================================================================

use crate::{Result, training::TrainingMetrics};
use plotters::prelude::*;
use std::path::Path;
use tracing::info;

/// Plot configuration
pub struct PlotConfig {
    pub width: u32,
    pub height: u32,
    pub dpi: u32,
}

impl Default for PlotConfig {
    fn default() -> Self {
        Self {
            width: 1600,
            height: 1200,
            dpi: 150,
        }
    }
}

/// Training plots generator
pub struct TrainingPlotter {
    config: PlotConfig,
}

impl TrainingPlotter {
    /// Create new plotter with default config
    pub fn new() -> Self {
        Self {
            config: PlotConfig::default(),
        }
    }

    /// Create plotter with custom config
    pub fn with_config(config: PlotConfig) -> Self {
        Self { config }
    }

    /// Generate comprehensive training plots
    pub fn plot_training_results(&self, metrics: &TrainingMetrics, output_path: &Path) -> Result<()> {
        info!("Generating training plots to: {}", output_path.display());

        let root = BitMapBackend::new(
            output_path,
            (self.config.width, self.config.height)
        ).into_drawing_area();

        root.fill(&WHITE)
            .map_err(|e| crate::ExtractionError::ModelError(format!("Plot fill error: {}", e)))?;

        // Split into 2x2 grid
        let areas = root.split_evenly((2, 2));

        // Plot 1: Episode Rewards
        self.plot_rewards(&areas[0], &metrics.episode_rewards)?;

        // Plot 2: Episode Quality
        self.plot_quality(&areas[1], &metrics.episode_qualities)?;

        // Plot 3: Reward Distribution
        self.plot_reward_distribution(&areas[2], &metrics.episode_rewards)?;

        // Plot 4: Quality Distribution
        self.plot_quality_distribution(&areas[3], &metrics.episode_qualities)?;

        root.present()
            .map_err(|e| crate::ExtractionError::ModelError(format!("Plot present error: {}", e)))?;

        info!("Training plots saved successfully");
        Ok(())
    }

    /// Plot episode rewards over time with moving average
    fn plot_rewards<'a, DB: DrawingBackend>(
        &self,
        area: &DrawingArea<DB, plotters::coord::Shift>,
        rewards: &[f32],
    ) -> Result<()>
    where
        DB::ErrorType: 'static,
    {
        if rewards.is_empty() {
            return Ok(());
        }

        let max_episodes = rewards.len();
        let max_reward = rewards.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min_reward = rewards.iter().copied().fold(f32::INFINITY, f32::min);

        let mut chart = ChartBuilder::on(area)
            .caption("Episode Rewards", ("sans-serif", 30).into_font())
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(50)
            .build_cartesian_2d(0..max_episodes, min_reward..max_reward)
            .map_err(|e| crate::ExtractionError::ModelError(format!("Chart build error: {}", e)))?;

        chart.configure_mesh()
            .x_desc("Episode")
            .y_desc("Reward")
            .draw()
            .map_err(|e| crate::ExtractionError::ModelError(format!("Mesh error: {}", e)))?;

        // Plot raw rewards
        chart.draw_series(LineSeries::new(
            rewards.iter().enumerate().map(|(i, &r)| (i, r)),
            &BLUE.mix(0.5),
        ))
            .map_err(|e| crate::ExtractionError::ModelError(format!("Series error: {}", e)))?
            .label("Raw")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

        // Plot moving average
        if rewards.len() > 100 {
            let window = rewards.len().min(100);
            let moving_avg = self.calculate_moving_average(rewards, window);

            chart.draw_series(LineSeries::new(
                moving_avg.into_iter()
                    .enumerate()
                    .map(|(i, avg)| (i + window - 1, avg)),
                &RED,
            ))
                .map_err(|e| crate::ExtractionError::ModelError(format!("Series error: {}", e)))?
                .label(format!("MA({})", window))
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
        }

        chart.configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()
            .map_err(|e| crate::ExtractionError::ModelError(format!("Legend error: {}", e)))?;

        Ok(())
    }

    /// Plot episode quality over time with moving average
    fn plot_quality<'a, DB: DrawingBackend>(
        &self,
        area: &DrawingArea<DB, plotters::coord::Shift>,
        qualities: &[f32],
    ) -> Result<()>
    where
        DB::ErrorType: 'static,
    {
        if qualities.is_empty() {
            return Ok(());
        }

        let max_episodes = qualities.len();
        let max_quality = qualities.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let mut chart = ChartBuilder::on(area)
            .caption("Episode Quality", ("sans-serif", 30).into_font())
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(50)
            .build_cartesian_2d(0..max_episodes, 0.0..max_quality.max(1.0))
            .map_err(|e| crate::ExtractionError::ModelError(format!("Chart build error: {}", e)))?;

        chart.configure_mesh()
            .x_desc("Episode")
            .y_desc("Quality Score")
            .draw()
            .map_err(|e| crate::ExtractionError::ModelError(format!("Mesh error: {}", e)))?;

        // Plot raw quality
        chart.draw_series(LineSeries::new(
            qualities.iter().enumerate().map(|(i, &q)| (i, q)),
            &GREEN.mix(0.5),
        ))
            .map_err(|e| crate::ExtractionError::ModelError(format!("Series error: {}", e)))?
            .label("Raw")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

        // Plot moving average
        if qualities.len() > 100 {
            let window = qualities.len().min(100);
            let moving_avg = self.calculate_moving_average(qualities, window);

            chart.draw_series(LineSeries::new(
                moving_avg.into_iter()
                    .enumerate()
                    .map(|(i, avg)| (i + window - 1, avg)),
                &RED,
            ))
                .map_err(|e| crate::ExtractionError::ModelError(format!("Series error: {}", e)))?
                .label(format!("MA({})", window))
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
        }

        chart.configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()
            .map_err(|e| crate::ExtractionError::ModelError(format!("Legend error: {}", e)))?;

        Ok(())
    }

    /// Plot reward distribution histogram
    fn plot_reward_distribution<'a, DB: DrawingBackend>(
        &self,
        area: &DrawingArea<DB, plotters::coord::Shift>,
        rewards: &[f32],
    ) -> Result<()>
    where
        DB::ErrorType: 'static,
    {
        if rewards.is_empty() {
            return Ok(());
        }

        let max_reward = rewards.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min_reward = rewards.iter().copied().fold(f32::INFINITY, f32::min);

        // Calculate histogram
        let n_bins = 50;
        let bin_width = (max_reward - min_reward) / n_bins as f32;
        let mut histogram = vec![0usize; n_bins];

        for &reward in rewards {
            let bin = ((reward - min_reward) / bin_width).floor() as usize;
            let bin = bin.min(n_bins - 1);
            histogram[bin] += 1;
        }

        let max_count = *histogram.iter().max().unwrap_or(&1);

        let mut chart = ChartBuilder::on(area)
            .caption("Reward Distribution", ("sans-serif", 30).into_font())
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(50)
            .build_cartesian_2d(min_reward..max_reward, 0..max_count)
            .map_err(|e| crate::ExtractionError::ModelError(format!("Chart build error: {}", e)))?;

        chart.configure_mesh()
            .x_desc("Reward")
            .y_desc("Frequency")
            .draw()
            .map_err(|e| crate::ExtractionError::ModelError(format!("Mesh error: {}", e)))?;

        // Draw histogram bars
        chart.draw_series(
            histogram.iter().enumerate().map(|(i, &count)| {
                let x0 = min_reward + i as f32 * bin_width;
                let x1 = x0 + bin_width;
                Rectangle::new([(x0, 0), (x1, count)], BLUE.mix(0.7).filled())
            })
        )
            .map_err(|e| crate::ExtractionError::ModelError(format!("Series error: {}", e)))?;

        // Draw mean line
        let mean = rewards.iter().sum::<f32>() / rewards.len() as f32;
        chart.draw_series(LineSeries::new(
            vec![(mean, 0), (mean, max_count)],
            RED.stroke_width(2),
        ))
            .map_err(|e| crate::ExtractionError::ModelError(format!("Series error: {}", e)))?
            .label(format!("Mean: {:.3}", mean))
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

        chart.configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()
            .map_err(|e| crate::ExtractionError::ModelError(format!("Legend error: {}", e)))?;

        Ok(())
    }

    /// Plot quality distribution histogram
    fn plot_quality_distribution<'a, DB: DrawingBackend>(
        &self,
        area: &DrawingArea<DB, plotters::coord::Shift>,
        qualities: &[f32],
    ) -> Result<()>
    where
        DB::ErrorType: 'static,
    {
        if qualities.is_empty() {
            return Ok(());
        }

        let max_quality = qualities.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min_quality = qualities.iter().copied().fold(f32::INFINITY, f32::min);

        // Calculate histogram
        let n_bins = 50;
        let bin_width = (max_quality - min_quality).max(0.01) / n_bins as f32;
        let mut histogram = vec![0usize; n_bins];

        for &quality in qualities {
            let bin = ((quality - min_quality) / bin_width).floor() as usize;
            let bin = bin.min(n_bins - 1);
            histogram[bin] += 1;
        }

        let max_count = *histogram.iter().max().unwrap_or(&1);

        let mut chart = ChartBuilder::on(area)
            .caption("Quality Distribution", ("sans-serif", 30).into_font())
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(50)
            .build_cartesian_2d(min_quality..max_quality.max(1.0), 0..max_count)
            .map_err(|e| crate::ExtractionError::ModelError(format!("Chart build error: {}", e)))?;

        chart.configure_mesh()
            .x_desc("Quality Score")
            .y_desc("Frequency")
            .draw()
            .map_err(|e| crate::ExtractionError::ModelError(format!("Mesh error: {}", e)))?;

        // Draw histogram bars
        chart.draw_series(
            histogram.iter().enumerate().map(|(i, &count)| {
                let x0 = min_quality + i as f32 * bin_width;
                let x1 = x0 + bin_width;
                Rectangle::new([(x0, 0), (x1, count)], GREEN.mix(0.7).filled())
            })
        )
            .map_err(|e| crate::ExtractionError::ModelError(format!("Series error: {}", e)))?;

        // Draw mean line
        let mean = qualities.iter().sum::<f32>() / qualities.len() as f32;
        chart.draw_series(LineSeries::new(
            vec![(mean, 0), (mean, max_count)],
            RED.stroke_width(2),
        ))
            .map_err(|e| crate::ExtractionError::ModelError(format!("Series error: {}", e)))?
            .label(format!("Mean: {:.3}", mean))
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

        chart.configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()
            .map_err(|e| crate::ExtractionError::ModelError(format!("Legend error: {}", e)))?;

        Ok(())
    }

    /// Calculate moving average
    fn calculate_moving_average(&self, data: &[f32], window: usize) -> Vec<f32> {
        let mut result = Vec::with_capacity(data.len() - window + 1);

        for i in window - 1..data.len() {
            let sum: f32 = data[i - window + 1..=i].iter().sum();
            result.push(sum / window as f32);
        }

        result
    }

    /// Generate plot periodically during training
    pub fn plot_intermediate(&self, metrics: &TrainingMetrics, output_path: &Path, episode: usize) -> Result<()> {
        let timestamped_path = output_path.parent().unwrap().join(
            format!("training_plot_ep{}.png", episode)
        );

        self.plot_training_results(metrics, &timestamped_path)
    }
}

impl Default for TrainingPlotter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_plot_generation() {
        let temp_dir = TempDir::new().unwrap();
        let plot_path = temp_dir.path().join("test_plot.png");

        let metrics = TrainingMetrics {
            episode_rewards: (0..100).map(|i| (i as f32 * 0.01) - 0.5).collect(),
            episode_qualities: (0..100).map(|i| i as f32 * 0.01).collect(),
            episode_losses: vec![],
            best_avg_quality: 0.9,
        };

        let plotter = TrainingPlotter::new();
        plotter.plot_training_results(&metrics, &plot_path).unwrap();

        assert!(plot_path.exists());
    }
}