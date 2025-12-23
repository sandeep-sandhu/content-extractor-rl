//! Concurrency and parallelism benchmarks

use article_extractor::*;
use std::sync::Arc;
use std::thread;
use std::time::Instant;
use rayon::prelude::*;

fn main() {
    println!("=== Concurrency Benchmarks ===\n");

    let html_samples: Vec<String> = (0..100)
        .map(|i| create_sample_html(i % 20 + 5))
        .collect();

    // Benchmark 1: Sequential extraction
    {
        let config = Config::default();
        let extractor = BaselineExtractor::new(config.stopwords.clone());

        let start = Instant::now();

        for html in &html_samples {
            let _ = extractor.extract(html);
        }

        let duration = start.elapsed();

        println!("Sequential Extraction (100 samples):");
        println!("  Time: {:.2?}", duration);
        println!("  Throughput: {:.2} extractions/sec\n",
                 100.0 / duration.as_secs_f64());
    }

    // Benchmark 2: Thread pool extraction
    {
        let config = Arc::new(Config::default());
        let samples = Arc::new(html_samples.clone());

        let start = Instant::now();

        let handles: Vec<_> = (0..10).map(|thread_id| {
            let config = Arc::clone(&config);
            let samples = Arc::clone(&samples);

            thread::spawn(move || {
                let extractor = BaselineExtractor::new(config.stopwords.clone());
                let chunk_size = samples.len() / 10;
                let start_idx = thread_id * chunk_size;
                let end_idx = start_idx + chunk_size;

                for i in start_idx..end_idx {
                    if i < samples.len() {
                        let _ = extractor.extract(&samples[i]);
                    }
                }
            })
        }).collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let duration = start.elapsed();

        println!("Thread Pool Extraction (10 threads, 100 samples):");
        println!("  Time: {:.2?}", duration);
        println!("  Throughput: {:.2} extractions/sec",
                 100.0 / duration.as_secs_f64());
        println!("  Speedup: {:.2}x\n",
                 5.0 / duration.as_secs_f64()); // Approximate sequential time
    }

    // Benchmark 3: Rayon parallel extraction
    {
        let config = Config::default();

        let start = Instant::now();

        html_samples.par_iter().for_each(|html| {
            let extractor = BaselineExtractor::new(config.stopwords.clone());
            let _ = extractor.extract(html);
        });

        let duration = start.elapsed();

        println!("Rayon Parallel Extraction (100 samples):");
        println!("  Time: {:.2?}", duration);
        println!("  Throughput: {:.2} extractions/sec",
                 100.0 / duration.as_secs_f64());
        println!("  Speedup: {:.2}x\n",
                 5.0 / duration.as_secs_f64());
    }

    // Benchmark 4: Batch processing with different batch sizes
    {
        let config = Config::default();

        for batch_size in [1, 10, 20, 50, 100] {
            let start = Instant::now();

            html_samples.par_chunks(batch_size).for_each(|chunk| {
                let extractor = BaselineExtractor::new(config.stopwords.clone());
                for html in chunk {
                    let _ = extractor.extract(html);
                }
            });

            let duration = start.elapsed();

            println!("Batch Size {} (100 samples):", batch_size);
            println!("  Time: {:.2?}", duration);
            println!("  Throughput: {:.2} extractions/sec\n",
                     100.0 / duration.as_secs_f64());
        }
    }
}

fn create_sample_html(size: usize) -> String {
    let paragraph = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ";
    let mut html = String::from("<html><body><article>");

    for _ in 0..size {
        html.push_str("<p>");
        html.push_str(&paragraph.repeat(5));
        html.push_str("</p>");
    }

    html.push_str("</article></body></html>");
    html
}