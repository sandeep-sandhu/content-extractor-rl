//! Memory usage benchmarks

use article_extractor::*;
use std::sync::Arc;
use std::thread;

fn main() {
    println!("=== Memory Usage Benchmarks ===\n");

    // Benchmark 1: Baseline extractor memory usage
    {
        let config = Config::default();
        let extractor = BaselineExtractor::new(config.stopwords.clone());

        let html = include_str!("../tests/fixtures/large_article.html");

        let initial_memory = get_memory_usage();

        for _ in 0..1000 {
            let _ = extractor.extract(html);
        }

        let final_memory = get_memory_usage();

        println!("Baseline Extractor (1000 iterations):");
        println!("  Initial memory: {:.2} MB", initial_memory);
        println!("  Final memory: {:.2} MB", final_memory);
        println!("  Delta: {:.2} MB\n", final_memory - initial_memory);
    }

    // Benchmark 2: Site profile memory growth
    {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let mut memory = SiteProfileMemory::new(temp_dir.path()).unwrap();

        let initial_memory = get_memory_usage();

        // Add 1000 extractions to 100 different domains
        for domain_id in 0..100 {
            let domain = format!("example{}.com", domain_id);
            let profile = memory.get_profile(&domain);

            for _ in 0..10 {
                let result = site_profile::ExtractionResult {
                    text: "Test content".repeat(100),
                    xpath: "//article[1]".to_string(),
                    quality_score: 0.8,
                    parameters: std::collections::HashMap::new(),
                };
                profile.add_extraction(result);
            }
        }

        let final_memory = get_memory_usage();

        println!("Site Profile Memory (100 domains, 10 extractions each):");
        println!("  Initial memory: {:.2} MB", initial_memory);
        println!("  Final memory: {:.2} MB", final_memory);
        println!("  Delta: {:.2} MB", final_memory - initial_memory);
        println!("  Memory per domain: {:.2} KB\n",
                 (final_memory - initial_memory) * 1024.0 / 100.0);
    }

    // Benchmark 3: Replay buffer memory usage
    {
        let initial_memory = get_memory_usage();

        let mut buffer = PrioritizedReplayBuffer::new(100_000, 0.6, 0.4);

        for i in 0..100_000 {
            let exp = replay_buffer::Experience {
                state: vec![0.5; 300],
                action: (i % 16, vec![0.1; 6]),
                reward: 0.5,
                next_state: vec![0.5; 300],
                done: false,
            };
            buffer.add(exp);
        }

        let final_memory = get_memory_usage();

        println!("Replay Buffer (100,000 experiences):");
        println!("  Initial memory: {:.2} MB", initial_memory);
        println!("  Final memory: {:.2} MB", final_memory);
        println!("  Delta: {:.2} MB", final_memory - initial_memory);
        println!("  Memory per experience: {:.2} KB\n",
                 (final_memory - initial_memory) * 1024.0 / 100_000.0);
    }

    // Benchmark 4: Concurrent extraction
    {
        let config = Arc::new(Config::default());
        let html = Arc::new(create_sample_html(10));

        let initial_memory = get_memory_usage();

        let handles: Vec<_> = (0..10).map(|_| {
            let config = Arc::clone(&config);
            let html = Arc::clone(&html);

            thread::spawn(move || {
                let extractor = BaselineExtractor::new(config.stopwords.clone());

                for _ in 0..100 {
                    let _ = extractor.extract(&html);
                }
            })
        }).collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let final_memory = get_memory_usage();

        println!("Concurrent Extraction (10 threads, 100 extractions each):");
        println!("  Initial memory: {:.2} MB", initial_memory);
        println!("  Final memory: {:.2} MB", final_memory);
        println!("  Delta: {:.2} MB\n", final_memory - initial_memory);
    }
}

fn get_memory_usage() -> f64 {
    // Simple memory estimation (in MB)
    // In production, use proper memory profiling tools
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let Ok(kb) = parts[1].parse::<f64>() {
                            return kb / 1024.0; // Convert to MB
                        }
                    }
                }
            }
        }
    }

    // Fallback for other platforms
    0.0
}

fn create_sample_html(size: usize) -> String {
    let paragraph = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ";
    let mut html = String::from("<html><body><article>");

    for _ in 0..size {
        html.push_str("<p>");
        html.push_str(&paragraph.repeat(10));
        html.push_str("</p>");
    }

    html.push_str("</article></body></html>");
    html
}
