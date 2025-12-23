//! Performance benchmarks for article extraction

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use article_extractor::*;
use std::collections::HashSet;

fn create_sample_html(size: usize) -> String {
    let paragraph = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. \
                     Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. ";

    let mut html = String::from(
        r#"<html><head><title>Test</title></head><body><article>"#
    );

    for i in 0..size {
        html.push_str(&format!("<h2>Section {}</h2>", i));
        html.push_str("<p>");
        html.push_str(&paragraph.repeat(5));
        html.push_str("</p>");
    }

    html.push_str("</article></body></html>");
    html
}

fn benchmark_baseline_extraction(c: &mut Criterion) {
    let config = Config::default();
    let extractor = BaselineExtractor::new(config.stopwords.clone());

    let mut group = c.benchmark_group("baseline_extraction");

    for size in [1, 5, 10, 20, 50].iter() {
        let html = create_sample_html(*size);

        group.throughput(Throughput::Bytes(html.len() as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}sections", size)),
            &html,
            |b, html| {
                b.iter(|| {
                    extractor.extract(black_box(html)).unwrap()
                });
            },
        );
    }

    group.finish();
}

fn benchmark_text_quality_calculation(c: &mut Criterion) {
    let config = Config::default();
    let text = "This is a well-written article with proper structure and content. \
                It contains multiple sentences that provide valuable information. \
                The text maintains good quality throughout with appropriate vocabulary. \
                This demonstrates the effectiveness of our quality scoring algorithm.";

    c.bench_function("text_quality_calculation", |b| {
        b.iter(|| {
            TextUtils::calculate_text_quality(
                black_box(text),
                black_box(&config.stopwords)
            )
        });
    });
}

fn benchmark_html_parsing(c: &mut Criterion) {
    let html = create_sample_html(20);

    c.bench_function("html_parsing", |b| {
        b.iter(|| {
            HtmlParser::parse(black_box(&html)).unwrap()
        });
    });
}

fn benchmark_reward_calculation(c: &mut Criterion) {
    let config = Config::default();
    let calculator = ImprovedRewardCalculator::new(config.stopwords);

    let text = "This is an excellent article with proper structure and content. \
                It contains multiple well-formed sentences that provide valuable information. \
                The text maintains good quality throughout with appropriate vocabulary.";

    c.bench_function("reward_calculation", |b| {
        b.iter(|| {
            calculator.calculate_reward(
                black_box(text),
                black_box(0.5)
            )
        });
    });
}

fn benchmark_site_profile_operations(c: &mut Criterion) {
    let temp_dir = tempfile::TempDir::new().unwrap();
    let mut memory = SiteProfileMemory::new(temp_dir.path()).unwrap();

    c.bench_function("site_profile_get", |b| {
        b.iter(|| {
            memory.get_profile(black_box("example.com"))
        });
    });

    let mut profile = SiteProfile::new("example.com".to_string());
    let result = site_profile::ExtractionResult {
        text: "Test content".to_string(),
        xpath: "//article[1]".to_string(),
        quality_score: 0.8,
        parameters: std::collections::HashMap::new(),
    };

    c.bench_function("site_profile_add_extraction", |b| {
        b.iter(|| {
            profile.add_extraction(black_box(result.clone()))
        });
    });
}

fn benchmark_replay_buffer_operations(c: &mut Criterion) {
    let mut buffer = PrioritizedReplayBuffer::new(10000, 0.6, 0.4);

    // Fill buffer
    for i in 0..1000 {
        let exp = replay_buffer::Experience {
            state: vec![0.0; 300],
            action: (i % 16, vec![0.0; 6]),
            reward: (i as f32) / 1000.0,
            next_state: vec![0.0; 300],
            done: false,
        };
        buffer.add(exp);
    }

    c.bench_function("replay_buffer_add", |b| {
        let exp = replay_buffer::Experience {
            state: vec![0.0; 300],
            action: (0, vec![0.0; 6]),
            reward: 0.5,
            next_state: vec![0.0; 300],
            done: false,
        };

        b.iter(|| {
            buffer.add(black_box(exp.clone()))
        });
    });

    c.bench_function("replay_buffer_sample", |b| {
        b.iter(|| {
            buffer.sample(black_box(32))
        });
    });
}

fn benchmark_curriculum_difficulty_estimation(c: &mut Criterion) {
    let curriculum = CurriculumManager::new();
    let html = create_sample_html(20);

    c.bench_function("curriculum_difficulty_estimation", |b| {
        b.iter(|| {
            curriculum.estimate_difficulty(black_box(&html))
        });
    });
}

fn benchmark_batch_extraction(c: &mut Criterion) {
    let config = Config::default();
    let extractor = BaselineExtractor::new(config.stopwords.clone());

    let mut group = c.benchmark_group("batch_extraction");

    for batch_size in [10, 50, 100].iter() {
        let htmls: Vec<String> = (0..*batch_size)
            .map(|_| create_sample_html(10))
            .collect();

        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &htmls,
            |b, htmls| {
                b.iter(|| {
                    for html in htmls {
                        extractor.extract(black_box(html)).unwrap();
                    }
                });
            },
        );
    }
    group.finish();
}

fn benchmark_state_building(c: &mut Criterion) {
    let config = Config::default();
    let baseline_extractor = BaselineExtractor::new(config.stopwords.clone());
    let mut env = ArticleExtractionEnvironment::new(baseline_extractor, config);
    let html = create_sample_html(20);

    c.bench_function("state_building", |b| {
        b.iter(|| {
            env.reset(
                black_box(&html),
                black_box("https://example.com/article".to_string()),
                None
            ).unwrap()
        });
    });
}

criterion_group!(
    benches,
    benchmark_baseline_extraction,
    benchmark_text_quality_calculation,
    benchmark_html_parsing,
    benchmark_reward_calculation,
    benchmark_site_profile_operations,
    benchmark_replay_buffer_operations,
    benchmark_curriculum_difficulty_estimation,
    benchmark_batch_extraction,
    benchmark_state_building,
    );
criterion_main!(benches);
