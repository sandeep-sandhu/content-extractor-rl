"""
Performance tests for Python bindings
"""

import pytest
import time
import statistics
from content_extractor_rl import RustArticleExtractor

# Skip if package not available
try:
    from content_extractor_rl import RustArticleExtractor
    PACKAGE_AVAILABLE = True
except ImportError:
    PACKAGE_AVAILABLE = False
    pytestmark = pytest.mark.skip("content_extractor_rl package not installed")


def create_sample_html(paragraphs=10):
    """Generate sample HTML with specified number of paragraphs"""
    html = """
    <html>
    <head><title>Performance Test Article</title></head>
    <body>
        <article>
            <h1>Performance Test Article</h1>
    """

    for i in range(paragraphs):
        html += f"""
            <p>This is paragraph {i}. Lorem ipsum dolor sit amet, consectetur 
            adipiscing elit. Sed do eiusmod tempor incididunt ut labore et 
            dolore magna aliqua. Ut enim ad minim veniam, quis nostrud 
            exercitation ullamco laboris.</p>
        """

    html += """
        </article>
    </body>
    </html>
    """
    return html


class TestExtractionPerformance:
    """Performance tests for extraction"""

    def test_single_extraction_latency(self):
        """Measure single extraction latency"""
        extractor = RustArticleExtractor()
        html = create_sample_html(10)
        url = "https://example.com/article"

        # Warmup
        for _ in range(10):
            extractor.extract(html, url)

        # Measure
        times = []
        for _ in range(100):
            start = time.perf_counter()
            extractor.extract(html, url)
            times.append(time.perf_counter() - start)

        avg_time = statistics.mean(times)
        median_time = statistics.median(times)
        std_dev = statistics.stdev(times)

        print(f"\nSingle Extraction Latency:")
        print(f"  Average: {avg_time*1000:.2f} ms")
        print(f"  Median: {median_time*1000:.2f} ms")
        print(f"  Std Dev: {std_dev*1000:.2f} ms")
        print(f"  Min: {min(times)*1000:.2f} ms")
        print(f"  Max: {max(times)*1000:.2f} ms")

        # Performance assertion
        assert avg_time < 0.1, "Average extraction should be < 100ms"

    def test_batch_extraction_throughput(self):
        """Measure batch extraction throughput"""
        extractor = RustArticleExtractor()

        # Create 100 sample articles
        samples = [
            (create_sample_html(10), f"https://example.com/article{i}")
            for i in range(100)
        ]

        start = time.perf_counter()
        result = extractor.extract_batch(samples)
        duration = time.perf_counter() - start

        throughput = len(samples) / duration

        print(f"\nBatch Extraction Throughput:")
        print(f"  Total time: {duration:.2f} seconds")
        print(f"  Samples: {len(samples)}")
        print(f"  Throughput: {throughput:.2f} extractions/sec")

        assert len(result['articles']) == 100
        assert throughput > 50, "Should process at least 50 articles/sec"

    def test_extraction_scaling(self):
        """Test extraction performance with different HTML sizes"""
        extractor = RustArticleExtractor()
        url = "https://example.com/article"

        results = {}

        for size in [5, 10, 20, 50, 100]:
            html = create_sample_html(size)

            times = []
            for _ in range(20):
                start = time.perf_counter()
                extractor.extract(html, url)
                times.append(time.perf_counter() - start)

            avg_time = statistics.mean(times)
            results[size] = avg_time

        print(f"\nExtraction Scaling (paragraphs -> avg time):")
        for size, time_taken in results.items():
            print(f"  {size:3d} paragraphs: {time_taken*1000:6.2f} ms")

        # Check that scaling is reasonable (sub-linear preferred)
        # Time for 100 paragraphs should be < 10x time for 10 paragraphs
        assert results[100] < results[10] * 10, "Scaling should be sub-linear"

    def test_concurrent_extraction_performance(self):
        """Test performance with concurrent extractions"""
        import concurrent.futures

        extractor = RustArticleExtractor()
        html = create_sample_html(10)

        def extract_one(i):
            return extractor.extract(html, f"https://example.com/article{i}")

        # Sequential baseline
        start = time.perf_counter()
        for i in range(100):
            extract_one(i)
        sequential_time = time.perf_counter() - start

        # Concurrent
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            list(executor.map(extract_one, range(100)))
        concurrent_time = time.perf_counter() - start

        speedup = sequential_time / concurrent_time

        print(f"\nConcurrent Extraction Performance:")
        print(f"  Sequential time: {sequential_time:.2f} seconds")
        print(f"  Concurrent time: {concurrent_time:.2f} seconds")
        print(f"  Speedup: {speedup:.2f}x")

        # Should see some speedup from concurrency
        assert speedup > 1.5, "Should see at least 1.5x speedup with concurrency"


class TestMemoryPerformance:
    """Memory usage tests"""

    def test_memory_stability(self):
        """Test that memory usage is stable over many extractions"""
        import tracemalloc

        extractor = RustArticleExtractor()
        html = create_sample_html(10)
        url = "https://example.com/article"

        tracemalloc.start()

        # Warmup
        for _ in range(100):
            extractor.extract(html, url)

        snapshot1 = tracemalloc.take_snapshot()

        # Process 1000 more
        for _ in range(1000):
            extractor.extract(html, url)

        snapshot2 = tracemalloc.take_snapshot()

        top_stats = snapshot2.compare_to(snapshot1, 'lineno')

        total_diff = sum(stat.size_diff for stat in top_stats)

        print(f"\nMemory Stability Test:")
        print(f"  Memory change after 1000 extractions: {total_diff / 1024:.2f} KB")

        tracemalloc.stop()

        # Memory growth should be minimal (< 1MB)
        assert abs(total_diff) < 1024 * 1024, "Memory should remain stable"

    def test_batch_memory_efficiency(self):
        """Test memory efficiency of batch processing"""
        import tracemalloc

        extractor = RustArticleExtractor()

        samples = [
            (create_sample_html(10), f"https://example.com/article{i}")
            for i in range(100)
        ]

        tracemalloc.start()

        current, peak = tracemalloc.get_traced_memory()
        baseline_peak = peak

        result = extractor.extract_batch(samples)

        current, peak = tracemalloc.get_traced_memory()

        memory_used = (peak - baseline_peak) / 1024 / 1024  # MB
        memory_per_article = memory_used / len(samples)

        print(f"\nBatch Memory Efficiency:")
        print(f"  Total memory: {memory_used:.2f} MB")
        print(f"  Per article: {memory_per_article*1024:.2f} KB")

        tracemalloc.stop()

        assert len(result['articles']) == 100
        # Memory per article should be reasonable (< 100KB)
        assert memory_per_article < 0.1, "Memory per article should be < 100KB"


class TestInitializationPerformance:
    """Initialization performance tests"""

    def test_extractor_initialization_time(self):
        """Measure time to initialize extractor"""
        times = []

        for _ in range(50):
            start = time.perf_counter()
            extractor = RustArticleExtractor()
            times.append(time.perf_counter() - start)

        avg_time = statistics.mean(times)

        print(f"\nExtractor Initialization:")
        print(f"  Average: {avg_time*1000:.2f} ms")

        # Initialization should be fast (< 10ms)
        assert avg_time < 0.01, "Initialization should be < 10ms"

    def test_model_loading_time(self, tmp_path):
        """Measure time to load model"""
        # Create dummy model file
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"dummy model data")

        start = time.perf_counter()
        extractor = RustArticleExtractor(model=str(model_path))
        load_time = time.perf_counter() - start

        print(f"\nModel Loading Time: {load_time*1000:.2f} ms")

        # Model loading should be reasonably fast
        assert load_time < 1.0, "Model loading should be < 1 second"


@pytest.mark.benchmark
class TestBenchmarkSuite:
    """Comprehensive benchmark suite"""

    def test_extraction_benchmark(self, benchmark):
        """Benchmark using pytest-benchmark"""
        extractor = RustArticleExtractor()
        html = create_sample_html(10)
        url = "https://example.com/article"

        result = benchmark(extractor.extract, html, url)

        assert result['quality_score'] > 0

        print(f"\nBenchmark Results:")
        print(f"  Mean: {benchmark.stats['mean']*1000:.2f} ms")
        print(f"  Median: {benchmark.stats['median']*1000:.2f} ms")
        print(f"  Std Dev: {benchmark.stats['stddev']*1000:.2f} ms")

    def test_batch_benchmark(self, benchmark):
        """Benchmark batch extraction"""
        extractor = RustArticleExtractor()

        samples = [
            (create_sample_html(10), f"https://example.com/article{i}")
            for i in range(50)
        ]

        result = benchmark(extractor.extract_batch, samples)

        assert len(result['articles']) == 50


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
