"""
Comprehensive tests for Python bindings
"""

import pytest
import os
import tempfile
from pathlib import Path

# Only run if package is installed
try:
    from article_extractor import RustArticleExtractor, ArticleExtractor, extract_article
    PACKAGE_AVAILABLE = True
except ImportError:
    PACKAGE_AVAILABLE = False
    pytestmark = pytest.mark.skip("article_extractor package not installed")


@pytest.fixture
def sample_html():
    return """
    <html>
    <head><title>Test Article</title></head>
    <body>
        <article>
            <h1>Breaking News: Important Discovery</h1>
            <p>Scientists have made a groundbreaking discovery in the field of 
               artificial intelligence that could revolutionize technology.</p>
            <p>The research team developed a novel approach combining deep learning 
               with reinforcement learning to create more efficient models.</p>
            <p>This breakthrough has significant implications for various industries 
               including healthcare, finance, and autonomous systems.</p>
        </article>
    </body>
    </html>
    """


@pytest.fixture
def sample_url():
    return "https://example.com/article"


class TestRustArticleExtractor:
    """Tests for RustArticleExtractor class"""

    def test_initialization_no_args(self):
        """Test extractor can be initialized without arguments"""
        extractor = RustArticleExtractor()
        assert extractor is not None

    def test_initialization_with_model(self, tmp_path):
        """Test extractor initialization with model path"""
        # Create dummy model file
        model_path = tmp_path / "model.onnx"
        model_path.touch()

        extractor = RustArticleExtractor(model=str(model_path))
        assert extractor is not None

    def test_extract_basic(self, sample_html, sample_url):
        """Test basic extraction functionality"""
        extractor = RustArticleExtractor()
        result = extractor.extract(sample_html, sample_url)

        assert isinstance(result, dict)
        assert 'content' in result
        assert 'quality_score' in result
        assert 'url' in result
        assert 'method' in result

        assert result['url'] == sample_url
        assert len(result['content']) > 0
        assert 0.0 <= result['quality_score'] <= 1.0

    def test_extract_quality_score(self, sample_html, sample_url):
        """Test that quality score is reasonable for good content"""
        extractor = RustArticleExtractor()
        result = extractor.extract(sample_html, sample_url)

        # Good article should have quality > 0.5
        assert result['quality_score'] > 0.5

    def test_extract_filters_navigation(self):
        """Test that navigation elements are filtered out"""
        html = """
        <html>
        <body>
            <nav>
                <a href="/">Home</a>
                <a href="/about">About</a>
            </nav>
            <article>
                <p>This is the actual article content.</p>
            </article>
        </body>
        </html>
        """

        extractor = RustArticleExtractor()
        result = extractor.extract(html, "https://example.com")

        assert "Home" not in result['content']
        assert "About" not in result['content']
        assert "article content" in result['content']

    def test_extract_empty_html(self):
        """Test extraction with empty HTML"""
        extractor = RustArticleExtractor()
        result = extractor.extract("", "https://example.com")

        assert result['quality_score'] == 0.0
        assert len(result['content']) == 0

    def test_extract_batch(self, sample_html, sample_url):
        """Test batch extraction"""
        extractor = RustArticleExtractor()

        html_url_pairs = [
            (sample_html, sample_url),
            (sample_html, "https://example.com/article2"),
        ]

        result = extractor.extract_batch(html_url_pairs)

        assert isinstance(result, dict)
        assert 'articles' in result
        assert len(result['articles']) == 2

        for article in result['articles']:
            assert 'content' in article
            assert 'quality_score' in article

    def test_get_stats(self):
        """Test getting extractor statistics"""
        extractor = RustArticleExtractor()
        stats = extractor.get_stats()

        assert isinstance(stats, dict)
        assert 'has_model' in stats


class TestArticleExtractor:
    """Tests for high-level ArticleExtractor wrapper"""

    def test_initialization(self):
        """Test wrapper initialization"""
        extractor = ArticleExtractor()
        assert extractor is not None

    def test_extract_from_html(self, sample_html, sample_url):
        """Test extraction through wrapper"""
        extractor = ArticleExtractor()
        result = extractor.extract_from_html(sample_html, sample_url)

        assert isinstance(result, dict)
        assert result['url'] == sample_url
        assert len(result['content']) > 0

    def test_extract_batch(self, sample_html):
        """Test batch extraction through wrapper"""
        extractor = ArticleExtractor()

        pairs = [
            (sample_html, "https://example.com/1"),
            (sample_html, "https://example.com/2"),
        ]

        result = extractor.extract_batch(pairs)
        assert len(result['articles']) == 2

    def test_stats_property(self):
        """Test stats property"""
        extractor = ArticleExtractor()
        stats = extractor.stats

        assert isinstance(stats, dict)


class TestConvenienceFunctions:
    """Tests for convenience functions"""

    def test_extract_article_function(self, sample_html, sample_url):
        """Test quick extraction function"""
        result = extract_article(sample_html, sample_url)

        assert isinstance(result, dict)
        assert result['url'] == sample_url
        assert len(result['content']) > 0


class TestTraining:
    """Tests for training functionality"""

    @pytest.mark.slow
    def test_train_basic(self, sample_html, sample_url):
        """Test basic training (slow test)"""
        extractor = RustArticleExtractor()

        html_samples = [(sample_html, sample_url)] * 10

        metrics = extractor.train(
            html_samples=html_samples,
            episodes=10,  # Very short training for testing
            improved=False
        )

        assert isinstance(metrics, dict)
        assert 'episode_rewards' in metrics
        assert 'episode_qualities' in metrics
        assert 'best_avg_quality' in metrics

        assert len(metrics['episode_rewards']) == 10
        assert len(metrics['episode_qualities']) == 10

    @pytest.mark.slow
    def test_train_improved(self, sample_html, sample_url):
        """Test improved training mode"""
        extractor = RustArticleExtractor()

        html_samples = [(sample_html, sample_url)] * 10

        metrics = extractor.train(
            html_samples=html_samples,
            episodes=10,
            improved=True
        )

        assert isinstance(metrics, dict)
        assert metrics['best_avg_quality'] >= 0.0


class TestEnvironmentVariables:
    """Tests for environment variable configuration"""

    def test_model_path_env_var(self, tmp_path):
        """Test MODEL_PATH environment variable"""
        model_path = tmp_path / "test_model.onnx"
        model_path.touch()

        os.environ["ARTICLE_EXTRACTOR_MODEL_PATH"] = str(model_path)

        try:
            extractor = RustArticleExtractor()
            # Should initialize without error
            assert extractor is not None
        finally:
            del os.environ["ARTICLE_EXTRACTOR_MODEL_PATH"]

    def test_output_dir_env_var(self, tmp_path):
        """Test OUTPUT_DIR environment variable"""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        os.environ["ARTICLE_EXTRACTOR_OUTPUT_DIR"] = str(output_dir)

        try:
            extractor = RustArticleExtractor()
            assert extractor is not None
        finally:
            del os.environ["ARTICLE_EXTRACTOR_OUTPUT_DIR"]


class TestErrorHandling:
    """Tests for error handling"""

    def test_invalid_html(self):
        """Test handling of invalid HTML"""
        extractor = RustArticleExtractor()

        # Malformed HTML should not crash
        result = extractor.extract("<<<>>>", "https://example.com")

        # Should return empty result, not crash
        assert isinstance(result, dict)

    def test_invalid_url(self, sample_html):
        """Test handling of invalid URL"""
        extractor = RustArticleExtractor()

        # Invalid URL should not crash
        result = extractor.extract(sample_html, "not-a-url")

        assert isinstance(result, dict)

    def test_nonexistent_model(self):
        """Test handling of nonexistent model file"""
        # Should initialize but use baseline extractor
        extractor = RustArticleExtractor(model="/nonexistent/model.onnx")
        assert extractor is not None


@pytest.mark.benchmark
class TestPerformance:
    """Performance benchmarks"""

    def test_extraction_speed(self, benchmark, sample_html, sample_url):
        """Benchmark extraction speed"""
        extractor = RustArticleExtractor()

        def extract():
            return extractor.extract(sample_html, sample_url)

        result = benchmark(extract)
        assert result['quality_score'] > 0

    def test_batch_extraction_speed(self, benchmark, sample_html):
        """Benchmark batch extraction speed"""
        extractor = RustArticleExtractor()

        pairs = [(sample_html, f"https://example.com/{i}") for i in range(100)]

        def extract_batch():
            return extractor.extract_batch(pairs)

        result = benchmark(extract_batch)
        assert len(result['articles']) == 100


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
