"""
Pytest configuration and fixtures
"""

import pytest


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as benchmarks"
    )


@pytest.fixture(scope="session")
def large_html_corpus():
    """Fixture providing large HTML corpus for performance tests"""
    corpus = []

    for i in range(100):
        html = f"""
        <html>
        <body>
            <article>
                <h1>Article {i}</h1>
                <p>{'Lorem ipsum dolor sit amet. ' * 20}</p>
                <p>{'Consectetur adipiscing elit. ' * 15}</p>
                <p>{'Sed do eiusmod tempor incididunt. ' * 18}</p>
            </article>
        </body>
        </html>
        """
        corpus.append((html, f"https://example.com/article{i}"))

    return corpus
