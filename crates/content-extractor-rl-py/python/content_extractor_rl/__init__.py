"""
Content Extractor RL - RL-based article extraction from HTML

This module provides a high-performance article extractor built in Rust
with Python bindings.

Example:
    >>> from content_extractor_rl import RustArticleExtractor
    >>> extractor = RustArticleExtractor(
    ...     model="path/to/model.onnx",
    ...     site_profile="path/to/profile.json"
    ... )
    >>> result = extractor.extract(
    ...     website_page_html="<html>...</html>",
    ...     url="https://example.com/article"
    ... )
    >>> print(result['content'])
"""

from .content_extractor_rl_rs import RustArticleExtractor

__version__ = "0.1.0"
__all__ = ["RustArticleExtractor"]


class ArticleExtractor:
    """
    High-level Python wrapper for the Rust article extractor.

    This class provides a more Pythonic interface to the underlying
    Rust implementation.

    Args:
        model_path (str, optional): Path to trained ONNX model
        site_profile_path (str, optional): Path to site profile JSON

    Example:
        >>> extractor = ArticleExtractor(model_path="model.onnx")
        >>> result = extractor.extract_from_html(html_content, url)
        >>> print(result.content)
    """

    def __init__(self, model_path=None, site_profile_path=None):
        self._extractor = RustArticleExtractor(
            site_profile=site_profile_path,
            model=model_path
        )

    def extract_from_html(self, html, url):
        """
        Extract article from HTML content.

        Args:
            html (str): HTML content as string
            url (str): URL of the page

        Returns:
            dict: Dictionary containing:
                - content (str): Extracted article text
                - quality_score (float): Quality score (0-1)
                - url (str): Original URL
                - title (str, optional): Article title
                - date (str, optional): Publication date
                - method (str): Extraction method used
                - xpath (str): XPath to extracted content
        """
        return self._extractor.extract(html, url)

    def extract_batch(self, html_url_pairs):
        """
        Extract multiple articles in batch.

        Args:
            html_url_pairs (list): List of (html, url) tuples

        Returns:
            dict: Dictionary with 'articles' key containing list of results
        """
        return self._extractor.extract_batch(html_url_pairs)

    def train(self, html_samples, episodes=1000, improved=False):
        """
        Train the model on HTML samples.

        Args:
            html_samples (list): List of (html, url) tuples
            episodes (int): Number of training episodes
            improved (bool): Use improved training features

        Returns:
            dict: Training metrics including rewards and qualities
        """
        return self._extractor.train(html_samples, episodes, improved)

    @property
    def stats(self):
        """Get extraction statistics."""
        return self._extractor.get_stats()


# Convenience function
def extract_article(html, url, model_path=None):
    """
    Quick extraction function for single articles.

    Args:
        html (str): HTML content
        url (str): Page URL
        model_path (str, optional): Path to trained model

    Returns:
        dict: Extraction result
    """
    extractor = ArticleExtractor(model_path=model_path)
    return extractor.extract_from_html(html, url)

def print_device_info():
    """Print device information at startup."""
    from content_extractor_rl_rs import check_cuda_available

    print("╔════════════════════════════════════════╗")
    print("║   Content Extractor RL - Device Info     ║")
    print("╠════════════════════════════════════════╣")

    cuda_available = check_cuda_available()
    if cuda_available:
        print("║ Status: ✅ CUDA GPU Available         ║")
        print("║ Training: Will use GPU acceleration   ║")
    else:
        print("║ Status: 💻 CPU Mode                   ║")
        print("║ Training: Will use CPU                 ║")

    print("╚════════════════════════════════════════╝")

# Call it on import if in verbose mode
import os
if os.environ.get('CONTENT_EXTRACTOR_RL_VERBOSE'):
    print_device_info()