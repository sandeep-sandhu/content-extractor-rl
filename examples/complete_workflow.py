#!/usr/bin/env python3
"""
Complete workflow example: Train and extract with the article extractor
"""

import os
from pathlib import Path
from article_extractor import RustArticleExtractor

def main():
    # Setup paths
    data_dir = Path("./data")
    model_path = Path("./models/best_model.onnx")

    # Set environment variables
    os.environ["ARTICLE_EXTRACTOR_MODEL_PATH"] = str(model_path)
    os.environ["ARTICLE_EXTRACTOR_SITE_PROFILES"] = "./site_profiles"
    os.environ["ARTICLE_EXTRACTOR_OUTPUT_DIR"] = "./output"

    # Example HTML samples for training
    training_samples = [
        (
            """
            <html>
            <body>
                <article>
                    <h1>Breaking News</h1>
                    <p>Important news article content here...</p>
                    <p>More detailed information in second paragraph...</p>
                </article>
            </body>
            </html>
            """,
            "https://news.example.com/article1"
        ),
        # Add more samples...
    ]

    # Initialize extractor
    print("Initializing extractor...")
    extractor = RustArticleExtractor()

    # Train if model doesn't exist
    if not model_path.exists():
        print("Training model...")
        metrics = extractor.train(
            html_samples=training_samples,
            episodes=1000,
            improved=True
        )
        print(f"Training completed! Best quality: {metrics['best_avg_quality']:.3f}")
    else:
        print(f"Using existing model: {model_path}")
        extractor = RustArticleExtractor(model=str(model_path))

    # Extract from new HTML
    print("\nExtracting article...")
    test_html = """
    <html>
    <body>
        <article>
            <h1>Test Article</h1>
            <p>This is a test article for extraction.</p>
        </article>
    </body>
    </html>
    """

    result = extractor.extract(
        website_page_html=test_html,
        url="https://test.example.com/article"
    )

    print(f"\nExtraction Results:")
    print(f"  URL: {result['url']}")
    print(f"  Quality: {result['quality_score']:.3f}")
    print(f"  Method: {result['method']}")
    print(f"  Content length: {len(result['content'])} chars")
    print(f"\nExtracted content:")
    print("-" * 80)
    print(result['content'])
    print("-" * 80)

    # Get statistics
    stats = extractor.get_stats()
    print(f"\nExtractor statistics:")
    print(f"  Has model: {stats['has_model']}")
    print(f"  Profiles: {stats['num_profiles']}")

if __name__ == '__main__':
    main()
