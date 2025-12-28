from article_extractor_rs import RustArticleExtractor
import os
import bz2

# Configure paths
os.environ["ARTICLE_EXTRACTOR_SITE_PROFILES"] = "/var/local/sss/article_extractor/site_profiles"
os.environ["ARTICLE_EXTRACTOR_MODELS_DIR"] = "/var/local/sss/article_extractor/models"

# Initialize with trained model
extractor = RustArticleExtractor(
    model="/var/local/sss/article_extractor/models/best_model.onnx"
)

# load example html:
html_file: str = "/var/local/sss/article_extractor/html_archive/2025-12-13/mod_en_in_timesofindia_260138.html.bz2"
article_url: str = "https://indianexpress.com/article/india/ec-extends-sir-in-6-states-10414938/"

with bz2.open(html_file, 'rt', encoding='utf-8') as f:
    html_content = f.read()

    # Extract from production HTML
    result = extractor.extract(
        website_page_html=html_content,
        url=article_url
    )

    print(f"Quality: {result['quality_score']:.2f}")
    print(f"Content (first 200 characters): {result['content'][:200]}...")
