"""
Main pipeline with parallel processing, metrics, and error handling.
"""
import yaml
import logging
import json
import os
import time
from pathlib import Path
from typing import Dict, Set, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from rss_scraper import (
    get_articles_from_feeds_parallel, 
    scrape_article_full_text,
    create_session
)
from document_processor import DocumentProcessor
from vector_store import VectorStore
from validators import InputValidator, ValidationError
from metrics import MetricsTracker
from device_manager import DeviceManager
from datetime import datetime
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROCESSED_URLS_FILE = "processed_urls.json"


class IndexingError(Exception):
    """Custom exception for indexing errors."""
    pass


def load_config(config_path: str = 'config.yaml') -> Optional[Dict]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary or None if failed
    """
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            logger.error(f"Config file not found: {config_path}")
            return None
        
        with open(config_file, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        # Validate configuration
        InputValidator.validate_config(config)
        logger.info(f"✓ Configuration loaded: {config_path}")
        return config
    
    except ValidationError as e:
        logger.error(f"Configuration validation failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return None


def load_processed_urls() -> Set[str]:
    """
    Load set of already processed URLs.
    
    Returns:
        Set of processed URLs
    """
    urls_file = Path(PROCESSED_URLS_FILE)
    
    if urls_file.exists():
        try:
            with open(urls_file, 'r') as f:
                urls = json.load(f)
                logger.info(f"Loaded {len(urls)} previously processed URLs")
                return set(urls)
        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse processed URLs file: {e}")
            return set()
        except Exception as e:
            logger.warning(f"Could not load processed URLs: {e}")
            return set()
    
    return set()


def save_processed_urls(urls: Set[str]) -> None:
    """
    Save set of processed URLs to file.
    
    Args:
        urls: Set of processed URLs
    """
    try:
        with open(PROCESSED_URLS_FILE, 'w') as f:
            json.dump(sorted(list(urls)), f, indent=2)
        logger.info(f"✓ Saved {len(urls)} processed URLs")
    except Exception as e:
        logger.error(f"Could not save processed URLs: {e}")


def scrape_article_with_metrics(
    article: Dict,
    session,
    metrics: Optional[MetricsTracker]
) -> Optional[Dict]:
    """
    Scrape article and log metrics.
    
    Args:
        article: Article metadata
        session: Requests session
        metrics: Metrics tracker
        
    Returns:
        Dictionary with text and metadata, or None if failed
    """
    start_time = time.time()
    url = article['link']
    
    try:
        full_text = scrape_article_full_text(article, session)
        processing_time = time.time() - start_time
        
        if full_text and InputValidator.validate_article_text(full_text):
            # Log success
            if metrics:
                metrics.log_article_processing(
                    url=url,
                    source=article.get('source', 'Unknown'),
                    success=True,
                    processing_time=processing_time,
                    text_length=len(full_text)
                )
            
            return {
                'text': full_text,
                'metadata': {
                    'source': article.get('source', 'Unknown'),
                    'title': article['title'],
                    'url': url,
                    'date': article.get('published', datetime.now().strftime('%Y-%m-%d')),
                    'text_length': len(full_text)
                }
            }
        else:
            # Log failure
            if metrics:
                metrics.log_article_processing(
                    url=url,
                    source=article.get('source', 'Unknown'),
                    success=False,
                    processing_time=processing_time,
                    error_message="Insufficient text extracted"
                )
            return None
    
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error scraping {url}: {e}")
        
        if metrics:
            metrics.log_article_processing(
                url=url,
                source=article.get('source', 'Unknown'),
                success=False,
                processing_time=processing_time,
                error_message=str(e)
            )
        return None


def scrape_articles_parallel(
    articles: List[Dict],
    max_workers: int = 5,
    metrics: Optional[MetricsTracker] = None
) -> List[Dict]:
    """
    Scrape multiple articles in parallel.
    
    Args:
        articles: List of article metadata
        max_workers: Maximum number of parallel workers
        metrics: Metrics tracker
        
    Returns:
        List of successfully scraped articles with text
    """
    logger.info(f"Scraping {len(articles)} articles (parallel with {max_workers} workers)...")
    
    scraped_articles = []
    session = create_session()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit scraping tasks
        future_to_article = {
            executor.submit(scrape_article_with_metrics, article, session, metrics): article
            for article in articles
        }
        
        # Use tqdm for progress tracking
        for future in tqdm(
            as_completed(future_to_article), 
            total=len(articles),
            desc="Scraping articles",
            unit="article"
        ):
            try:
                result = future.result()
                if result:
                    scraped_articles.append(result)
            except Exception as e:
                logger.error(f"Article scraping task failed: {e}")
    
    logger.info(f"✓ Successfully scraped {len(scraped_articles)}/{len(articles)} articles")
    return scraped_articles


def index_articles() -> None:
    """
    Main pipeline: Fetch RSS articles, extract text, and add to vector store.
    Only indexes NEW articles that haven't been processed before.
    
    Raises:
        IndexingError: If indexing fails critically
    """
    logger.info("="*70)
    logger.info("STARTING PC HARDWARE NEWS INDEXING")
    logger.info("="*70)
    
    pipeline_start = time.time()
    
    # Initialize device manager
    device_manager = DeviceManager()
    device_manager.log_device_info()
    
    # Initialize metrics tracker
    metrics = MetricsTracker()
    
    # Load config
    config = load_config()
    if not config:
        raise IndexingError("Failed to load configuration")
    
    # Load already processed URLs
    processed_urls = load_processed_urls()
    
    # Initialize components
    try:
        processor = DocumentProcessor(
            chunk_size=config['article_processing']['chunk_size'],
            chunk_overlap=config['article_processing']['chunk_overlap']
        )
        
        vector_store = VectorStore(
            persist_directory=config['vector_db']['persist_directory'],
            collection_name=config['vector_db']['collection_name']
        )
    except Exception as e:
        raise IndexingError(f"Component initialization failed: {e}")
    
    # Get RSS feed URLs
    feed_urls = [feed['url'] for feed in config['rss_feeds']]
    
    # Fetch articles from all feeds (parallel)
    logger.info(f"\nFetching articles from {len(feed_urls)} RSS feeds...")
    try:
        articles = get_articles_from_feeds_parallel(
            feed_urls, 
            articles_per_feed=config['article_processing']['articles_per_feed'],
            max_workers=5
        )
    except Exception as e:
        raise IndexingError(f"Failed to fetch RSS feeds: {e}")
    
    # Filter out already processed articles
    new_articles = [a for a in articles if a['link'] not in processed_urls]
    logger.info(f"New articles to process: {len(new_articles)}")
    
    if len(new_articles) == 0:
        logger.info("No new articles to index.")
        
        # Log system health
        doc_count = vector_store.count_documents()
        metrics.log_system_health(
            vector_store_size=doc_count,
            total_articles=len(processed_urls),
            device=device_manager.device
        )
        
        logger.info(f"Current vector store has {doc_count} document chunks.")
        return
    
    # Limit to max_articles
    max_articles = config['article_processing']['max_articles_to_process']
    articles_to_process = new_articles[:max_articles]
    
    logger.info(f"Processing up to {len(articles_to_process)} articles...")
    
    # Scrape articles in parallel
    try:
        scraped_articles = scrape_articles_parallel(
            articles_to_process,
            max_workers=5,
            metrics=metrics
        )
    except Exception as e:
        logger.error(f"Article scraping failed: {e}")
        scraped_articles = []
    
    if not scraped_articles:
        logger.warning("No articles successfully scraped")
        return
    
    # Process articles into chunks (with relevance filtering)
    logger.info(f"\nProcessing {len(scraped_articles)} articles into chunks...")
    try:
        documents = processor.process_batch(
            scraped_articles,
            filter_relevance=True  # Enable hardware relevance filtering
        )
    except Exception as e:
        raise IndexingError(f"Document processing failed: {e}")
    
    if not documents:
        logger.warning("No document chunks created (all filtered or failed)")
        return
    
    # Add to vector store
    logger.info(f"\nAdding {len(documents)} chunks to vector store...")
    try:
        vector_store.add_documents(documents)
    except Exception as e:
        raise IndexingError(f"Failed to add documents to vector store: {e}")
    
    # Update processed URLs
    new_urls = {a['link'] for a in articles_to_process if a['link']}
    processed_urls.update(new_urls)
    save_processed_urls(processed_urls)
    
    # Calculate final statistics
    pipeline_duration = time.time() - pipeline_start
    total_indexed = len(scraped_articles)
    total_failed = len(articles_to_process) - len(scraped_articles)
    doc_count = vector_store.count_documents()
    
    # Log indexing run metrics
    metrics.log_indexing_run(
        articles_fetched=len(articles),
        articles_indexed=total_indexed,
        articles_failed=total_failed,
        total_chunks=doc_count,
        duration_seconds=pipeline_duration,
        status="success"
    )
    
    # Log system health
    metrics.log_system_health(
        vector_store_size=doc_count,
        total_articles=len(processed_urls),
        device=device_manager.device
    )
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("INDEXING COMPLETE")
    logger.info("="*70)
    logger.info(f"Articles fetched:     {len(articles)}")
    logger.info(f"New articles indexed: {total_indexed}")
    logger.info(f"Failed/Filtered:      {total_failed}")
    logger.info(f"Document chunks:      {len(documents)}")
    logger.info(f"Total in store:       {doc_count}")
    logger.info(f"Duration:             {pipeline_duration:.1f}s")
    logger.info("="*70)
    
    if total_indexed > 0:
        logger.info("\n✓ You can now run 'python cli_improved.py' to ask questions!")


if __name__ == "__main__":
    try:
        index_articles()
    except IndexingError as e:
        logger.error(f"❌ Indexing failed: {e}")
        exit(1)
    except KeyboardInterrupt:
        logger.info("\n⚠ Indexing interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}", exc_info=True)
        exit(1)