"""
Improved RSS scraper with parallel processing and robust error handling.
"""
import feedparser
import requests
from bs4 import BeautifulSoup
import logging
from datetime import datetime
import time
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from validators import InputValidator

logger = logging.getLogger(__name__)


class ScraperError(Exception):
    """Custom exception for scraping errors."""
    pass


class RateLimiter:
    """Simple rate limiter to prevent overwhelming servers."""
    
    def __init__(self, requests_per_second: float = 2.0):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_second: Maximum requests per second
        """
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0.0
    
    def wait_if_needed(self) -> None:
        """Wait if necessary to maintain rate limit."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()


def create_session() -> requests.Session:
    """
    Create a requests session with retry logic.
    
    Returns:
        Configured requests.Session object
    """
    session = requests.Session()
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session


def fetch_rss_feed(
    feed_url: str, 
    max_articles: int = 50,
    session: Optional[requests.Session] = None
) -> List[Dict]:
    """
    Fetch articles from an RSS feed.
    
    Args:
        feed_url: URL of the RSS feed
        max_articles: Maximum number of articles to fetch
        session: Optional requests session
        
    Returns:
        List of article dictionaries
    """
    logger.info(f"Fetching RSS feed: {feed_url}")
    
    if not InputValidator.validate_url(feed_url):
        logger.error(f"Invalid feed URL: {feed_url}")
        return []
    
    try:
        # Use session if provided, otherwise use feedparser directly
        if session:
            response = session.get(feed_url, timeout=30)
            response.raise_for_status()
            feed = feedparser.parse(response.content)
        else:
            feed = feedparser.parse(feed_url)
        
        if feed.bozo and not feed.entries:
            logger.warning(f"Feed parsing error for {feed_url}: {feed.bozo_exception}")
            return []
        
        articles = []
        for entry in feed.entries[:max_articles]:
            article = {
                'title': entry.get('title', 'No Title'),
                'link': entry.get('link', ''),
                'summary': entry.get('summary', ''),
                'published': entry.get('published', datetime.now().strftime('%Y-%m-%d')),
                'source': feed.feed.get('title', 'Unknown Source')
            }
            
            # Validate article has essential fields
            if article['link'] and InputValidator.validate_url(article['link']):
                articles.append(article)
        
        logger.info(f"✓ Found {len(articles)} valid articles from {feed_url}")
        return articles
    
    except requests.Timeout:
        logger.error(f"⏱ Timeout fetching RSS feed: {feed_url}")
        return []
    
    except requests.RequestException as e:
        logger.error(f"⚠ Request error for {feed_url}: {e}")
        return []
    
    except Exception as e:
        logger.error(f"⚠ Unexpected error fetching RSS feed {feed_url}: {e}")
        return []


def extract_article_text(
    article_url: str, 
    timeout: int = 30,
    session: Optional[requests.Session] = None
) -> Optional[str]:
    """
    Extract full article text from URL.
    
    Args:
        article_url: URL of the article
        timeout: Request timeout in seconds
        session: Optional requests session with retry logic
        
    Returns:
        Extracted text or None if failed
    """
    logger.debug(f"Extracting text from: {article_url}")
    
    if not InputValidator.validate_url(article_url):
        logger.error(f"Invalid article URL: {article_url}")
        return None
    
    if session is None:
        session = create_session()
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                         '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = session.get(article_url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'iframe', 
                            'aside', 'header', 'form']):
            element.decompose()
        
        article_text = ""
        
        # Method 1: Look for article tag
        article = soup.find('article')
        if article:
            article_text = article.get_text(separator='\n', strip=True)
        
        # Method 2: Look for main content divs
        if not article_text or len(article_text) < 200:
            content_divs = soup.find_all(
                ['div', 'main'], 
                class_=lambda x: x and any(
                    keyword in str(x).lower() 
                    for keyword in ['content', 'article', 'post', 'entry', 'body']
                )
            )
            for div in content_divs:
                text = div.get_text(separator='\n', strip=True)
                if len(text) > len(article_text):
                    article_text = text
        
        # Method 3: Fall back to paragraphs
        if not article_text or len(article_text) < 200:
            paragraphs = soup.find_all('p')
            article_text = '\n'.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
        
        # Clean up text
        lines = [line.strip() for line in article_text.split('\n') if line.strip()]
        article_text = '\n'.join(lines)
        
        if InputValidator.validate_article_text(article_text):
            logger.debug(f"✓ Extracted {len(article_text)} characters")
            return article_text
        else:
            logger.warning(f"⚠ Insufficient text extracted from {article_url}")
            return None
    
    except requests.Timeout:
        logger.error(f"⏱ Timeout extracting text from {article_url}")
        return None
    
    except requests.RequestException as e:
        logger.error(f"⚠ Request error for {article_url}: {e}")
        return None
    
    except Exception as e:
        logger.error(f"⚠ Unexpected error extracting text from {article_url}: {e}")
        return None


def get_articles_from_feeds_parallel(
    feed_urls: List[str], 
    articles_per_feed: int = 20,
    max_workers: int = 5
) -> List[Dict]:
    """
    Fetch articles from multiple RSS feeds in parallel.
    
    Args:
        feed_urls: List of RSS feed URLs
        articles_per_feed: Number of articles per feed
        max_workers: Maximum number of parallel workers
        
    Returns:
        List of article dictionaries
    """
    logger.info(f"Fetching articles from {len(feed_urls)} feeds (parallel)...")
    
    all_articles = []
    session = create_session()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all feed fetch tasks
        future_to_url = {
            executor.submit(fetch_rss_feed, url, articles_per_feed, session): url 
            for url in feed_urls
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                articles = future.result()
                all_articles.extend(articles)
                logger.debug(f"Completed: {url}")
            except Exception as e:
                logger.error(f"⚠ Feed fetch failed for {url}: {e}")
    
    logger.info(f"✓ Total articles fetched: {len(all_articles)}")
    return all_articles


def scrape_article_full_text(
    article_metadata: Dict,
    session: Optional[requests.Session] = None
) -> Optional[str]:
    """
    Given article metadata from RSS, fetch and extract full article text.
    
    Args:
        article_metadata: Dictionary with article metadata
        session: Optional requests session
        
    Returns:
        Full text string or None if failed
    """
    url = article_metadata.get('link')
    if not url:
        logger.warning("Article metadata missing 'link' field")
        return None
    
    # Try to get full text
    full_text = extract_article_text(url, session=session)
    
    # If extraction fails, use RSS summary as fallback
    if not full_text:
        summary = article_metadata.get('summary', '')
        if summary:
            # Clean HTML from summary
            soup = BeautifulSoup(summary, 'html.parser')
            full_text = soup.get_text(strip=True)
            
            if InputValidator.validate_article_text(full_text, min_length=50):
                logger.info(f"✓ Using RSS summary as fallback ({len(full_text)} chars)")
                return full_text
    
    return full_text


# Backward compatibility
def get_articles_from_feeds(
    feed_urls: List[str], 
    articles_per_feed: int = 20,
    delay_between_feeds: float = 1
) -> List[Dict]:
    """
    Sequential version for backward compatibility.
    For better performance, use get_articles_from_feeds_parallel().
    
    Args:
        feed_urls: List of RSS feed URLs
        articles_per_feed: Number of articles per feed
        delay_between_feeds: Delay between feeds (for rate limiting)
        
    Returns:
        List of article dictionaries
    """
    logger.info(f"Fetching articles from {len(feed_urls)} feeds (sequential)...")
    
    all_articles = []
    rate_limiter = RateLimiter(requests_per_second=1.0 / delay_between_feeds)
    session = create_session()
    
    for i, feed_url in enumerate(feed_urls):
        rate_limiter.wait_if_needed()
        articles = fetch_rss_feed(feed_url, articles_per_feed, session)
        all_articles.extend(articles)
    
    logger.info(f"✓ Total articles fetched: {len(all_articles)}")
    return all_articles