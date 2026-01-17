"""
Input validation and sanitization utilities.
"""
import re
import logging
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class InputValidator:
    """Validates and sanitizes user inputs."""
    
    # Hardware-related keywords for relevance checking
    HARDWARE_KEYWORDS = {
        'gpu', 'cpu', 'graphics', 'processor', 'motherboard', 'ram', 'memory',
        'ssd', 'hdd', 'storage', 'nvme', 'pcie', 'ddr4', 'ddr5', 'nvidia',
        'amd', 'intel', 'rtx', 'radeon', 'geforce', 'ryzen', 'core i',
        'gaming', 'benchmark', 'performance', 'fps', 'cooling', 'thermal',
        'overclocking', 'case', 'psu', 'power supply', 'monitor', 'display',
        'refresh rate', 'resolution', '4k', '1440p', 'Hz', 'peripheral',
        'keyboard', 'mouse', 'headset', 'rgb', 'hardware', 'pc', 'computer',
        'laptop', 'desktop', 'workstation', 'chipset', 'socket', 'tdp',
        'wattage', 'performance', 'review', 'release', 'announcement'
    }
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """
        Validate URL format.
        
        Args:
            url: URL string to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
        except Exception:
            return False
    
    @staticmethod
    def validate_question(question: str, max_length: int = 1000) -> str:
        """
        Validate and sanitize user question.
        
        Args:
            question: User's question
            max_length: Maximum allowed length
            
        Returns:
            Sanitized question string
            
        Raises:
            ValidationError: If question is invalid
        """
        if not question or not question.strip():
            raise ValidationError("Question cannot be empty")
        
        question = question.strip()
        
        if len(question) > max_length:
            raise ValidationError(f"Question too long (max {max_length} characters)")
        
        if len(question) < 3:
            raise ValidationError("Question too short (min 3 characters)")
        
        # Remove potentially harmful characters
        question = re.sub(r'[<>{}\\]', '', question)
        
        return question
    
    @staticmethod
    def validate_article_text(text: str, min_length: int = 100) -> bool:
        """
        Validate article text content.
        
        Args:
            text: Article text
            min_length: Minimum required length
            
        Returns:
            True if valid, False otherwise
        """
        if not text or not text.strip():
            return False
        
        text = text.strip()
        
        if len(text) < min_length:
            logger.debug(f"Article text too short: {len(text)} chars")
            return False
        
        # Check if text has reasonable word count
        words = text.split()
        if len(words) < 20:
            logger.debug(f"Article has too few words: {len(words)}")
            return False
        
        return True
    
    @classmethod
    def is_hardware_related(cls, text: str, threshold: float = 0.15) -> bool:
        """
        Check if text is hardware-related based on keyword matching.
        
        Args:
            text: Text to check
            threshold: Minimum ratio of hardware keywords required (0-1)
            
        Returns:
            True if text appears to be hardware-related
        """
        if not text:
            return False
        
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        if len(words) < 10:
            return False
        
        # Count hardware keyword matches
        matches = sum(1 for word in words if word in cls.HARDWARE_KEYWORDS)
        
        # Check for multi-word keywords
        for keyword in cls.HARDWARE_KEYWORDS:
            if ' ' in keyword and keyword in text_lower:
                matches += 2  # Weight multi-word matches higher
        
        ratio = matches / len(words)
        is_relevant = ratio >= threshold
        
        if not is_relevant:
            logger.debug(f"Article relevance too low: {ratio:.2%} (threshold: {threshold:.2%})")
        
        return is_relevant
    
    @staticmethod
    def validate_config(config: dict) -> None:
        """
        Validate configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Raises:
            ValidationError: If configuration is invalid
        """
        required_keys = ['rss_feeds', 'article_processing', 'vector_db', 'llm', 'qa']
        
        for key in required_keys:
            if key not in config:
                raise ValidationError(f"Missing required config key: {key}")
        
        # Validate RSS feeds
        if not config['rss_feeds']:
            raise ValidationError("No RSS feeds configured")
        
        for feed in config['rss_feeds']:
            if 'url' not in feed or 'name' not in feed:
                raise ValidationError(f"RSS feed missing 'url' or 'name': {feed}")
            
            if not InputValidator.validate_url(feed['url']):
                raise ValidationError(f"Invalid RSS feed URL: {feed['url']}")
        
        # Validate numeric parameters
        try:
            articles_per_feed = config['article_processing']['articles_per_feed']
            if not isinstance(articles_per_feed, int) or articles_per_feed < 1:
                raise ValidationError("articles_per_feed must be positive integer")
            
            chunk_size = config['article_processing']['chunk_size']
            if not isinstance(chunk_size, int) or chunk_size < 100:
                raise ValidationError("chunk_size must be >= 100")
            
            k_retriever = config['qa']['retriever_k']
            if not isinstance(k_retriever, int) or k_retriever < 1:
                raise ValidationError("retriever_k must be positive integer")
        
        except KeyError as e:
            raise ValidationError(f"Missing config parameter: {e}")
        except (TypeError, ValueError) as e:
            raise ValidationError(f"Invalid config parameter type: {e}")
    
    @staticmethod
    def sanitize_metadata(metadata: dict) -> dict:
        """
        Sanitize metadata dictionary to prevent injection attacks.
        
        Args:
            metadata: Metadata dictionary
            
        Returns:
            Sanitized metadata dictionary
        """
        sanitized = {}
        
        for key, value in metadata.items():
            # Only allow specific keys
            if key not in ['source', 'title', 'url', 'date', 'text_length', 
                          'chunk_id', 'total_chunks']:
                continue
            
            # Sanitize string values
            if isinstance(value, str):
                # Remove potentially harmful characters
                value = re.sub(r'[<>{}\\]', '', value)
                # Limit length
                value = value[:1000]
            
            sanitized[key] = value
        
        return sanitized