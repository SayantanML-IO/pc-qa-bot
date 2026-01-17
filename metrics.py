"""
Metrics tracking and monitoring for the RAG system.
"""
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Tracks system metrics and performance in SQLite database."""
    
    def __init__(self, db_path: str = "./metrics.db"):
        """
        Initialize metrics tracker.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._initialize_db()
    
    def _initialize_db(self) -> None:
        """Create database tables if they don't exist."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Indexing metrics
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS indexing_runs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        articles_fetched INTEGER NOT NULL,
                        articles_indexed INTEGER NOT NULL,
                        articles_failed INTEGER NOT NULL,
                        total_chunks INTEGER NOT NULL,
                        duration_seconds REAL NOT NULL,
                        status TEXT NOT NULL
                    )
                """)
                
                # Article processing metrics
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS article_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        url TEXT NOT NULL,
                        source TEXT NOT NULL,
                        success INTEGER NOT NULL,
                        processing_time_seconds REAL,
                        error_message TEXT,
                        text_length INTEGER,
                        chunks_created INTEGER
                    )
                """)
                
                # Query metrics
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS query_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        question TEXT NOT NULL,
                        response_time_seconds REAL NOT NULL,
                        chunks_retrieved INTEGER NOT NULL,
                        success INTEGER NOT NULL,
                        error_message TEXT
                    )
                """)
                
                # System health metrics
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS system_health (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        vector_store_size INTEGER NOT NULL,
                        total_articles INTEGER NOT NULL,
                        device TEXT NOT NULL,
                        memory_usage_mb REAL
                    )
                """)
                
                conn.commit()
                logger.info(f"✓ Metrics database initialized: {self.db_path}")
        
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize metrics database: {e}")
            raise
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
    
    def log_indexing_run(
        self,
        articles_fetched: int,
        articles_indexed: int,
        articles_failed: int,
        total_chunks: int,
        duration_seconds: float,
        status: str = "success"
    ) -> None:
        """Log an indexing run."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO indexing_runs 
                    (timestamp, articles_fetched, articles_indexed, articles_failed, 
                     total_chunks, duration_seconds, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    articles_fetched,
                    articles_indexed,
                    articles_failed,
                    total_chunks,
                    duration_seconds,
                    status
                ))
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to log indexing run: {e}")
    
    def log_article_processing(
        self,
        url: str,
        source: str,
        success: bool,
        processing_time: Optional[float] = None,
        error_message: Optional[str] = None,
        text_length: Optional[int] = None,
        chunks_created: Optional[int] = None
    ) -> None:
        """Log individual article processing."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO article_metrics 
                    (timestamp, url, source, success, processing_time_seconds, 
                     error_message, text_length, chunks_created)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    url,
                    source,
                    1 if success else 0,
                    processing_time,
                    error_message,
                    text_length,
                    chunks_created
                ))
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to log article processing: {e}")
    
    def log_query(
        self,
        question: str,
        response_time: float,
        chunks_retrieved: int,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> None:
        """Log a query execution."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO query_metrics 
                    (timestamp, question, response_time_seconds, chunks_retrieved, 
                     success, error_message)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    question[:500],  # Truncate long questions
                    response_time,
                    chunks_retrieved,
                    1 if success else 0,
                    error_message
                ))
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to log query: {e}")
    
    def log_system_health(
        self,
        vector_store_size: int,
        total_articles: int,
        device: str,
        memory_usage_mb: Optional[float] = None
    ) -> None:
        """Log system health snapshot."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO system_health 
                    (timestamp, vector_store_size, total_articles, device, memory_usage_mb)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    vector_store_size,
                    total_articles,
                    device,
                    memory_usage_mb
                ))
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to log system health: {e}")
    
    def get_recent_stats(self, days: int = 7) -> dict:
        """Get statistics for recent operations."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Indexing stats
                cursor.execute("""
                    SELECT 
                        COUNT(*) as runs,
                        SUM(articles_indexed) as total_indexed,
                        AVG(duration_seconds) as avg_duration
                    FROM indexing_runs
                    WHERE timestamp >= datetime('now', '-' || ? || ' days')
                """, (days,))
                indexing_stats = cursor.fetchone()
                
                # Query stats
                cursor.execute("""
                    SELECT 
                        COUNT(*) as queries,
                        AVG(response_time_seconds) as avg_response_time,
                        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_queries
                    FROM query_metrics
                    WHERE timestamp >= datetime('now', '-' || ? || ' days')
                """, (days,))
                query_stats = cursor.fetchone()
                
                # Latest system health
                cursor.execute("""
                    SELECT vector_store_size, total_articles, device
                    FROM system_health
                    ORDER BY timestamp DESC
                    LIMIT 1
                """)
                health = cursor.fetchone()
                
                return {
                    "indexing": {
                        "runs": indexing_stats[0] or 0,
                        "total_indexed": indexing_stats[1] or 0,
                        "avg_duration": round(indexing_stats[2] or 0, 2)
                    },
                    "queries": {
                        "total": query_stats[0] or 0,
                        "avg_response_time": round(query_stats[1] or 0, 3),
                        "success_rate": round((query_stats[2] or 0) / max(query_stats[0] or 1, 1) * 100, 1)
                    },
                    "system": {
                        "vector_store_size": health[0] if health else 0,
                        "total_articles": health[1] if health else 0,
                        "device": health[2] if health else "unknown"
                    }
                }
        except sqlite3.Error as e:
            logger.error(f"Failed to get recent stats: {e}")
            return {}
    
    def get_failure_rate(self, hours: int = 24) -> dict:
        """Get failure rates for recent operations."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failures
                    FROM article_metrics
                    WHERE timestamp >= datetime('now', '-' || ? || ' hours')
                """, (hours,))
                
                result = cursor.fetchone()
                total = result[0] or 0
                failures = result[1] or 0
                
                return {
                    "total_attempts": total,
                    "failures": failures,
                    "failure_rate": round(failures / max(total, 1) * 100, 1)
                }
        except sqlite3.Error as e:
            logger.error(f"Failed to get failure rate: {e}")
            return {}