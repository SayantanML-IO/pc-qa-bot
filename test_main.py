"""
Comprehensive test suite for PC Hardware RAG system.
Run with: pytest test_main.py -v
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import shutil

# Import modules to test
from validators import InputValidator, ValidationError
from device_manager import DeviceManager, get_device
from metrics import MetricsTracker
from document_processor import DocumentProcessor, DocumentProcessorError
from vector_store import VectorStore, VectorStoreError


class TestInputValidator:
    """Test input validation functions."""
    
    def test_validate_url_valid(self):
        """Test URL validation with valid URLs."""
        assert InputValidator.validate_url("https://example.com")
        assert InputValidator.validate_url("http://example.com/path")
        assert InputValidator.validate_url("https://example.com/path?query=1")
    
    def test_validate_url_invalid(self):
        """Test URL validation with invalid URLs."""
        assert not InputValidator.validate_url("")
        assert not InputValidator.validate_url("not-a-url")
        assert not InputValidator.validate_url("ftp://example.com")
        assert not InputValidator.validate_url("example.com")
    
    def test_validate_question_valid(self):
        """Test question validation with valid questions."""
        question = "What are the latest GPUs?"
        result = InputValidator.validate_question(question)
        assert result == question
        
        long_question = "A" * 500
        result = InputValidator.validate_question(long_question)
        assert len(result) == 500
    
    def test_validate_question_empty(self):
        """Test question validation with empty input."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            InputValidator.validate_question("")
        
        with pytest.raises(ValidationError, match="cannot be empty"):
            InputValidator.validate_question("   ")
    
    def test_validate_question_too_short(self):
        """Test question validation with too short input."""
        with pytest.raises(ValidationError, match="too short"):
            InputValidator.validate_question("ab")
    
    def test_validate_question_too_long(self):
        """Test question validation with too long input."""
        long_question = "A" * 2000
        with pytest.raises(ValidationError, match="too long"):
            InputValidator.validate_question(long_question)
    
    def test_validate_question_sanitization(self):
        """Test that dangerous characters are removed."""
        question = "What is <script>alert('xss')</script> the best GPU?"
        result = InputValidator.validate_question(question)
        assert "<script>" not in result
        assert "alert" in result  # Content remains, tags removed
    
    def test_validate_article_text_valid(self):
        """Test article text validation with valid text."""
        text = "This is a valid article " * 20
        assert InputValidator.validate_article_text(text)
    
    def test_validate_article_text_too_short(self):
        """Test article text validation with short text."""
        assert not InputValidator.validate_article_text("Too short")
        assert not InputValidator.validate_article_text("")
    
    def test_is_hardware_related_positive(self):
        """Test hardware relevance detection with relevant text."""
        text = """
        NVIDIA has announced their new RTX 5090 graphics card with 
        impressive performance improvements. The GPU features 24GB of 
        GDDR7 memory and supports ray tracing. Gaming benchmarks show 
        significant FPS improvements over the RTX 4090.
        """
        assert InputValidator.is_hardware_related(text)
    
    def test_is_hardware_related_negative(self):
        """Test hardware relevance detection with irrelevant text."""
        text = """
        The weather today is sunny with a chance of rain. 
        I went to the park and saw many birds flying around.
        It was a beautiful day for outdoor activities.
        """
        assert not InputValidator.is_hardware_related(text)
    
    def test_validate_config_valid(self):
        """Test configuration validation with valid config."""
        config = {
            'rss_feeds': [
                {'name': 'Test', 'url': 'https://example.com/feed'}
            ],
            'article_processing': {
                'articles_per_feed': 20,
                'chunk_size': 1000,
                'chunk_overlap': 200,
                'max_articles_to_process': 80
            },
            'vector_db': {
                'persist_directory': './test_db',
                'collection_name': 'test'
            },
            'llm': {'provider': 'groq', 'groq': {'model': 'test'}},
            'qa': {'retriever_k': 5, 'reranker_top_n': 3, 'temperature': 0.3}
        }
        InputValidator.validate_config(config)  # Should not raise
    
    def test_validate_config_missing_key(self):
        """Test configuration validation with missing keys."""
        config = {'rss_feeds': []}
        with pytest.raises(ValidationError, match="Missing required config key"):
            InputValidator.validate_config(config)
    
    def test_validate_config_empty_feeds(self):
        """Test configuration validation with empty feed list."""
        config = {
            'rss_feeds': [],
            'article_processing': {},
            'vector_db': {},
            'llm': {},
            'qa': {}
        }
        with pytest.raises(ValidationError, match="No RSS feeds configured"):
            InputValidator.validate_config(config)
    
    def test_sanitize_metadata(self):
        """Test metadata sanitization."""
        metadata = {
            'source': 'Test <script>',
            'title': 'Title with {brackets}',
            'url': 'https://example.com',
            'malicious_key': 'should be removed',
            'date': '2025-01-01'
        }
        
        result = InputValidator.sanitize_metadata(metadata)
        
        assert 'source' in result
        assert '<script>' not in result['source']
        assert 'malicious_key' not in result
        assert result['url'] == 'https://example.com'


class TestDeviceManager:
    """Test device management functionality."""
    
    def test_device_manager_singleton(self):
        """Test that DeviceManager is a singleton."""
        dm1 = DeviceManager()
        dm2 = DeviceManager()
        assert dm1 is dm2
    
    def test_get_device(self):
        """Test get_device function."""
        device = get_device()
        assert device in ['cuda', 'cpu']
    
    def test_get_device_kwargs(self):
        """Test get_device_kwargs function."""
        from device_manager import get_device_kwargs
        kwargs = get_device_kwargs()
        assert 'device' in kwargs
        assert kwargs['device'] in ['cuda', 'cpu']
    
    @patch('torch.cuda.is_available')
    def test_device_manager_cuda_available(self, mock_cuda):
        """Test device manager when CUDA is available."""
        mock_cuda.return_value = True
        # Force re-initialization
        DeviceManager._instance = None
        dm = DeviceManager()
        assert dm.is_cuda_available
        assert dm.device == 'cuda'
    
    @patch('torch.cuda.is_available')
    def test_device_manager_cuda_unavailable(self, mock_cuda):
        """Test device manager when CUDA is not available."""
        mock_cuda.return_value = False
        # Force re-initialization
        DeviceManager._instance = None
        dm = DeviceManager()
        assert not dm.is_cuda_available
        assert dm.device == 'cpu'


class TestMetricsTracker:
    """Test metrics tracking functionality."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test_metrics.db"
        yield str(db_path)
        shutil.rmtree(temp_dir)
    
    def test_metrics_initialization(self, temp_db):
        """Test metrics tracker initialization."""
        metrics = MetricsTracker(temp_db)
        assert Path(temp_db).exists()
    
    def test_log_indexing_run(self, temp_db):
        """Test logging indexing run."""
        metrics = MetricsTracker(temp_db)
        metrics.log_indexing_run(
            articles_fetched=100,
            articles_indexed=80,
            articles_failed=20,
            total_chunks=400,
            duration_seconds=120.5,
            status="success"
        )
        
        stats = metrics.get_recent_stats(days=1)
        assert stats['indexing']['runs'] == 1
        assert stats['indexing']['total_indexed'] == 80
    
    def test_log_article_processing(self, temp_db):
        """Test logging article processing."""
        metrics = MetricsTracker(temp_db)
        metrics.log_article_processing(
            url="https://example.com",
            source="Test Source",
            success=True,
            processing_time=2.5,
            text_length=5000,
            chunks_created=5
        )
        
        failure_rate = metrics.get_failure_rate(hours=1)
        assert failure_rate['total_attempts'] == 1
        assert failure_rate['failures'] == 0
    
    def test_log_query(self, temp_db):
        """Test logging query."""
        metrics = MetricsTracker(temp_db)
        metrics.log_query(
            question="Test question",
            response_time=1.5,
            chunks_retrieved=5,
            success=True
        )
        
        stats = metrics.get_recent_stats(days=1)
        assert stats['queries']['total'] == 1
        assert stats['queries']['success_rate'] == 100.0
    
    def test_log_system_health(self, temp_db):
        """Test logging system health."""
        metrics = MetricsTracker(temp_db)
        metrics.log_system_health(
            vector_store_size=1000,
            total_articles=200,
            device="cuda"
        )
        
        stats = metrics.get_recent_stats(days=1)
        assert stats['system']['vector_store_size'] == 1000
        assert stats['system']['device'] == "cuda"
    
    def test_failure_rate_calculation(self, temp_db):
        """Test failure rate calculation."""
        metrics = MetricsTracker(temp_db)
        
        # Log successes and failures
        for i in range(7):
            metrics.log_article_processing(
                url=f"https://example.com/{i}",
                source="Test",
                success=True
            )
        
        for i in range(3):
            metrics.log_article_processing(
                url=f"https://example.com/fail{i}",
                source="Test",
                success=False,
                error_message="Test error"
            )
        
        failure_rate = metrics.get_failure_rate(hours=24)
        assert failure_rate['total_attempts'] == 10
        assert failure_rate['failures'] == 3
        assert failure_rate['failure_rate'] == 30.0


class TestDocumentProcessor:
    """Test document processing functionality."""
    
    def test_processor_initialization(self):
        """Test processor initialization."""
        processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
        assert processor.chunk_size == 1000
        assert processor.chunk_overlap == 200
    
    def test_processor_invalid_chunk_size(self):
        """Test processor with invalid chunk size."""
        with pytest.raises(DocumentProcessorError, match="chunk_size must be at least 100"):
            DocumentProcessor(chunk_size=50)
    
    def test_processor_invalid_overlap(self):
        """Test processor with invalid overlap."""
        with pytest.raises(DocumentProcessorError):
            DocumentProcessor(chunk_size=1000, chunk_overlap=1500)
    
    def test_process_document_valid(self):
        """Test processing valid document."""
        processor = DocumentProcessor(chunk_size=200, chunk_overlap=50)
        
        text = """
        NVIDIA has announced their new RTX 5090 graphics card with 
        impressive performance improvements. The GPU features 24GB of 
        GDDR7 memory and supports ray tracing. Gaming benchmarks show 
        significant FPS improvements over the RTX 4090. This is a major
        release in the PC hardware space with many improvements.
        """ * 5  # Make it long enough
        
        metadata = {
            'source': 'Test Source',
            'title': 'Test Article',
            'url': 'https://example.com',
            'date': '2025-01-01'
        }
        
        docs = processor.process_document(text, metadata, filter_relevance=True)
        
        assert len(docs) > 0
        assert all(doc.metadata['source'] == 'Test Source' for doc in docs)
        assert all('chunk_id' in doc.metadata for doc in docs)
    
    def test_process_document_too_short(self):
        """Test processing document that's too short."""
        processor = DocumentProcessor()
        
        text = "Too short"
        metadata = {'source': 'Test'}
        
        docs = processor.process_document(text, metadata)
        assert len(docs) == 0
    
    def test_process_document_not_hardware_related(self):
        """Test that non-hardware content is filtered."""
        processor = DocumentProcessor()
        
        text = """
        The weather is nice today. I went for a walk in the park.
        There were many birds and trees. It was a peaceful day.
        I enjoyed the sunshine and fresh air very much.
        """ * 10  # Make it long enough
        
        metadata = {'source': 'Test'}
        
        docs = processor.process_document(text, metadata, filter_relevance=True)
        assert len(docs) == 0  # Should be filtered out
    
    def test_process_batch(self):
        """Test batch processing."""
        # FIXED: Changed chunk_overlap to 50 (must be < chunk_size)
        processor = DocumentProcessor(chunk_size=200, chunk_overlap=50)
        
        hardware_text = """
        AMD Ryzen 9 9950X is the latest processor from AMD featuring
        16 cores and 32 threads with impressive performance. The CPU
        uses 5nm process technology and supports DDR5 memory.
        """ * 5
        
        documents_data = [
            {
                'text': hardware_text,
                'metadata': {'source': 'Test1', 'title': 'Article 1', 'url': 'https://test1.com'}
            },
            {
                'text': hardware_text,
                'metadata': {'source': 'Test2', 'title': 'Article 2', 'url': 'https://test2.com'}
            }
        ]
        
        docs = processor.process_batch(documents_data, filter_relevance=True)
        assert len(docs) > 0
    
    def test_get_stats(self):
        """Test getting processor stats."""
        processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
        stats = processor.get_stats()
        
        assert stats['chunk_size'] == 1000
        assert stats['chunk_overlap'] == 200
        assert stats['overlap_percentage'] == 20.0


class TestVectorStore:
    """Test vector store functionality."""
    
    @pytest.fixture
    def temp_store_dir(self):
        """Create temporary directory for vector store."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    # FIXED: Changed 'vector_store_improved' to 'vector_store'
    @patch('vector_store.HuggingFaceEmbeddings')
    @patch('vector_store.Chroma')
    def test_vector_store_initialization(self, mock_chroma, mock_embeddings, temp_store_dir):
        """Test vector store initialization."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_chroma.return_value._collection = mock_collection
        
        vector_store = VectorStore(
            persist_directory=temp_store_dir,
            collection_name="test"
        )
        
        assert vector_store.persist_directory == Path(temp_store_dir)
        assert vector_store.collection_name == "test"
        mock_embeddings.assert_called_once()
    
    # FIXED: Changed 'vector_store_improved' to 'vector_store'
    @patch('vector_store.HuggingFaceEmbeddings')
    @patch('vector_store.Chroma')
    def test_vector_store_health_check(self, mock_chroma, mock_embeddings, temp_store_dir):
        """Test vector store health check."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 100
        mock_chroma.return_value._collection = mock_collection
        
        vector_store = VectorStore(persist_directory=temp_store_dir)
        health = vector_store.health_check()
        
        assert health['status'] == 'healthy'
        assert health['document_count'] == 100


# Integration test
class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_validator_and_processor_integration(self):
        """Test validator and processor working together."""
        processor = DocumentProcessor()
        
        # Valid hardware article
        text = """
        Intel Core i9-14900K is a powerful processor with 24 cores.
        It features excellent gaming performance and supports DDR5 RAM.
        The CPU has a base clock of 3.2 GHz and boost up to 6.0 GHz.
        """ * 10
        
        # Validate first
        assert InputValidator.validate_article_text(text)
        assert InputValidator.is_hardware_related(text)
        
        # Then process
        metadata = {'source': 'Test', 'title': 'Test', 'url': 'https://test.com'}
        docs = processor.process_document(text, metadata, filter_relevance=True)
        
        assert len(docs) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])