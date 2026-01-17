"""
Improved document processor with type hints and error handling.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import logging
from typing import List, Dict, Optional
from validators import InputValidator, ValidationError

logger = logging.getLogger(__name__)


class DocumentProcessorError(Exception):
    """Custom exception for document processing errors."""
    pass


class DocumentProcessor:
    """Processes documents into chunks for vector storage."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize document processor with text splitting settings.
        
        Args:
            chunk_size: Size of each text chunk in characters
            chunk_overlap: Overlap between chunks to maintain context
            
        Raises:
            DocumentProcessorError: If parameters are invalid
        """
        if chunk_size < 100:
            raise DocumentProcessorError("chunk_size must be at least 100")
        
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise DocumentProcessorError(
                f"chunk_overlap must be between 0 and chunk_size ({chunk_size})"
            )
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        try:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            logger.info(
                f"✓ Document processor initialized "
                f"(chunk_size={chunk_size}, overlap={chunk_overlap})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize text splitter: {e}")
            raise DocumentProcessorError(f"Text splitter initialization failed: {e}")
    
    def process_document(
        self, 
        text: str, 
        metadata: Dict,
        filter_relevance: bool = True
    ) -> List[Document]:
        """
        Process a single document: validate, filter, split into chunks.
        
        Args:
            text: Full document text
            metadata: Dict with document info (url, source, date, etc.)
            filter_relevance: Whether to check if content is hardware-related
            
        Returns:
            List of LangChain Document objects
            
        Raises:
            DocumentProcessorError: If processing fails
        """
        if not InputValidator.validate_article_text(text):
            logger.warning(
                f"Document too short or invalid: {metadata.get('url', 'unknown')}"
            )
            return []
        
        # Check if content is hardware-related
        if filter_relevance:
            if not InputValidator.is_hardware_related(text):
                logger.info(
                    f"⊘ Document not hardware-related (filtered): "
                    f"{metadata.get('title', 'unknown')[:50]}..."
                )
                return []
        
        try:
            # Sanitize metadata
            safe_metadata = InputValidator.sanitize_metadata(metadata)
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            if not chunks:
                logger.warning(f"No chunks created for document: {metadata.get('url')}")
                return []
            
            # Create Document objects with metadata
            documents = []
            for i, chunk in enumerate(chunks):
                doc_metadata = safe_metadata.copy()
                doc_metadata['chunk_id'] = i
                doc_metadata['total_chunks'] = len(chunks)
                
                documents.append(Document(
                    page_content=chunk,
                    metadata=doc_metadata
                ))
            
            logger.info(
                f"✓ Processed document into {len(documents)} chunks: "
                f"{metadata.get('title', 'unknown')[:50]}..."
            )
            return documents
        
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise DocumentProcessorError(f"Document processing failed: {e}")
    
    def process_batch(
        self, 
        documents_data: List[Dict],
        filter_relevance: bool = True
    ) -> List[Document]:
        """
        Process multiple documents at once.
        
        Args:
            documents_data: List of dicts with 'text' and 'metadata' keys
            filter_relevance: Whether to filter out non-hardware content
            
        Returns:
            List of all Document objects from all documents
        """
        if not documents_data:
            logger.warning("No documents to process in batch")
            return []
        
        all_docs = []
        successful = 0
        failed = 0
        filtered = 0
        
        for doc_data in documents_data:
            try:
                if 'text' not in doc_data or 'metadata' not in doc_data:
                    logger.warning("Document missing 'text' or 'metadata' key")
                    failed += 1
                    continue
                
                docs = self.process_document(
                    text=doc_data['text'],
                    metadata=doc_data['metadata'],
                    filter_relevance=filter_relevance
                )
                
                if docs:
                    all_docs.extend(docs)
                    successful += 1
                else:
                    filtered += 1
            
            except DocumentProcessorError as e:
                logger.error(f"Document processing error: {e}")
                failed += 1
            except Exception as e:
                logger.error(f"Unexpected error in batch processing: {e}")
                failed += 1
        
        logger.info(
            f"✓ Batch processing complete: {len(all_docs)} total chunks "
            f"from {successful} documents "
            f"({filtered} filtered, {failed} failed)"
        )
        
        return all_docs
    
    def get_stats(self) -> Dict:
        """
        Get processor configuration stats.
        
        Returns:
            Dictionary with processor statistics
        """
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "overlap_percentage": round(self.chunk_overlap / self.chunk_size * 100, 1)
        }