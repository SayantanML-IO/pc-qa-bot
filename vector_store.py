"""
Improved vector store with robust error handling and type hints.
"""
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import logging
from pathlib import Path
from typing import Optional, List, Tuple
from device_manager import get_device_kwargs

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Custom exception for vector store operations."""
    pass


class VectorStore:
    """Vector store with embeddings for semantic search."""
    
    def __init__(
        self, 
        persist_directory: str = "./chroma_db", 
        collection_name: str = "hardware_articles"
    ):
        """
        Initialize vector store with embeddings model.
        
        Args:
            persist_directory: Where to save the vector database
            collection_name: Name of the collection
            
        Raises:
            VectorStoreError: If initialization fails
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        
        # Ensure directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.vector_store: Optional[Chroma] = None
        
        self._initialize_embeddings()
        self._load_or_create_store()
    
    def _initialize_embeddings(self) -> None:
        """Initialize embeddings model with device fallback."""
        try:
            logger.info("Loading embeddings model (nomic-embed-text-v1.5)...")
            
            device_kwargs = get_device_kwargs()
            device_kwargs['trust_remote_code'] = True
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name="nomic-ai/nomic-embed-text-v1.5",
                model_kwargs=device_kwargs
            )
            
            logger.info(f"✓ Embeddings model loaded on {device_kwargs['device'].upper()}")
        
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise VectorStoreError(f"Embeddings initialization failed: {e}")
    
    def _load_or_create_store(self) -> None:
        """Load existing store or create new one."""
        try:
            self.vector_store = Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            
            doc_count = self.count_documents()
            logger.info(f"✓ Vector store loaded: {self.persist_directory} ({doc_count} chunks)")
        
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise VectorStoreError(f"Vector store initialization failed: {e}")
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of LangChain Document objects
            
        Raises:
            VectorStoreError: If adding documents fails
        """
        if not documents:
            logger.warning("No documents to add")
            return
        
        if not self.vector_store:
            raise VectorStoreError("Vector store not initialized")
        
        try:
            self.vector_store.add_documents(documents)
            logger.info(f"✓ Added {len(documents)} document chunks to vector store")
        
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise VectorStoreError(f"Failed to add documents: {e}")
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 3
    ) -> List[Document]:
        """
        Search for documents similar to the query.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of Document objects
            
        Raises:
            VectorStoreError: If search fails
        """
        if not self.vector_store:
            raise VectorStoreError("Vector store not initialized")
        
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            logger.debug(f"Found {len(results)} relevant chunks for query")
            return results
        
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise VectorStoreError(f"Similarity search failed: {e}")
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 3
    ) -> List[Tuple[Document, float]]:
        """
        Search with relevance scores.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of tuples (Document, score)
            
        Raises:
            VectorStoreError: If search fails
        """
        if not self.vector_store:
            raise VectorStoreError("Vector store not initialized")
        
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            logger.debug(f"Found {len(results)} relevant chunks with scores")
            return results
        
        except Exception as e:
            logger.error(f"Search with score failed: {e}")
            raise VectorStoreError(f"Similarity search with score failed: {e}")
    
    def get_retriever(self, k: int = 3):
        """
        Get a retriever object for use with LangChain chains.
        
        Args:
            k: Number of documents to retrieve
            
        Returns:
            LangChain retriever object
            
        Raises:
            VectorStoreError: If retriever creation fails
        """
        if not self.vector_store:
            raise VectorStoreError("Vector store not initialized")
        
        try:
            return self.vector_store.as_retriever(search_kwargs={"k": k})
        except Exception as e:
            logger.error(f"Failed to create retriever: {e}")
            raise VectorStoreError(f"Retriever creation failed: {e}")
    
    def count_documents(self) -> int:
        """
        Get total number of chunks in the store.
        
        Returns:
            Number of document chunks
        """
        if not self.vector_store:
            return 0
        
        try:
            collection = self.vector_store._collection
            return collection.count()
        except Exception as e:
            logger.error(f"Failed to count documents: {e}")
            return 0
    
    def delete_collection(self) -> None:
        """
        Delete the entire collection (use with caution!).
        
        Raises:
            VectorStoreError: If deletion fails
        """
        if not self.vector_store:
            raise VectorStoreError("Vector store not initialized")
        
        try:
            self.vector_store.delete_collection()
            logger.info("✓ Collection deleted successfully")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise VectorStoreError(f"Collection deletion failed: {e}")
    
    def health_check(self) -> dict:
        """
        Check vector store health.
        
        Returns:
            Dictionary with health status
        """
        try:
            doc_count = self.count_documents()
            
            return {
                "status": "healthy",
                "document_count": doc_count,
                "collection_name": self.collection_name,
                "persist_directory": str(self.persist_directory)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }