"""
Knowledge base for storing and retrieving therapeutic documents.
Handles document storage, embedding generation, and similarity search.
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from sentence_transformers import SentenceTransformer

class TherapeuticContent:
    """Represents a therapeutic document/technique in the knowledge base."""
    
    def __init__(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        self.content = content
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.doc_id = f"therapy_{int(self.created_at.timestamp())}"
        self.embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert therapeutic content to dictionary format."""
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "has_embedding": self.embedding is not None
        }

class KnowledgeBase:
    """
    In-memory knowledge base for storing and searching therapeutic documents.
    Uses sentence transformers for embedding generation and cosine similarity for search.
    """
    
    def __init__(self):
        """Initialize the knowledge base."""
        self.logger = logging.getLogger("rag_agent")
        
        # Storage
        self.therapeutic_content: Dict[str, TherapeuticContent] = {}
        self.embeddings: List[np.ndarray] = []
        self.doc_ids: List[str] = []
        
        # Configuration 
        self.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.embedding_model = None
        self.vector_dimension = 384
        self.max_documents = 100  # Limit for CodeSandbox
        
        # Initialize embedding model
        self._initialize_model()
        
        self.logger.info("Knowledge base initialized")
    
    def _initialize_model(self):
        """Initialize the sentence transformer model."""
        try:
            self.logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            self.logger.info("Embedding model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading embedding model: {e}")
            raise
    
    async def add_document(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a new document to the knowledge base.
        
        Args:
            content: The document text content
            metadata: Optional metadata (category, urgency, etc.)
            
        Returns:
            True if document was added successfully
        """
        try:
            # Check document limit
            if len(self.therapeutic_content) >= self.max_documents:
                self.logger.warning(f"Knowledge base at capacity ({self.max_documents} documents)")
                return False
            
            # Create therapeutic content
            content_item = TherapeuticContent(content, metadata)
            
            # Generate embedding
            embedding = await self._generate_embedding(content)
            if embedding is None:
                self.logger.error("Failed to generate embedding for document")
                return False
            
            content_item.embedding = embedding
            
            # Store therapeutic content
            self.therapeutic_content[content_item.doc_id] = content_item
            self.embeddings.append(embedding)
            self.doc_ids.append(content_item.doc_id)
            
            self.logger.info(f"Added therapeutic content {content_item.doc_id} to knowledge base")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding document: {e}")
            return False
    
    async def search(self, query: str, limit: int = 5, min_similarity: float = 0.1) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query.
        
        Args:
            query: Search query text
            limit: Maximum number of results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of matching documents with similarity scores
        """
        try:
            if not self.therapeutic_content:
                self.logger.info("No documents in knowledge base")
                return []
            
            # Generate query embedding
            query_embedding = await self._generate_embedding(query)
            if query_embedding is None:
                self.logger.error("Failed to generate query embedding")
                return []
            
            # Calculate similarities
            similarities = []
            for i, doc_embedding in enumerate(self.embeddings):
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                if similarity >= min_similarity:
                    doc_id = self.doc_ids[i]
                    content_item = self.therapeutic_content[doc_id]
                    similarities.append({
                        'doc_id': doc_id,
                        'content': content_item.content,
                        'metadata': content_item.metadata,
                        'similarity': float(similarity),
                        'created_at': content_item.created_at.isoformat()
                    })
            
            # Sort by similarity (highest first) and limit results
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            results = similarities[:limit]
            
            self.logger.info(f"Found {len(results)} matching documents for query")
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching knowledge base: {e}")
            return []
    
    async def _generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Generate embedding for text using sentence transformer.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if failed
        """
        try:
            # Run embedding generation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, 
                self.embedding_model.encode, 
                text
            )
            
            # Ensure consistent data type and shape
            embedding = np.array(embedding, dtype=np.float32)
            
            # Validate embedding dimension
            if embedding.shape[0] != self.vector_dimension:
                self.logger.error(f"Unexpected embedding dimension: {embedding.shape[0]}")
                return None
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}")
            return None
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        try:
            # Calculate norms
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            # Avoid division by zero
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            
            # Ensure result is between 0 and 1
            return max(0.0, float(similarity))
            
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def get_content_by_id(self, doc_id: str) -> Optional[TherapeuticContent]:
        """Get therapeutic content by its ID."""
        return self.therapeutic_content.get(doc_id)
    
    def get_document_count(self) -> int:
        """Get total number of therapeutic content items in the knowledge base."""
        return len(self.therapeutic_content)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get knowledge base statistics.
        
        Returns:
            Dictionary with knowledge base statistics
        """
        # Calculate category distribution
        categories = {}
        urgency_levels = {}
        
        for content_item in self.therapeutic_content.values():
            # Count categories
            category = content_item.metadata.get('category', 'uncategorized')
            categories[category] = categories.get(category, 0) + 1
            
            # Count urgency levels
            urgency = content_item.metadata.get('urgency', 'normal')
            urgency_levels[urgency] = urgency_levels.get(urgency, 0) + 1
        
        return {
            'total_documents': len(self.therapeutic_content),
            'total_embeddings': len(self.embeddings),
            'categories': categories,
            'urgency_levels': urgency_levels,
            'vector_dimension': self.vector_dimension,
            'model_name': self.embedding_model_name,
            'capacity_used': f"{len(self.therapeutic_content)}/{self.max_documents}",
            'capacity_percentage': int((len(self.therapeutic_content) / self.max_documents) * 100)
        }
    
    def search_by_category(self, category: str) -> List[TherapeuticContent]:
        """
        Get all therapeutic content in a specific category.
        
        Args:
            category: Category to search for
            
        Returns:
            List of therapeutic content in the category
        """
        results = []
        for content_item in self.therapeutic_content.values():
            if content_item.metadata.get('category') == category:
                results.append(content_item)
        
        return results
    
    def search_by_urgency(self, urgency: str) -> List[TherapeuticContent]:
        """
        Get all therapeutic content with a specific urgency level.
        
        Args:
            urgency: Urgency level to search for
            
        Returns:
            List of therapeutic content with the urgency level
        """
        results = []
        for content_item in self.therapeutic_content.values():
            if content_item.metadata.get('urgency') == urgency:
                results.append(content_item)
        
        return results
    
    async def update_document(self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update an existing document.
        
        Args:
            doc_id: ID of document to update
            content: New content
            metadata: New metadata
            
        Returns:
            True if document was updated successfully
        """
        try:
            if doc_id not in self.therapeutic_content:
                self.logger.error(f"Therapeutic content {doc_id} not found")
                return False
            
            # Find document index
            doc_index = self.doc_ids.index(doc_id)
            
            # Generate new embedding
            new_embedding = await self._generate_embedding(content)
            if new_embedding is None:
                return False
            
            # Update therapeutic content
            content_item = self.therapeutic_content[doc_id]
            content_item.content = content
            content_item.metadata = metadata or content_item.metadata
            content_item.embedding = new_embedding
            
            # Update embedding in list
            self.embeddings[doc_index] = new_embedding
            
            self.logger.info(f"Updated therapeutic content {doc_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating document: {e}")
            return False
    
    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document from the knowledge base.
        
        Args:
            doc_id: ID of document to remove
            
        Returns:
            True if document was removed successfully
        """
        try:
            if doc_id not in self.therapeutic_content:
                self.logger.error(f"Therapeutic content {doc_id} not found")
                return False
            
            # Find and remove from all lists
            doc_index = self.doc_ids.index(doc_id)
            
            del self.therapeutic_content[doc_id]
            del self.embeddings[doc_index]
            del self.doc_ids[doc_index]
            
            self.logger.info(f"Removed therapeutic content {doc_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error removing document: {e}")
            return False
    
    def clear_all_documents(self) -> bool:
        """
        Clear all therapeutic content from the knowledge base.
        
        Returns:
            True if all content was cleared
        """
        try:
            self.therapeutic_content.clear()
            self.embeddings.clear()
            self.doc_ids.clear()
            
            self.logger.info("Cleared all therapeutic content from knowledge base")
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing documents: {e}")
            return False
    
    async def get_similar_documents(self, doc_id: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Find therapeutic content similar to a given item.
        
        Args:
            doc_id: ID of the reference content
            limit: Maximum number of similar items to return
            
        Returns:
            List of similar therapeutic content with similarity scores
        """
        try:
            if doc_id not in self.therapeutic_content:
                return []
            
            reference_content = self.therapeutic_content[doc_id]
            if reference_content.embedding is None:
                return []
            
            # Calculate similarities to all other content
            similarities = []
            for other_id, other_content in self.therapeutic_content.items():
                if other_id != doc_id and other_content.embedding is not None:
                    similarity = self._cosine_similarity(reference_content.embedding, other_content.embedding)
                    similarities.append({
                        'doc_id': other_id,
                        'content': other_content.content,
                        'metadata': other_content.metadata,
                        'similarity': float(similarity)
                    })
            
            # Sort and limit
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return similarities[:limit]
            
        except Exception as e:
            self.logger.error(f"Error finding similar therapeutic content: {e}")
            return []
    
    def __str__(self) -> str:
        """String representation of the knowledge base."""
        return f"KnowledgeBase(therapeutic_content={len(self.therapeutic_content)}, model={self.embedding_model_name})"