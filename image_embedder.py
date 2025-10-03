"""
Image Embedding System using Voyage AI for Warmwind OS
Converts UI screenshots to vector embeddings for similarity matching.
Uses Voyage AI's multimodal-3 model for superior embedding quality.
"""

import numpy as np
from PIL import Image
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import os
import pickle
import voyageai
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

@dataclass
class EmbeddingConfig:
    """Configuration for the Voyage AI embedding model."""
    model_name: str = "voyage-multimodal-3"
    api_key_env: str = "VOYAGE_API_KEY"  # Name of env variable, not its value
    timeout: int = 30

class ImageEmbedder:
    """
    Handles image embedding using Voyage AI's multimodal models.
    Converts screenshots to fixed-size vectors for similarity comparison.
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None, cache_file: str = "cache/embedding_cache.pkl"):
        """
        Initialize the image embedder with Voyage AI.
        
        Args:
            config: Configuration object for the model
            cache_file: File path for storing embedding cache
            
        Raises:
            ValueError: If API key is not found
        """
        self.config = config or EmbeddingConfig()
        self.cache_file = Path(cache_file)
        
        # Get API key from environment
        api_key = os.getenv(self.config.api_key_env)
        if not api_key:
            raise ValueError(
                f"Voyage AI API key not found. Please set the {self.config.api_key_env} "
                f"environment variable. Get your API key from: https://www.voyageai.com/"
            )
        
        print(f"🔧 Initializing Voyage AI client with model: {self.config.model_name}")
        
        # Initialize Voyage AI client
        self.client = voyageai.Client(api_key=api_key)
        
        # Set embedding dimension (voyage-multimodal-3 uses 1024 dimensions)
        self.embedding_dim = 1024
        
        # Simple cache: dict mapping image_data -> embedding
        self.cache: Dict[str, np.ndarray] = {}
        self._load_cache()
        
        print(f"✅ Voyage AI client initialized! Embedding dimension: {self.embedding_dim}")
        if self.cache:
            print(f"   Loaded {len(self.cache)} cached embeddings")
    
    def _load_cache(self):
        """Load cache from disk if it exists."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
            except Exception as e:
                print(f"⚠️  Could not load cache: {e}")
                self.cache = {}
    
    def _save_cache(self):
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            print(f"⚠️  Could not save cache: {e}")
    
    def embed_image(self, image: Image.Image, image_data: Optional[str] = None) -> np.ndarray:
        """
        Embed a single PIL Image using Voyage AI.
        Uses cache to avoid re-computing embeddings for same image.
        
        Args:
            image: PIL Image object
            image_data: Optional base64 image data (used as cache key)
            
        Returns:
            Numpy array of shape (embedding_dim,) - normalized unit vector
            
        Raises:
            Exception: If API call fails
        """
        # Check cache if image_data is provided
        if image_data and image_data in self.cache:
            return self.cache[image_data]
        
        # Call Voyage AI API
        try:
            result = self.client.multimodal_embed(
                inputs=[[image]],
                model=self.config.model_name,
                input_type="document"
            )
            
            # Extract embedding
            embedding = np.array(result.embeddings[0], dtype=np.float32)
            
            # Normalize to unit vector (standard for similarity search)
            embedding = embedding / np.linalg.norm(embedding)
            
            # Store in cache and save immediately
            if image_data:
                self.cache[image_data] = embedding
                self._save_cache()  # Save immediately after each new embedding
            
            return embedding
            
        except voyageai.error.VoyageError as e:
            raise Exception(f"Voyage AI API error: {str(e)}")
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score between 0 and 1
        """
        # Embeddings are already normalized, so dot product gives cosine similarity
        similarity = np.dot(embedding1, embedding2)
        return float(similarity)
    
    def find_most_similar(
        self, 
        query_embedding: np.ndarray, 
        candidate_embeddings: np.ndarray,
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find the most similar embeddings to a query embedding.
        
        Args:
            query_embedding: Query embedding vector (embedding_dim,)
            candidate_embeddings: Array of candidate embeddings (n, embedding_dim)
            top_k: Number of top matches to return
            
        Returns:
            List of (index, similarity_score) tuples, sorted by similarity
        """
        # Compute similarities (dot product since vectors are normalized)
        similarities = np.dot(candidate_embeddings, query_embedding)
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [(int(idx), float(similarities[idx])) for idx in top_indices]
        return results
    
    def save_cache(self):
        """Save the embedding cache to disk (called automatically after each new embedding)."""
        self._save_cache()



