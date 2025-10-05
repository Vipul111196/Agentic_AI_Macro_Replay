"""
Image Embedding System for Warmwind OS
Converts UI screenshots to vector embeddings for similarity matching.

Supports two providers:
1. Voyage AI - High-quality multimodal embeddings (1024 dimensions)
2. SelfMade - Simple pixel-based embeddings (256 dimensions)
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


class SelfMadeEmbedder:
    """
    Simple embedding based on image pixels.
    Resizes image to 16x16 and flattens to 256-dimensional vector.
    
    This is a lightweight alternative to API-based embeddings.
    """
    
    def __init__(self, cache_file: str = "cache/selfmade_embedding_cache.pkl"):
        """
        Initialize the self-made embedder.
        
        Args:
            cache_file: File path for storing embedding cache
        """
        self.cache_file = Path(cache_file)
        self.embedding_dim = 256  # 16x16 pixels
        
        # Simple cache: dict mapping image_data -> embedding
        self.cache: Dict[str, np.ndarray] = {}
        self._load_cache()
        
        print(f"✅ SelfMade Embedder initialized! Embedding dimension: {self.embedding_dim}")
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
            # Ensure cache directory exists
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            print(f"⚠️  Could not save cache: {e}")
    
    def embed_image(self, image: Image.Image, image_data: Optional[str] = None) -> np.ndarray:
        """
        Embed a single PIL Image using simple pixel-based approach.
        
        Process:
        1. Convert to grayscale
        2. Resize to 16x16 pixels
        3. Flatten to 256-dimensional vector
        4. Normalize to unit vector
        
        Args:
            image: PIL Image object
            image_data: Optional base64 image data (used as cache key)
            
        Returns:
            Numpy array of shape (256,) - normalized unit vector
        """
        # Check cache if image_data is provided
        if image_data and image_data in self.cache:
            return self.cache[image_data]
        
        # Convert to grayscale for simplicity
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize to 16x16
        image_resized = image.resize((16, 16), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and flatten
        pixels = np.array(image_resized, dtype=np.float32).flatten()
        
        # Normalize pixel values to [0, 1] range
        pixels = pixels / 255.0
        
        # Normalize to unit vector (for cosine similarity)
        norm = np.linalg.norm(pixels)
        if norm > 0:
            embedding = pixels / norm
        else:
            embedding = pixels
        
        # Store in cache
        if image_data:
            self.cache[image_data] = embedding
            self._save_cache()
        
        return embedding
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score between 0 and 1
        """
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
            query_embedding: Query embedding vector (256,)
            candidate_embeddings: Array of candidate embeddings (n, 256)
            top_k: Number of top matches to return
            
        Returns:
            List of (index, similarity_score) tuples, sorted by similarity
        """
        similarities = np.dot(candidate_embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(int(idx), float(similarities[idx])) for idx in top_indices]
        return results
    
    def save_cache(self):
        """Save the embedding cache to disk."""
        self._save_cache()


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


def create_embedder(provider: str = "voyage", config: Optional[EmbeddingConfig] = None, cache_file: Optional[str] = None):
    """
    Factory function to create the appropriate embedder based on provider.
    
    Args:
        provider: Embedding provider ("voyage" or "selfmade")
        config: Optional config for VoyageAI embedder
        cache_file: Optional cache file path (if not provided, uses default)
        
    Returns:
        Embedder instance (ImageEmbedder or SelfMadeEmbedder)
        
    Raises:
        ValueError: If provider is not recognized
    """
    provider = provider.lower().strip()
    
    if provider == "voyage":
        if cache_file:
            return ImageEmbedder(config=config, cache_file=cache_file)
        return ImageEmbedder(config=config)
    elif provider == "selfmade":
        if cache_file:
            return SelfMadeEmbedder(cache_file=cache_file)
        return SelfMadeEmbedder()
    else:
        raise ValueError(
            f"Unknown embedding provider: '{provider}'. "
            f"Must be 'voyage' or 'selfmade'"
        )


def get_embedding_dim(provider: str) -> int:
    """
    Get the embedding dimension for a given provider.
    
    Args:
        provider: Embedding provider ("voyage" or "selfmade")
        
    Returns:
        Embedding dimension
    """
    provider = provider.lower().strip()
    
    if provider == "voyage":
        return 1024
    elif provider == "selfmade":
        return 256
    else:
        raise ValueError(f"Unknown embedding provider: '{provider}'")


def demo_embedders():
    """
    Demo script showing both embedding providers in action.
    """
    print("=" * 70)
    print("🎨 IMAGE EMBEDDER DEMO - Comparing Both Providers")
    print("=" * 70)
    
    # Create a simple test image (100x100 grayscale gradient)
    print("\n📸 Creating test image (100x100 grayscale gradient)...")
    test_image = Image.new('L', (100, 100))
    for y in range(100):
        for x in range(100):
            test_image.putpixel((x, y), int((x + y) / 2 * 2.55))
    
    print("\n" + "=" * 70)
    print("Testing SelfMade Embedder (No API Required)")
    print("=" * 70)
    
    try:
        selfmade = SelfMadeEmbedder()
        embedding1 = selfmade.embed_image(test_image)
        
        print(f"✅ Embedding generated!")
        print(f"   • Dimension: {len(embedding1)}")
        print(f"   • Shape: {embedding1.shape}")
        print(f"   • Norm: {np.linalg.norm(embedding1):.6f} (should be ~1.0)")
        print(f"   • First 10 values: {embedding1[:10]}")
        
        # Test similarity with itself
        similarity = selfmade.compute_similarity(embedding1, embedding1)
        print(f"   • Self-similarity: {similarity:.6f} (should be 1.0)")
        
    except Exception as e:
        print(f"❌ SelfMade embedder failed: {e}")
    
    print("\n" + "=" * 70)
    print("Testing Voyage AI Embedder (Requires API Key)")
    print("=" * 70)
    
    try:
        voyage = ImageEmbedder()
        embedding2 = voyage.embed_image(test_image)
        
        print(f"✅ Embedding generated!")
        print(f"   • Dimension: {len(embedding2)}")
        print(f"   • Shape: {embedding2.shape}")
        print(f"   • Norm: {np.linalg.norm(embedding2):.6f} (should be ~1.0)")
        print(f"   • First 10 values: {embedding2[:10]}")
        
        # Test similarity with itself
        similarity = voyage.compute_similarity(embedding2, embedding2)
        print(f"   • Self-similarity: {similarity:.6f} (should be 1.0)")
        
    except Exception as e:
        print(f"❌ Voyage AI embedder failed: {e}")
        print(f"   (This is expected if VOYAGE_API_KEY is not set)")
    
    print("\n" + "=" * 70)
    print("✅ Demo Complete!")
    print("=" * 70)
    print("\n💡 To switch providers:")
    print("   1. Edit config.yaml")
    print("   2. Change 'provider' to 'voyage' or 'selfmade'")
    print("   3. Run train.py\n")


if __name__ == '__main__':
    # Run demo when script is executed directly
    demo_embedders()



