"""
Macro Graph Engine for Warmwind OS
Builds and manages execution graph of UI states with FAISS-based similarity search.
"""

import numpy as np
import pickle
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import faiss
from .replay_types import ReplayDecision
from .utils import compare_actions


@dataclass
class GraphNode:
    """Represents a single state in the execution graph."""
    node_id: int
    conversation_id: int
    step_index: int
    embedding: np.ndarray  # Image embedding vector
    image_data: Optional[str]  # Base64 encoded image (optional to save memory)
    action_text: Optional[str]  # Action to execute
    think_text: Optional[str]  # Reasoning/extraction hints
    
    # Metadata
    visit_count: int = 0
    success_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def get_success_rate(self) -> float:
        """Calculate success rate for this node."""
        if self.visit_count == 0:
            return 0.0
        return self.success_count / self.visit_count
    
    def record_visit(self, success: bool = True):
        """Record a visit to this node."""
        self.visit_count += 1
        if success:
            self.success_count += 1


@dataclass
class GraphEdge:
    """Represents a transition between two states."""
    from_node_id: int
    to_node_id: int
    transition_count: int = 0
    
    def record_transition(self):
        """Record a transition along this edge."""
        self.transition_count += 1


class MacroGraph:
    """
    Manages the execution graph with FAISS-based similarity search.
    """
    
    def __init__(self, embedding_dim: int = 1024, similarity_threshold: float = 0.90):
        """
        Initialize the macro graph.
        
        Args:
            embedding_dim: Dimension of embedding vectors
            similarity_threshold: Threshold for considering states as "same"
        """
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        
        # Storage
        self.nodes: Dict[int, GraphNode] = {}
        self.edges: Dict[Tuple[int, int], GraphEdge] = {}
        self.next_node_id = 0
        
        # FAISS index for fast similarity search
        # Using IndexFlatIP (inner product) since embeddings are normalized
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.index_to_node_id: List[int] = []  # Maps FAISS index position to node_id
        
        print(f"📊 Initialized MacroGraph with embedding_dim={embedding_dim}, threshold={similarity_threshold}")
    
    def add_node(
        self, 
        embedding: np.ndarray,
        conversation_id: int,
        step_index: int,
        action_text: Optional[str] = None,
        think_text: Optional[str] = None,
        image_data: Optional[str] = None,
        check_similarity: bool = True
    ) -> Tuple[int, bool]:
        """
        Add a new node to the graph or return existing similar node.
        
        Args:
            embedding: Image embedding vector (must be normalized by ImageEmbedder)
            conversation_id: ID of conversation this step is from
            step_index: Step index within conversation
            action_text: Action to execute
            think_text: Reasoning text
            image_data: Base64 encoded image (optional)
            check_similarity: If True, check for similar existing nodes first
            
        Returns:
            Tuple of (node_id, is_new_node)
        """
        # Embedding should already be normalized by ImageEmbedder
        # No need to normalize again here
        
        # Check if similar node already exists
        if check_similarity and len(self.nodes) > 0:
            similar_nodes = self.find_similar_nodes(embedding, top_k=1)
            if similar_nodes and similar_nodes[0][1] >= self.similarity_threshold:
                existing_node_id = similar_nodes[0][0]
                existing_node = self.nodes[existing_node_id]
                existing_node.record_visit()
                return existing_node_id, False
        
        # Create new node
        node_id = self.next_node_id
        self.next_node_id += 1
        
        node = GraphNode(
            node_id=node_id,
            conversation_id=conversation_id,
            step_index=step_index,
            embedding=embedding,
            image_data=image_data,
            action_text=action_text,
            think_text=think_text
        )
        
        self.nodes[node_id] = node
        
        # Add to FAISS index (embeddings come pre-normalized from ImageEmbedder)
        embedding_2d = embedding.reshape(1, -1).astype(np.float32)
        self.index.add(embedding_2d)
        self.index_to_node_id.append(node_id)
        
        return node_id, True
    
    def add_edge(self, from_node_id: int, to_node_id: int):
        """
        Add or update an edge between two nodes.
        
        Args:
            from_node_id: Source node ID
            to_node_id: Destination node ID
        """
        edge_key = (from_node_id, to_node_id)
        
        if edge_key in self.edges:
            self.edges[edge_key].record_transition()
        else:
            self.edges[edge_key] = GraphEdge(from_node_id, to_node_id)
            self.edges[edge_key].record_transition()
    
    def find_similar_nodes(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 5,
        min_similarity: Optional[float] = None
    ) -> List[Tuple[int, float]]:
        """
        Find most similar nodes to a query embedding using cosine similarity.
        
        Args:
            query_embedding: Query embedding vector (must be normalized by ImageEmbedder)
            top_k: Number of top matches to return
            min_similarity: Minimum similarity threshold (optional)
            
        Returns:
            List of (node_id, similarity_score) tuples sorted by similarity (highest first)
        """
        if len(self.nodes) == 0:
            return []
        
        # Embedding should already be normalized by ImageEmbedder
        # No need to normalize again here
        
        # Prepare query for FAISS
        query_2d = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search using FAISS (returns inner product, which is cosine similarity for normalized vectors)
        k = min(top_k, len(self.nodes))
        distances, indices = self.index.search(query_2d, k)
        
        # Convert to results
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0:  # FAISS returns -1 for empty slots
                continue
            
            node_id = self.index_to_node_id[idx]
            similarity = float(dist)  # Cosine similarity (inner product of normalized vectors)
            
            if min_similarity is None or similarity >= min_similarity:
                results.append((node_id, similarity))
        
        return results
    
    def get_node(self, node_id: int) -> Optional[GraphNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def get_outgoing_edges(self, node_id: int) -> List[GraphEdge]:
        """Get all edges going out from a node."""
        return [edge for (from_id, _), edge in self.edges.items() if from_id == node_id]
    
    def get_incoming_edges(self, node_id: int) -> List[GraphEdge]:
        """Get all edges coming into a node."""
        return [edge for (_, to_id), edge in self.edges.items() if to_id == node_id]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the graph."""
        total_nodes = len(self.nodes)
        total_edges = len(self.edges)
        
        if total_nodes == 0:
            return {
                'total_nodes': 0,
                'total_edges': 0,
                'avg_visit_count': 0,
                'avg_success_rate': 0,
                'nodes_with_actions': 0
            }
        
        avg_visit_count = np.mean([n.visit_count for n in self.nodes.values()])
        success_rates = [n.get_success_rate() for n in self.nodes.values() if n.visit_count > 0]
        avg_success_rate = np.mean(success_rates) if success_rates else 0
        nodes_with_actions = sum(1 for n in self.nodes.values() if n.action_text)
        
        unique_conversations = len(set(n.conversation_id for n in self.nodes.values()))
        
        return {
            'total_nodes': total_nodes,
            'total_edges': total_edges,
            'unique_conversations': unique_conversations,
            'avg_visit_count': avg_visit_count,
            'avg_success_rate': avg_success_rate,
            'nodes_with_actions': nodes_with_actions,
            'index_size': self.index.ntotal
        }
    
    def save(self, filepath: str):
        """Save graph to disk."""
        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.faiss")
        
        # Save rest of the graph
        graph_data = {
            'nodes': self.nodes,
            'edges': self.edges,
            'index_to_node_id': self.index_to_node_id,
            'next_node_id': self.next_node_id,
            'embedding_dim': self.embedding_dim,
            'similarity_threshold': self.similarity_threshold
        }
        
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump(graph_data, f)
        
        print(f"💾 Graph saved to {filepath}")
    
    def load(self, filepath: str):
        """Load graph from disk."""
        # Load FAISS index
        self.index = faiss.read_index(f"{filepath}.faiss")
        
        # Load rest of the graph
        with open(f"{filepath}.pkl", 'rb') as f:
            graph_data = pickle.load(f)
        
        self.nodes = graph_data['nodes']
        self.edges = graph_data['edges']
        self.index_to_node_id = graph_data['index_to_node_id']
        self.next_node_id = graph_data['next_node_id']
        self.embedding_dim = graph_data['embedding_dim']
        self.similarity_threshold = graph_data['similarity_threshold']
        
        print(f"📂 Graph loaded from {filepath}")
        print(f"   Nodes: {len(self.nodes)}, Edges: {len(self.edges)}")


class GraphBuilder:
    """
    Builds graph incrementally as steps arrive (streaming mode).
    Designed to work with StreamingConversationParser.
    
    Includes replay engine integration for action comparison.
    """
    
    def __init__(self, graph: MacroGraph, embedder, replay_engine=None, link_sequential: bool = True):
        """
        Initialize graph builder.
        
        Args:
            graph: MacroGraph to add nodes to
            embedder: ImageEmbedder instance for generating embeddings
            replay_engine: Optional ReplayEngine for testing replay accuracy
            link_sequential: If True, link consecutive steps with edges
        """
        self.graph = graph
        self.embedder = embedder
        self.replay_engine = replay_engine
        self.link_sequential = link_sequential
        
        # State tracking
        self.previous_node_id = None
        self.nodes_added = 0
        self.nodes_reused = 0
        
        # Replay comparison tracking
        self.comparison_results = []
    
    def add_step(self, step) -> Tuple[int, bool]:
        """
        Add a single step to the graph.
        
        Pipeline:
        1. Generate embedding
        2. Query replay_engine (if available) → REPLAY or AI_FALLBACK?
        3. Compare action_taken vs correct_action
        4. Add node to graph
        
        Args:
            step: ConversationStep object
            
        Returns:
            Tuple of (node_id, is_new_node)
        """
        # Skip steps without images
        if not step.image_data:
            return -1, False
        
        # Generate embedding for this step
        image = step.get_pil_image()
        embedding = self.embedder.embed_image(image, image_data=step.image_data)
        
        # If replay_engine is available, query it BEFORE adding to graph
        if self.replay_engine:
            # Get correct action from dataset
            correct_action = step.action_text if step.action_text else "NO_ACTION"
            
            # Query replay engine
            result = self.replay_engine.query(embedding, verbose=False)
            
            # Determine action_taken based on decision
            if result.decision == ReplayDecision.REPLAY:
                action_taken = result.action_text if result.action_text else "NO_ACTION"
            elif result.decision == ReplayDecision.AI_FALLBACK:
                action_taken = correct_action  # Use dataset action
            else:  # NO_ACTION
                action_taken = "NO_ACTION"
            
            # Compare actions with fuzzy matching for click coordinates
            is_correct = compare_actions(action_taken, correct_action, click_tolerance=10)
            
            # Store comparison result (including think_text for visualization)
            self.comparison_results.append({
                'step_index': step.step_index,
                'conversation_id': step.conversation_id,
                'decision': result.decision.value,
                'action_taken': action_taken,
                'correct_action': correct_action,
                'confidence': result.confidence,
                'matched_node_id': result.matched_node_id,
                'is_correct': is_correct,
                'think_text': step.think_text  # Add thinking block
            })
        
        # Add node to graph (with similarity check)
        node_id, is_new = self.graph.add_node(
            embedding=embedding,
            conversation_id=step.conversation_id,
            step_index=step.step_index,
            action_text=step.action_text,
            think_text=step.think_text,
            image_data=None,  # Don't store images to save memory
            check_similarity=True
        )
        
        # Track statistics
        if is_new:
            self.nodes_added += 1
        else:
            self.nodes_reused += 1
        
        # Link to previous step if sequential
        if self.link_sequential and self.previous_node_id is not None and self.previous_node_id != -1:
            self.graph.add_edge(self.previous_node_id, node_id)
        
        self.previous_node_id = node_id
        return node_id, is_new
    
    def reset_sequence(self):
        """Reset the sequential linking (e.g., when starting a new conversation)."""
        self.previous_node_id = None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about nodes added."""
        return {
            'nodes_added': self.nodes_added,
            'nodes_reused': self.nodes_reused,
            'total_processed': self.nodes_added + self.nodes_reused
        }
    
    def get_accuracy_stats(self) -> Dict[str, Any]:
        """Get replay accuracy statistics from comparison results."""
        if not self.comparison_results:
            return {}
        
        total = len(self.comparison_results)
        correct = sum(1 for r in self.comparison_results if r['is_correct'])
        
        replay_results = [r for r in self.comparison_results if r['decision'] == 'replay']
        ai_fallback_results = [r for r in self.comparison_results if r['decision'] == 'ai_fallback']
        
        replay_correct = sum(1 for r in replay_results if r['is_correct'])
        
        return {
            'total_comparisons': total,
            'overall_accuracy': correct / total if total > 0 else 0,
            'correct_predictions': correct,
            'wrong_predictions': total - correct,
            'replay_count': len(replay_results),
            'replay_correct': replay_correct,
            'replay_accuracy': replay_correct / len(replay_results) if replay_results else 0,
            'ai_fallback_count': len(ai_fallback_results),
        }
    
    def save_report(self, filepath: str, graph_stats: Dict[str, Any], engine_stats: Dict[str, Any]):
        """
        Save execution report with all statistics and comparison results.
        
        Args:
            filepath: Path to save report JSON
            graph_stats: Graph statistics
            engine_stats: Replay engine statistics
        """
        import json
        from datetime import datetime
        
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'num_conversations': len(set(r['conversation_id'] for r in self.comparison_results)),
                'total_steps_processed': len(self.comparison_results),
            },
            'builder_stats': self.get_statistics(),
            'graph_stats': graph_stats,
            'engine_stats': engine_stats,
            'accuracy_stats': self.get_accuracy_stats(),
            'comparison_results': self.comparison_results,
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Report saved to {filepath}")
    
    def save_embeddings(self, filepath: str):
        """
        Save all node embeddings for visualization.
        
        Args:
            filepath: Path to save embeddings (numpy format)
        """
        import numpy as np
        
        node_ids = list(self.graph.nodes.keys())
        embeddings = np.array([self.graph.nodes[nid].embedding for nid in node_ids])
        
        # Save embeddings and node IDs
        np.savez(filepath, 
                 embeddings=embeddings,
                 node_ids=node_ids)
        
        print(f"🎨 Embeddings saved to {filepath}")

