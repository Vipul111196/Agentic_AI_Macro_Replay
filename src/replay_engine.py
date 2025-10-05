"""
Replay Engine for Warmwind OS
Handles state matching and action replay with AI fallback.
"""

import numpy as np
from typing import Dict, Any
from .replay_types import ReplayDecision, ReplayResult
from .utils import print_statistics


class ReplayEngine:
    """
    Core replay engine that matches states and decides actions.
    """
    
    def __init__(
        self, 
        macro_graph,
        similarity_threshold: float = 0.90
    ):
        """
        Initialize the replay engine.
        
        Args:
            macro_graph: MacroGraph instance with learned states
            similarity_threshold: Minimum similarity to consider a match
        """
        self.graph = macro_graph
        self.similarity_threshold = similarity_threshold
        
        # Statistics
        self.stats = {
            'total_queries': 0,
            'replay_decisions': 0,
            'ai_fallback_decisions': 0,
            'no_action_decisions': 0
        }
        
        print(f"🎮 ReplayEngine initialized with threshold={similarity_threshold}")
    
    def query(
        self, 
        state_embedding: np.ndarray,
        verbose: bool = False
    ) -> ReplayResult:
        """
        Query the replay engine with a new state.
        
        Args:
            state_embedding: Embedding vector for the current state
            verbose: If True, print detailed decision process
            
        Returns:
            ReplayResult with decision and action
        """
        self.stats['total_queries'] += 1
        
        if verbose:
            print(f"\n🔍 Querying replay engine (query #{self.stats['total_queries']})...")
        
        # Find similar nodes
        similar_nodes = self.graph.find_similar_nodes(
            state_embedding,
            top_k=5,
            min_similarity=None  # Get all, we'll filter
        )
        
        if verbose and similar_nodes:
            print(f"  Found {len(similar_nodes)} similar nodes:")
            for i, (node_id, sim) in enumerate(similar_nodes[:3]):
                print(f"    {i+1}. Node {node_id}: similarity={sim:.4f}")
        
        # No similar nodes found or graph is empty
        if not similar_nodes:
            if verbose:
                print("  ❌ No similar nodes found → AI FALLBACK")
            self.stats['ai_fallback_decisions'] += 1
            return ReplayResult(
                decision=ReplayDecision.AI_FALLBACK,
                confidence=0.0,
                action_text=None,
                think_text=None,
                matched_node_id=None
            )
        
        # Check best match
        best_node_id, best_similarity = similar_nodes[0]
        best_node = self.graph.get_node(best_node_id)
        
        # Similarity below threshold → AI fallback
        if best_similarity < self.similarity_threshold:
            if verbose:
                print(f"  ⚠️  Best similarity {best_similarity:.4f} < threshold {self.similarity_threshold:.4f} → AI FALLBACK")
            self.stats['ai_fallback_decisions'] += 1
            return ReplayResult(
                decision=ReplayDecision.AI_FALLBACK,
                confidence=best_similarity,
                action_text=None,
                think_text=None,
                matched_node_id=best_node_id
            )
        
        # Good match but no action → informational only
        if not best_node.action_text:
            if verbose:
                print(f"  ℹ️  Node {best_node_id} has no action → NO ACTION")
            self.stats['no_action_decisions'] += 1
            return ReplayResult(
                decision=ReplayDecision.NO_ACTION,
                confidence=best_similarity,
                action_text=None,
                think_text=best_node.think_text,
                matched_node_id=best_node_id
            )
        
        # All checks passed → REPLAY!
        if verbose:
            print(f"  ✅ REPLAY action from node {best_node_id}")
            print(f"     Action: {best_node.action_text[:60]}...")
        
        self.stats['replay_decisions'] += 1
        return ReplayResult(
            decision=ReplayDecision.REPLAY,
            confidence=best_similarity,
            action_text=best_node.action_text,
            think_text=best_node.think_text,
            matched_node_id=best_node_id
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about replay engine performance."""
        total = self.stats['total_queries']
        
        if total == 0:
            return {**self.stats, 'replay_rate': 0.0, 'ai_fallback_rate': 0.0}
        
        return {
            **self.stats,
            'replay_rate': self.stats['replay_decisions'] / total,
            'ai_fallback_rate': self.stats['ai_fallback_decisions'] / total
        }
    
    def simulate_workflow(
        self,
        test_embeddings: np.ndarray,
        test_steps: list,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Simulate replaying a workflow and measure performance.
        
        Args:
            test_embeddings: Array of test state embeddings
            test_steps: List of ConversationStep objects
            verbose: Print detailed progress
            
        Returns:
            Dict with simulation results
        """
        if verbose:
            print("\n" + "=" * 60)
            print("Simulating Workflow Replay")
            print("=" * 60)
        
        results = []
        
        for i, (embedding, step) in enumerate(zip(test_embeddings, test_steps)):
            if verbose and i % 10 == 0:
                print(f"\n🔄 Processing step {i}/{len(test_embeddings)}...")
            
            result = self.query(embedding, verbose=(i < 3 and verbose))
            results.append({
                'step_index': i,
                'decision': result.decision.value,
                'confidence': result.confidence,
                'had_action': step.action_text is not None,
                'matched_node': result.matched_node_id
            })
        
        # Analyze results
        total_steps = len(results)
        replayed = sum(1 for r in results if r['decision'] == 'replay')
        ai_fallback = sum(1 for r in results if r['decision'] == 'ai_fallback')
        no_action = sum(1 for r in results if r['decision'] == 'no_action')
        
        steps_with_actions = sum(1 for r in results if r['had_action'])
        
        summary = {
            'total_steps': total_steps,
            'replayed': replayed,
            'ai_fallback': ai_fallback,
            'no_action': no_action,
            'steps_with_actions': steps_with_actions,
            'replay_rate': replayed / total_steps if total_steps > 0 else 0,
            'ai_fallback_rate': ai_fallback / total_steps if total_steps > 0 else 0,
            'avg_confidence': np.mean([r['confidence'] for r in results])
        }
        
        if verbose:
            print("\n" + "=" * 60)
            print_statistics(summary, "📊 Simulation Results")
            print("=" * 60)
        
        return summary


