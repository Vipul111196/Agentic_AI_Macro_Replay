"""
Test Script for Trained Graph
==============================

This script tests a trained graph on validation/test data:
1. Load trained graph from disk
2. Process validation conversations
3. Measure replay accuracy WITHOUT modifying the graph
4. Generate test report

Run: python test.py
"""

import json
from datetime import datetime
from typing import List, Dict, Any

from data_parser import StreamingConversationParser
from image_embedder import ImageEmbedder
from macro_graph import MacroGraph
from replay_engine import ReplayEngine
from replay_types import ReplayDecision
from utils import print_statistics, truncate_text


class GraphTester:
    """
    Tests a trained graph on new conversations without modifying it.
    """

    def __init__(self, graph: MacroGraph, embedder, replay_engine: ReplayEngine):
        """
        Initialize graph tester.

        Args:
            graph: Trained MacroGraph (loaded from disk)
            embedder: ImageEmbedder instance
            replay_engine: ReplayEngine instance
        """
        self.graph = graph
        self.embedder = embedder
        self.replay_engine = replay_engine

        # Test results tracking
        self.test_results = []
        self.steps_processed = 0
        self.conversations_processed = 0

        # Test embeddings tracking
        self.test_embeddings = []
        self.test_step_indices = []
    
    def test_conversation(self, conversation_id: int, messages: List[Dict]) -> List[Dict]:
        """
        Test a single conversation against the trained graph.
        
        Args:
            conversation_id: ID for this conversation
            messages: List of message dictionaries
            
        Returns:
            List of test result dictionaries for each step
        """
        parser = StreamingConversationParser()
        parser.reset(conversation_id=conversation_id)
        
        # Parse all messages
        steps = []
        for message in messages:
            step = parser.add_message(message)
            if step:
                steps.append(step)
        
        # Test each step with images
        conversation_results = []
        for step in steps:
            if not step.image_data:
                continue
            
            # Generate embedding
            image = step.get_pil_image()
            embedding = self.embedder.embed_image(image, image_data=step.image_data)

            # Store test embedding for visualization
            self.test_embeddings.append(embedding)
            self.test_step_indices.append(step.step_index)

            # Get correct action from dataset
            correct_action = step.action_text if step.action_text else "NO_ACTION"
            
            # Query replay engine (WITHOUT adding to graph)
            result = self.replay_engine.query(embedding, verbose=False)
            
            # Determine predicted action
            if result.decision == ReplayDecision.REPLAY:
                predicted_action = result.action_text if result.action_text else "NO_ACTION"
            elif result.decision == ReplayDecision.AI_FALLBACK:
                predicted_action = "AI_FALLBACK"  # Mark as needing AI
            else:  # NO_ACTION
                predicted_action = "NO_ACTION"
            
            # Check if prediction is correct
            is_correct = (predicted_action == correct_action)
            
            # For AI_FALLBACK, it's not wrong, just needs human/AI assistance
            is_ai_fallback = (result.decision == ReplayDecision.AI_FALLBACK)
            
            # Store result (including think_text for visualization)
            test_result = {
                'conversation_id': conversation_id,
                'step_index': step.step_index,
                'decision': result.decision.value,
                'predicted_action': predicted_action,
                'correct_action': correct_action,
                'confidence': result.confidence,
                'matched_node_id': result.matched_node_id,
                'is_correct': is_correct,
                'is_ai_fallback': is_ai_fallback,
                'think_text': step.think_text  # Add thinking block
            }
            
            conversation_results.append(test_result)
            self.test_results.append(test_result)
            self.steps_processed += 1
        
        self.conversations_processed += 1
        return conversation_results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Calculate test statistics."""
        if not self.test_results:
            return {}
        
        total = len(self.test_results)
        
        # Decision breakdown
        replay_results = [r for r in self.test_results if r['decision'] == 'replay']
        ai_fallback_results = [r for r in self.test_results if r['decision'] == 'ai_fallback']
        no_action_results = [r for r in self.test_results if r['decision'] == 'no_action']
        
        # Accuracy metrics
        correct_predictions = sum(1 for r in self.test_results if r['is_correct'])
        
        # Replay-only accuracy (excluding AI fallback)
        replay_correct = sum(1 for r in replay_results if r['is_correct'])
        
        # Confidence stats
        confidences = [r['confidence'] for r in self.test_results if r['confidence'] is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            'total_steps': total,
            'conversations_processed': self.conversations_processed,
            'correct_predictions': correct_predictions,
            'wrong_predictions': total - correct_predictions,
            'overall_accuracy': correct_predictions / total if total > 0 else 0,
            
            # Decision breakdown
            'replay_count': len(replay_results),
            'replay_correct': replay_correct,
            'replay_accuracy': replay_correct / len(replay_results) if replay_results else 0,
            
            'ai_fallback_count': len(ai_fallback_results),
            'ai_fallback_rate': len(ai_fallback_results) / total if total > 0 else 0,
            
            'no_action_count': len(no_action_results),
            
            # Confidence
            'avg_confidence': avg_confidence,
        }
    
    def save_report(self, filepath: str, test_stats: Dict[str, Any], graph_stats: Dict[str, Any]):
        """
        Save test report with all statistics and results.

        Args:
            filepath: Path to save report JSON
            test_stats: Test statistics
            graph_stats: Graph statistics
        """
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'mode': 'TEST',
                'num_conversations': self.conversations_processed,
                'total_steps_processed': self.steps_processed,
            },
            'test_stats': test_stats,
            'graph_stats': graph_stats,
            'test_results': self.test_results,
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"📄 Test report saved to {filepath}")

    def save_test_embeddings(self, filepath: str):
        """
        Save test embeddings for visualization.

        Args:
            filepath: Path to save embeddings (numpy format)
        """
        import numpy as np

        if not self.test_embeddings:
            print("⚠️  No test embeddings to save")
            return

        test_embeddings = np.array(self.test_embeddings)
        test_step_indices = np.array(self.test_step_indices)

        # Save test embeddings
        np.savez(filepath,
                 test_embeddings=test_embeddings,
                 test_step_indices=test_step_indices)

        print(f"🎨 Test embeddings saved to {filepath} ({len(test_embeddings)} embeddings)")


def main():
    """Main test pipeline."""
    print("=" * 70)
    print("🧪 AUTONOMOUS MACRO REPLAY SYSTEM - TESTING")
    print("=" * 70)
    
    # Configuration
    TEST_FILES = ['data_validation_split/maf_validate.json']
    MAX_CONVERSATIONS = 5  # Set to None for all conversations
    TRAINED_GRAPH_PATH = "models/trained_graph"  # Load from models/ directory
    OUTPUT_PREFIX = "test_results/test"  # Save to test_results/ directory
    
    print(f"\n⚙️  Configuration:")
    print(f"  • Test files: {TEST_FILES}")
    print(f"  • Max conversations: {MAX_CONVERSATIONS if MAX_CONVERSATIONS else 'ALL'}")
    print(f"  • Trained graph: {TRAINED_GRAPH_PATH}")
    
    # ========================================================================
    # STEP 1: Load Trained Graph
    # ========================================================================
    print("\n🔧 Step 1: Loading trained graph...")
    
    # Initialize embedder (same as training)
    embedder = ImageEmbedder()
    
    # Create empty graph with same config
    graph = MacroGraph(embedding_dim=1024, similarity_threshold=0.95)
    
    # Load trained graph from disk
    try:
        graph.load(TRAINED_GRAPH_PATH)
        print(f"✅ Loaded trained graph with {len(graph.nodes)} nodes")
    except Exception as e:
        print(f"❌ Error loading trained graph: {e}")
        print("   Please run train.py first to create a trained graph!")
        return
    
    # Initialize replay engine with loaded graph
    replay_engine = ReplayEngine(graph, similarity_threshold=0.90)
    
    # Initialize tester
    tester = GraphTester(graph, embedder, replay_engine)
    print("✅ All components initialized")
    
    # ========================================================================
    # STEP 2: Test on Validation Data
    # ========================================================================
    print("\n🧪 Step 2: Testing on validation data...")
    
    for test_file in TEST_FILES:
        print(f"\n📂 Loading {test_file}...")
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                conversations = json.load(f)
        except FileNotFoundError:
            print(f"❌ Test file not found: {test_file}")
            print("   Please ensure validation data exists!")
            continue
        
        # Limit conversations if specified
        conversations_to_test = conversations[:MAX_CONVERSATIONS] if MAX_CONVERSATIONS else conversations
        print(f"  Testing {len(conversations_to_test)} conversations...")
        
        for conv_idx, conv_obj in enumerate(conversations_to_test):
            messages = conv_obj.get('conversation', [])
            
            # Test this conversation
            results = tester.test_conversation(conv_idx, messages)
            
            # Show progress
            if conv_idx % 5 == 0 and results:
                print(f"    Conversation {conv_idx + 1}/{len(conversations_to_test)}: {len(results)} steps tested")
                
                # Show first conversation details
                if conv_idx == 0:
                    print(f"      Sample results:")
                    for result in results[:5]:  # Show first 5 steps
                        decision_emoji = "🔄" if result['decision'] == 'replay' else "🤖" if result['decision'] == 'ai_fallback' else "ℹ️"
                        correctness = "✅" if result['is_correct'] else "❌" if not result['is_ai_fallback'] else "⚠️"
                        
                        pred = truncate_text(result['predicted_action'], 35)
                        correct = truncate_text(result['correct_action'], 35)
                        
                        print(f"        Step {result['step_index']}: {decision_emoji} {correctness} | Pred: {pred} | True: {correct}")
    
    print(f"\n✅ Tested {tester.conversations_processed} conversations with {tester.steps_processed} steps")
    
    # ========================================================================
    # STEP 3: Display Test Statistics
    # ========================================================================
    print("\n" + "=" * 70)
    print("📊 TEST RESULTS")
    print("=" * 70)
    
    # Test statistics
    test_stats = tester.get_statistics()
    if test_stats:
        print("\n🧪 Test Statistics:")
        print(f"  • Total steps tested: {test_stats['total_steps']}")
        print(f"  • Conversations tested: {test_stats['conversations_processed']}")
        print(f"  • Overall accuracy: {test_stats['overall_accuracy']:.1%}")
        print(f"  • Correct predictions: {test_stats['correct_predictions']}/{test_stats['total_steps']}")
        print(f"  • Wrong predictions: {test_stats['wrong_predictions']}")
        
        print(f"\n  🔄 REPLAY Performance:")
        print(f"  • REPLAY count: {test_stats['replay_count']}")
        print(f"  • REPLAY correct: {test_stats['replay_correct']}")
        print(f"  • REPLAY accuracy: {test_stats['replay_accuracy']:.1%}")
        
        print(f"\n  🤖 AI_FALLBACK:")
        print(f"  • AI_FALLBACK count: {test_stats['ai_fallback_count']}")
        print(f"  • AI_FALLBACK rate: {test_stats['ai_fallback_rate']:.1%}")
        print(f"  • (These cases need AI/human assistance)")
        
        print(f"\n  📈 Confidence:")
        print(f"  • Average confidence: {test_stats['avg_confidence']:.4f}")
    
    # Graph statistics (should be unchanged)
    print("\n🌐 Graph Statistics (from trained graph):")
    graph_stats = graph.get_statistics()
    print_statistics(graph_stats, "")
    
    # ========================================================================
    # STEP 4: Save Test Report
    # ========================================================================
    print("\n💾 Step 3: Saving test results...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"test_results/test_report_{timestamp}.json"
    tester.save_report(report_file, test_stats, graph_stats)
    
    # Save latest test report
    tester.save_report("test_results/test_report_latest.json", test_stats, graph_stats)

    # Save test embeddings for visualization
    embeddings_file = f"embeddings/test_embeddings.npz"
    tester.save_test_embeddings(embeddings_file)

    print("\n" + "=" * 70)
    print("✅ TESTING COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Output Files:")
    print(f"  • Test Report: {report_file}")
    print(f"  • Latest: test_results/test_report_latest.json")
    print(f"  • Test Embeddings: {embeddings_file}")
    
    # Summary
    if test_stats:
        print(f"\n📊 Summary:")
        print(f"  • Replay Rate: {test_stats['replay_count']}/{test_stats['total_steps']} ({test_stats['replay_count']/test_stats['total_steps']:.1%})")
        print(f"  • Replay Accuracy: {test_stats['replay_accuracy']:.1%}")
        print(f"  • AI Fallback Rate: {test_stats['ai_fallback_rate']:.1%}")
    
    print("=" * 70)


if __name__ == '__main__':
    main()

