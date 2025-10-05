"""
Main Execution Script
====================

This script runs the complete training pipeline:
1. Parse conversations from dataset
2. Build graph with replay engine
3. Save graph, embeddings, and report
4. Display summary statistics

Run: python train.py
Then: streamlit run app_dashboard.py (to view results)
"""

import json
import yaml
from datetime import datetime
from pathlib import Path

from src.data_parser import StreamingConversationParser
from src.image_embedder import create_embedder, get_embedding_dim
from src.macro_graph import MacroGraph, GraphBuilder
from src.replay_engine import ReplayEngine
from src.utils import print_statistics, truncate_text


def main():
    """Main execution pipeline."""
    print("=" * 70)
    print("🎮 AUTONOMOUS MACRO REPLAY SYSTEM - TRAINING")
    print("=" * 70)
    
    # ========================================================================
    # Load Configuration
    # ========================================================================
    print("\n📋 Loading configuration from config/config.yaml...")
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract configuration values
    embedding_provider = config.get('embedding', {}).get('provider', 'voyage')
    embedding_dim = get_embedding_dim(embedding_provider)
    
    # Data configuration
    TRAIN_FILES = config.get('data', {}).get('training_files', ['data_validation_split/maf_train.json'])
    MAX_CONVERSATIONS = config.get('data', {}).get('max_conversations_train', None)
    
    # Matching configuration
    GRAPH_SIMILARITY = config.get('matching', {}).get('similarity_threshold', 0.90)
    REPLAY_SIMILARITY = config.get('matching', {}).get('similarity_threshold', 0.90)
    
    # Output paths
    OUTPUT_PREFIX = config.get('output', {}).get('trained_graph_path', 'models/trained_graph')
    EMBEDDINGS_PATH = config.get('output', {}).get('trained_embeddings_path', 'embeddings/trained_graph_embeddings.npz')
    REPORT_DIR = config.get('output', {}).get('training_report_dir', 'reports')
    REPORT_LATEST = config.get('output', {}).get('training_report_latest', 'reports/report_latest.json')
    
    # Cache paths
    if embedding_provider == "voyage":
        CACHE_FILE = config.get('output', {}).get('voyage_cache', 'cache/embedding_cache.pkl')
    else:
        CACHE_FILE = config.get('output', {}).get('selfmade_cache', 'cache/selfmade_embedding_cache.pkl')
    
    print(f"\n⚙️  Configuration:")
    print(f"  • Embedding provider: {embedding_provider.upper()}")
    print(f"  • Embedding dimension: {embedding_dim}")
    print(f"  • Training files: {TRAIN_FILES}")
    print(f"  • Max conversations: {MAX_CONVERSATIONS if MAX_CONVERSATIONS else 'ALL'}")
    print(f"  • Graph similarity: {GRAPH_SIMILARITY}")
    print(f"  • Replay similarity: {REPLAY_SIMILARITY}")
    print(f"  • Output path: {OUTPUT_PREFIX}")
    print(f"  • Cache file: {CACHE_FILE}")
    
    # ========================================================================
    # STEP 1: Initialize Components
    # ========================================================================
    print("\n🔧 Step 1: Initializing components...")
    parser = StreamingConversationParser()
    embedder = create_embedder(provider=embedding_provider, cache_file=CACHE_FILE)
    graph = MacroGraph(embedding_dim=embedding_dim, similarity_threshold=GRAPH_SIMILARITY)
    replay_engine = ReplayEngine(graph, similarity_threshold=REPLAY_SIMILARITY)
    builder = GraphBuilder(graph, embedder, replay_engine=replay_engine, link_sequential=True)
    print("✅ All components initialized")
    
    # ========================================================================
    # STEP 2: Process Conversations
    # ========================================================================
    print("\n🌊 Step 2: Processing conversations...")
    
    total_conversations = 0
    for train_file in TRAIN_FILES:
        print(f"\n📂 Loading {train_file}...")
        with open(train_file, 'r', encoding='utf-8') as f:
            conversations = json.load(f)
        
        # Limit conversations if specified
        conversations_to_process = conversations[:MAX_CONVERSATIONS] if MAX_CONVERSATIONS else conversations
        total_conversations += len(conversations_to_process)
        print(f"  Processing {len(conversations_to_process)} conversations...")
        
        for conv_idx, conv_obj in enumerate(conversations_to_process):
            parser.reset(conversation_id=conv_idx)
            builder.reset_sequence()
            
            messages = conv_obj.get('conversation', [])
            
            # Parse all messages first (so actions get populated from n+1 messages)
            steps = []
            for message in messages:
                step = parser.add_message(message)
                if step:
                    steps.append(step)
            
            # Process complete steps through replay engine and graph
            steps_with_images = [s for s in steps if s.image_data]
            
            # Show progress every 10 conversations
            if conv_idx % 10 == 0:
                print(f"    Conversation {conv_idx}/{len(conversations_to_process)}: {len(steps)} steps ({len(steps_with_images)} with images)")
            
            for step in steps:
                node_id, is_new = builder.add_step(step)
                
                # Optionally show detailed progress for first conversation
                if conv_idx == 0 and node_id != -1 and builder.comparison_results:
                    result = builder.comparison_results[-1]
                    decision = result['decision']
                    
                    decision_emoji = "🤖" if decision == 'ai_fallback' else "🔄" if decision == 'replay' else "ℹ️"
                    correctness = "✅" if result['is_correct'] else "❌"
                    status = "NEW" if is_new else "REUSED"
                    
                    taken = truncate_text(result['action_taken'], 40)
                    correct = truncate_text(result['correct_action'], 40)
                    
                    print(f"      Step {step.step_index}: Node {node_id} ({status}) {decision_emoji} | ({taken}, {correct}) {correctness}")
    
    print(f"\n✅ Processed {total_conversations} conversations")
    
    # ========================================================================
    # STEP 3: Display Statistics
    # ========================================================================
    print("\n" + "=" * 70)
    print("📊 TRAINING RESULTS")
    print("=" * 70)
    
    # Builder statistics
    builder_stats = builder.get_statistics()
    print_statistics(builder_stats, "📦 Builder Statistics")
    
    # Graph statistics
    graph_stats = graph.get_statistics()
    print_statistics(graph_stats, "🌐 Graph Statistics")
    
    # Replay engine statistics
    engine_stats = replay_engine.get_statistics()
    print_statistics(engine_stats, "🎮 Replay Engine Statistics")
    
    # Accuracy statistics
    print("\n🎯 Replay Accuracy Statistics:")
    accuracy_stats = builder.get_accuracy_stats()
    if accuracy_stats:
        print(f"  • Total comparisons: {accuracy_stats['total_comparisons']}")
        print(f"  • Overall accuracy: {accuracy_stats['overall_accuracy']:.1%}")
        print(f"  • Correct predictions: {accuracy_stats['correct_predictions']}/{accuracy_stats['total_comparisons']}")
        print(f"\n  REPLAY Performance:")
        print(f"  • REPLAY count: {accuracy_stats['replay_count']}")
        print(f"  • REPLAY correct: {accuracy_stats['replay_correct']}")
        if accuracy_stats['replay_count'] > 0:
            print(f"  • REPLAY accuracy: {accuracy_stats['replay_accuracy']:.1%}")
        print(f"\n  AI_FALLBACK:")
        print(f"  • AI_FALLBACK count: {accuracy_stats['ai_fallback_count']}")
        print(f"  • (Always 100% since we use dataset action)")
    
    # ========================================================================
    # STEP 4: Save Everything
    # ========================================================================
    print("\n💾 Step 3: Saving results...")
    
    # Note: Embedding cache is saved incrementally during processing
    
    # Save graph
    graph.save(OUTPUT_PREFIX)
    print(f"✅ Graph saved: {OUTPUT_PREFIX}.faiss + {OUTPUT_PREFIX}.pkl")
    
    # Save embeddings
    builder.save_embeddings(EMBEDDINGS_PATH)
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"{REPORT_DIR}/report_{timestamp}.json"
    builder.save_report(report_file, graph_stats, engine_stats)
    
    # Save latest report (for dashboard to load easily)
    builder.save_report(REPORT_LATEST, graph_stats, engine_stats)
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Output Files:")
    print(f"  • Graph: {OUTPUT_PREFIX}.pkl, {OUTPUT_PREFIX}.faiss")
    print(f"  • Embeddings: {EMBEDDINGS_PATH}")
    print(f"  • Report: {report_file}")
    print(f"  • Latest: {REPORT_LATEST}")
    print(f"\n🎨 Next Step:")
    print(f"  Run: streamlit run app_dashboard.py")
    print("=" * 70)


if __name__ == '__main__':
    main()

