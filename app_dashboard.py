"""
🎮 Autonomous Macro Replay System - Dashboard
Beautiful UI for visualizing training results
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
import os
import base64
from io import BytesIO
from pathlib import Path
from PIL import Image

from src.macro_graph import MacroGraph
from src.utils import truncate_text, format_key_label, format_stat_value

# Page config
st.set_page_config(
    page_title="Macro Replay Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .big-metric {
        font-size: 48px !important;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 14px;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_report(report_path="output/reports/report_latest.json"):
    """Load the execution report."""
    if not os.path.exists(report_path):
        return None
    with open(report_path, 'r', encoding='utf-8') as f:
        return json.load(f)


@st.cache_data
def load_conversation_data():
    """Load all conversation JSON files for screenshot access."""
    # This matches EXACTLY how train.py loads data
    conversations = {}
    conv_index = 0
    
    # Load ONLY the files that train.py uses
    train_files = ['data/data_validation_split_hannes/maf_train.json', 'data/data_validation_split_hannes/sona_train.json']
    
    for file_path in train_files:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Each conversation object gets an index
                for conv_obj in data:
                    conversations[conv_index] = conv_obj
                    conv_index += 1
    
    return conversations


@st.cache_resource
def load_graph(graph_path="output/models/trained_graph"):
    """Load the trained graph."""
    if not os.path.exists(f"{graph_path}.faiss"):
        return None
    graph = MacroGraph(embedding_dim=1024)
    graph.load(graph_path)
    return graph


@st.cache_data
def load_embeddings(embeddings_path="output/embeddings/trained_graph_embeddings.npz"):
    """Load training embeddings for visualization."""
    if not os.path.exists(embeddings_path):
        return None, None
    data = np.load(embeddings_path)
    return data['embeddings'], data['node_ids']


@st.cache_data(ttl=60)  # Cache for 60 seconds to allow updates
def load_test_embeddings(embeddings_path="output/embeddings/test_embeddings.npz"):
    """Load test embeddings for visualization."""
    if not os.path.exists(embeddings_path):
        return None, None
    try:
        data = np.load(embeddings_path)
        return data['test_embeddings'], data['test_step_indices']
    except Exception as e:
        st.warning(f"Error loading test embeddings: {e}")
        return None, None


@st.cache_data
def load_test_report(report_path="output/test_results/test_report_latest.json"):
    """Load test report for predictions."""
    if not os.path.exists(report_path):
        return None
    with open(report_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    st.title("🎮 Autonomous Macro Replay System")
    st.markdown("### Training Results Dashboard")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Select report
        report_files = list(Path("output/reports").glob("report_*.json")) if Path("output/reports").exists() else []
        if report_files:
            report_options = ["output/reports/report_latest.json"] + [f"output/reports/{f.name}" for f in sorted(report_files, reverse=True) if f.name != "report_latest.json"]
            selected_report = st.selectbox("Select Report", report_options)
        else:
            selected_report = "output/reports/report_latest.json"
        
        st.markdown("---")
        st.markdown("### 📚 Navigation")
        page = st.radio(
            "Go to:",
            ["📊 Overview", "🎯 Accuracy Analysis", "🌐 Graph Visualization", "📋 Comparison Details"]
        )
    
    # Load data
    report = load_report(selected_report)
    
    if report is None:
        st.error("❌ No training report found!")
        st.info("💡 Run `python main.py` first to train the system!")
        st.code("python main.py", language="bash")
        return
    
    graph = load_graph("output/models/trained_graph")
    embeddings, node_ids = load_embeddings("output/embeddings/trained_graph_embeddings.npz")

    # Try to load test embeddings and report (may not exist)
    test_embeddings, test_step_indices = load_test_embeddings("output/embeddings/test_embeddings.npz")
    test_report = load_test_report("output/test_results/test_report_latest.json")
    
    # Route to pages
    if page == "📊 Overview":
        show_overview(report, graph)
    elif page == "🎯 Accuracy Analysis":
        show_accuracy_analysis(report)
    elif page == "🌐 Graph Visualization":
        show_graph_viz(graph, embeddings, node_ids, test_embeddings, test_step_indices, report, test_report)
    elif page == "📋 Comparison Details":
        show_comparison_details(report)


def show_overview(report, graph):
    """Show overview dashboard."""
    st.header("📊 Training Overview")
    
    metadata = report['metadata']
    builder_stats = report['builder_stats']
    graph_stats = report['graph_stats']
    engine_stats = report['engine_stats']
    accuracy_stats = report['accuracy_stats']
    
    # Metadata
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📅 Training Date", metadata['timestamp'].split('T')[0])
    with col2:
        st.metric("💬 Conversations", metadata['num_conversations'])
    with col3:
        st.metric("📝 Steps Processed", metadata['total_steps_processed'])
    
    st.markdown("---")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📦 Total Nodes",
            f"{graph_stats['total_nodes']:,}",
            help="Unique UI states in the graph"
        )
    
    with col2:
        st.metric(
            "🔗 Total Edges",
            f"{graph_stats['total_edges']:,}",
            help="Transitions between states"
        )
    
    with col3:
        replay_rate = engine_stats['replay_rate'] * 100
        st.metric(
            "🔄 Replay Rate",
            f"{replay_rate:.1f}%",
            help="Percentage of steps that were replayed"
        )
    
    with col4:
        if accuracy_stats.get('replay_count', 0) > 0:
            replay_acc = accuracy_stats['replay_accuracy'] * 100
            st.metric(
                "🎯 Replay Accuracy",
                f"{replay_acc:.1f}%",
                help="How often replayed actions were correct"
            )
        else:
            st.metric("🎯 Replay Accuracy", "N/A", help="No replays yet")
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎮 Decision Breakdown")
        
        decision_data = pd.DataFrame({
            'Decision': ['🔄 Replay', '🤖 AI Fallback', 'ℹ️ No Action'],
            'Count': [
                engine_stats['replay_decisions'],
                engine_stats['ai_fallback_decisions'],
                engine_stats['no_action_decisions']
            ]
        })
        
        fig = px.pie(
            decision_data,
            values='Count',
            names='Decision',
            color='Decision',
            color_discrete_map={
                '🔄 Replay': '#28a745',
                '🤖 AI Fallback': '#ffc107',
                'ℹ️ No Action': '#6c757d'
            },
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(showlegend=True, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Graph Growth")
        
        metrics_data = pd.DataFrame({
            'Metric': ['Nodes Added', 'Nodes Reused', 'Nodes with Actions'],
            'Count': [
                builder_stats['nodes_added'],
                builder_stats['nodes_reused'],
                graph_stats['nodes_with_actions']
            ]
        })
        
        fig = px.bar(
            metrics_data,
            x='Metric',
            y='Count',
            color='Metric',
            text='Count'
        )
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Top Visited Nodes
    st.markdown("---")
    st.subheader("🔥 Top Visited Nodes (Most Reused)")
    
    if graph:
        # Get top 10 most visited nodes
        top_nodes = sorted(graph.nodes.values(), key=lambda n: n.visit_count, reverse=True)[:10]
        
        if top_nodes:
            node_data = []
            for node in top_nodes:
                node_data.append({
                    'Node ID': node.node_id,
                    'Visits': node.visit_count,
                    'Success Rate': f"{node.get_success_rate():.1%}",
                    'Action': truncate_text(node.action_text, 50) if node.action_text else 'None'
                })
            
            df = pd.DataFrame(node_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Visualization of visit distribution
            st.subheader("📊 Visit Count Distribution")
            all_visits = [n.visit_count for n in graph.nodes.values()]
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=all_visits,
                nbinsx=20,
                marker_color='#1f77b4',
                name='Nodes'
            ))
            fig.update_layout(
                xaxis_title="Visit Count",
                yaxis_title="Number of Nodes",
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No nodes in graph yet")
    
    # Statistics Tables
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📦 Builder Stats")
        for key, value in builder_stats.items():
            st.write(f"**{format_key_label(key)}:** {format_stat_value(key, value)}")
    
    with col2:
        st.subheader("🌐 Graph Stats")
        for key, value in graph_stats.items():
            st.write(f"**{format_key_label(key)}:** {format_stat_value(key, value)}")
    
    with col3:
        st.subheader("🎮 Engine Stats")
        for key, value in engine_stats.items():
            st.write(f"**{format_key_label(key)}:** {format_stat_value(key, value)}")


def show_accuracy_analysis(report):
    """Show detailed accuracy analysis."""
    st.header("🎯 Replay Accuracy Analysis")
    
    accuracy_stats = report['accuracy_stats']
    comparisons = report['comparison_results']
    
    if not comparisons:
        st.warning("No comparison data available")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Comparisons", accuracy_stats['total_comparisons'])
    
    with col2:
        st.metric("Overall Accuracy", f"{accuracy_stats['overall_accuracy']:.1%}")
    
    with col3:
        st.metric("✅ Correct", accuracy_stats['correct_predictions'])
    
    with col4:
        st.metric("❌ Wrong", accuracy_stats['wrong_predictions'])
    
    st.markdown("---")
    
    # Replay vs AI Fallback
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔄 REPLAY Performance")
        st.metric("Replay Count", accuracy_stats['replay_count'])
        st.metric("Replay Correct", accuracy_stats['replay_correct'])
        if accuracy_stats['replay_count'] > 0:
            st.metric("Replay Accuracy", f"{accuracy_stats['replay_accuracy']:.1%}")
        else:
            st.info("No replay decisions yet")
    
    with col2:
        st.subheader("🤖 AI_FALLBACK Performance")
        st.metric("AI Fallback Count", accuracy_stats['ai_fallback_count'])
        st.info("AI_FALLBACK always uses dataset action (100% accuracy)")
    
    st.markdown("---")
    
    # Accuracy over time
    st.subheader("📈 Accuracy Trend Over Time")
    
    df = pd.DataFrame(comparisons)
    df['step_number'] = range(len(df))
    
    # Calculate rolling accuracy
    window = 50
    df['rolling_correct'] = df['is_correct'].rolling(window=window, min_periods=1).mean()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['step_number'],
        y=df['rolling_correct'] * 100,
        mode='lines',
        name=f'Rolling Accuracy (window={window})',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.update_layout(
        xaxis_title="Step Number",
        yaxis_title="Accuracy (%)",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Decision distribution by correctness
    st.markdown("---")
    st.subheader("📊 Correctness by Decision Type")
    
    decision_correctness = df.groupby(['decision', 'is_correct']).size().reset_index(name='count')
    
    fig = px.bar(
        decision_correctness,
        x='decision',
        y='count',
        color='is_correct',
        barmode='group',
        labels={'is_correct': 'Correct', 'decision': 'Decision Type', 'count': 'Count'},
        color_discrete_map={True: '#28a745', False: '#dc3545'}
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def show_graph_viz(graph, embeddings, node_ids, test_embeddings=None, test_step_indices=None, train_report=None, test_report=None):
    """Show 2D graph visualization with training and test data including predictions."""
    st.header("🌐 Graph Visualization")
    st.markdown("Nodes positioned by **semantic similarity** from embeddings. Select a step to see prediction details!")
    
    if graph is None:
        st.error("Graph not loaded!")
        return
    
    if embeddings is None:
        st.warning("Training embeddings not found. Run training with embedding save enabled.")
        return
    
    # Check if test embeddings are available
    if test_embeddings is not None and test_step_indices is not None and len(test_embeddings) > 0:
        st.success(f"✅ Loaded {test_embeddings.shape[0]} test embeddings!")
        st.info(f"🎨 Combining {embeddings.shape[0]} training + {test_embeddings.shape[0]} test embeddings...")
        st.info(f"🔄 Reducing {embeddings.shape[1]}D embeddings to 2D using MDS...")

        # Combine all embeddings for joint MDS projection
        all_embeddings = np.vstack([embeddings, test_embeddings])
        is_training = [True] * len(embeddings) + [False] * len(test_embeddings)
        
        # Reduce to 2D
        from sklearn.manifold import MDS
        with st.spinner("Computing MDS projection for combined data..."):
            reducer = MDS(n_components=2, random_state=42, dissimilarity='euclidean', n_init=1)
            pos_2d = reducer.fit_transform(all_embeddings)
        
        # Split back into training and test positions
        train_pos_2d = pos_2d[:len(embeddings)]
        test_pos_2d = pos_2d[len(embeddings):]
        
        has_test_data = True
    else:
        st.info(f"🎨 Reducing {embeddings.shape[1]}D embeddings to 2D using MDS...")
        st.warning("⚠️ Test embeddings not found. Run testing to generate test embeddings.")
        
        # Reduce to 2D (training only)
        from sklearn.manifold import MDS
        with st.spinner("Computing MDS projection..."):
            reducer = MDS(n_components=2, random_state=42, dissimilarity='euclidean', n_init=1)
            train_pos_2d = reducer.fit_transform(embeddings)
        
        has_test_data = False
    
    # Settings for visualization (BEFORE creating DataFrame)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if has_test_data:
            color_options = ["Data Type", "Has Action", "Visit Count"]
        else:
            color_options = ["Has Action", "Visit Count"]
        color_by = st.radio("Color By", color_options, horizontal=True)
    with col2:
        size_by = st.radio("Size By", ["Visit Count", "Uniform"], horizontal=True)
    with col3:
        show_edges = st.checkbox("Show Edges", value=True)
    with col4:
        if has_test_data:
            show_test_data = st.checkbox("Show Test Data", value=True)
        else:
            show_test_data = False
    
    st.markdown("---")
    
    # Build step selector options from ALL training steps (not just unique nodes)
    # Note: We use matched_node_id from comparison_results, not a lookup table,
    # because many steps were deduplicated/merged during training
    step_selector_options = []
    step_to_node_map = {}  # Maps step selector key to node_id or test index
    
    if train_report and 'comparison_results' in train_report:
        for idx, result in enumerate(train_report['comparison_results']):
            conv_id = result['conversation_id']
            step_idx = result['step_index']
            
            # Use matched_node_id from the report (the node this step was matched to)
            node_id = result.get('matched_node_id')
            
            pred_info = {
                'predicted': result['action_taken'],
                'correct': result['correct_action'],
                'is_correct': result['is_correct'],
                'decision': result['decision'],
                'confidence': result.get('confidence', 0)
            }
            
            # Add to selector options - now showing ALL steps
            decision = result['decision']
            match_icon = "✅" if result['is_correct'] else "❌"
            
            # Always include decision in label for filtering
            if node_id is not None:
                selector_key = f"{match_icon} Train Step {step_idx} (Conv {conv_id}, Node {node_id}) → {decision}"
            else:
                # No node (shouldn't happen for REPLAY, but could for edge cases)
                selector_key = f"{match_icon} Train Step {step_idx} (Conv {conv_id}) → {decision}"
            
            step_selector_options.append(selector_key)
            step_to_node_map[selector_key] = {
                'type': 'train',
                'node_id': node_id,
                'conv_id': conv_id,
                'step_idx': step_idx,
                'pred_info': pred_info,
                'think_text': result.get('think_text')  # Store thinking block directly
            }
    
    # Create DataFrame for plotting (still using unique nodes)
    all_data = []

    # Add training data (unique nodes for visualization)
    for i, node_id in enumerate(node_ids):
        node = graph.nodes[node_id]
        
        all_data.append({
            'id': f"Train_{node_id}",
            'x': train_pos_2d[i, 0],
            'y': train_pos_2d[i, 1],
            'type': 'Training',
            'visits': node.visit_count,
            'has_action': 'Yes' if node.action_text else 'No',
            'success_rate': f"{node.get_success_rate():.1%}",
            'node_id': node_id,
            'step_index': node.step_index,
            'conv_id': node.conversation_id,
            'is_correct': None
        })

    # Add test data if available and enabled
    if has_test_data and show_test_data:
        # Create mapping for test predictions
        test_predictions = {}
        if test_report and 'test_results' in test_report:
            for result in test_report['test_results']:
                test_predictions[result['step_index']] = {
                    'predicted': result['predicted_action'],
                    'correct': result['correct_action'],
                    'is_correct': result['is_correct'],
                    'decision': result['decision'],
                    'confidence': result.get('confidence', 0),
                    'think_text': result.get('think_text')
                }
        
        for i, step_idx in enumerate(test_step_indices):
            pred_info = test_predictions.get(step_idx, None)
            
            # Add to selector options
            if pred_info:
                match_icon = "✅" if pred_info['is_correct'] else "❌"
                decision = pred_info['decision']
                selector_key = f"{match_icon} Test Step {step_idx} → {decision}"
            else:
                selector_key = f"Test Step {step_idx}"
            
            step_selector_options.append(selector_key)
            step_to_node_map[selector_key] = {
                'type': 'test',
                'test_index': i,
                'step_idx': step_idx,
                'pred_info': pred_info,
                'think_text': pred_info.get('think_text') if pred_info else None,  # Store thinking block directly
                'conv_id': None  # Test data doesn't have conv_id in the same way
            }
            
            all_data.append({
                'id': f"Test_{i}",
                'x': test_pos_2d[i, 0],
                'y': test_pos_2d[i, 1],
                'type': 'Test',
                'visits': 1,  # Test points don't have visit counts
                'has_action': 'Test',
                'success_rate': 'N/A',
                'node_id': None,
                'step_index': step_idx,
                'test_index': i,
                'is_correct': pred_info['is_correct'] if pred_info else None,
                'pred_info': pred_info
            })

    df = pd.DataFrame(all_data)
    
    # Step selector with navigation
    st.markdown("### 🔍 Step Inspector")
    
    # Show current selection at the top (if any)
    if 'current_step_index' in st.session_state and st.session_state.current_step_index > 0:
        temp_options = ["None"] + step_selector_options  # Temporarily build to get selection
        if st.session_state.current_step_index < len(temp_options):
            current_selection = temp_options[st.session_state.current_step_index]
            st.info(f"📍 **Currently Viewing:** {current_selection}")
    
    # Add filters for predictions
    st.markdown("**Filters:**")
    
    # Row 1: Correctness filter
    filter_col1, filter_col2 = st.columns([2, 4])
    with filter_col1:
        correctness_filter = st.radio(
            "By Result:",
            ["All", "✅ Correct", "❌ Incorrect"],
            horizontal=True,
            key="correctness_filter"
        )
    
    with filter_col2:
        # Show count info
        total_steps = len(step_selector_options)
        correct_count = sum(1 for key in step_selector_options if "✅" in key)
        incorrect_count = sum(1 for key in step_selector_options if "❌" in key)
        st.info(f"📊 Total: {total_steps} | ✅ Correct: {correct_count} ({correct_count/total_steps*100:.1f}%) | ❌ Incorrect: {incorrect_count} ({incorrect_count/total_steps*100:.1f}%)")
    
    # Row 2: Decision type filter
    decision_col1, decision_col2 = st.columns([2, 4])
    with decision_col1:
        decision_filter = st.radio(
            "By Decision:",
            ["All", "🎯 REPLAY", "🤖 AI_FALLBACK", "⏸️ NO_ACTION"],
            horizontal=True,
            key="decision_filter"
        )
    
    with decision_col2:
        # Show decision counts (case-insensitive)
        replay_count = sum(1 for key in step_selector_options if "replay" in key.lower())
        ai_fallback_count = sum(1 for key in step_selector_options if "ai_fallback" in key.lower())
        no_action_count = sum(1 for key in step_selector_options if "no_action" in key.lower())
        st.info(f"🎯 REPLAY: {replay_count} | 🤖 AI_FALLBACK: {ai_fallback_count} | ⏸️ NO_ACTION: {no_action_count}")
    
    # Apply filters to step selector options
    filtered_options = step_selector_options
    
    # Apply correctness filter
    if correctness_filter == "✅ Correct":
        filtered_options = [opt for opt in filtered_options if "✅" in opt]
    elif correctness_filter == "❌ Incorrect":
        filtered_options = [opt for opt in filtered_options if "❌" in opt]
    
    # Apply decision filter (case-insensitive)
    if decision_filter == "🎯 REPLAY":
        filtered_options = [opt for opt in filtered_options if "replay" in opt.lower()]
    elif decision_filter == "🤖 AI_FALLBACK":
        filtered_options = [opt for opt in filtered_options if "ai_fallback" in opt.lower()]
    elif decision_filter == "⏸️ NO_ACTION":
        filtered_options = [opt for opt in filtered_options if "no_action" in opt.lower()]
    
    # Show filtered results count
    if len(filtered_options) > 0:
        st.success(f"🔍 **Filtered Results: {len(filtered_options)} steps found**")
    else:
        st.warning(f"⚠️ **No steps match the current filter combination**")
        with st.expander("💡 Debugging Info"):
            st.write(f"Correctness filter: {correctness_filter}")
            st.write(f"Decision filter: {decision_filter}")
            st.write(f"Total steps before filtering: {len(step_selector_options)}")
            
            # Show sample of available options
            st.write("\nSample of first 5 options:")
            for opt in step_selector_options[:5]:
                st.code(opt)
    
    all_options = ["None"] + filtered_options
    
    # Initialize session state for step navigation
    if 'current_step_index' not in st.session_state:
        st.session_state.current_step_index = 0  # Start at "None"
    
    # Reset index if filter changed
    if 'last_correctness_filter' not in st.session_state:
        st.session_state.last_correctness_filter = correctness_filter
    if 'last_decision_filter' not in st.session_state:
        st.session_state.last_decision_filter = decision_filter
    
    if (st.session_state.last_correctness_filter != correctness_filter or 
        st.session_state.last_decision_filter != decision_filter):
        st.session_state.current_step_index = 0  # Reset to "None"
        st.session_state.last_correctness_filter = correctness_filter
        st.session_state.last_decision_filter = decision_filter
    
    # Ensure index is within bounds
    if st.session_state.current_step_index >= len(all_options):
        st.session_state.current_step_index = 0
    
    # Navigation buttons with callbacks
    def go_first():
        st.session_state.current_step_index = 1  # Skip "None"
    
    def go_prev():
        if st.session_state.current_step_index > 1:
            st.session_state.current_step_index -= 1
    
    def go_next():
        if st.session_state.current_step_index < len(all_options) - 1:
            st.session_state.current_step_index += 1
    
    def go_last():
        st.session_state.current_step_index = len(all_options) - 1
    
    st.markdown("**Navigation:**")
    nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([1, 1, 1, 1, 3])
    
    with nav_col1:
        st.button("⏮️ First", on_click=go_first, use_container_width=True)
    
    with nav_col2:
        st.button("◀️ Prev", on_click=go_prev, use_container_width=True)
    
    with nav_col3:
        st.button("▶️ Next", on_click=go_next, use_container_width=True)
    
    with nav_col4:
        st.button("⏭️ Last", on_click=go_last, use_container_width=True)
    
    # Dropdown selector with callback to track manual changes
    def on_selectbox_change():
        # Only update if user manually changed the selectbox
        new_index = all_options.index(st.session_state.step_selector_viz)
        st.session_state.current_step_index = new_index
    
    st.markdown("**Select Step:**")
    col1, col2 = st.columns([5, 1])
    with col1:
        # Get the currently selected step based on index
        selected_step = all_options[st.session_state.current_step_index]
        
        # Display selectbox with default value
        st.selectbox(
            f"🔍 Choose from {len(filtered_options)} steps:",
            all_options,
            index=st.session_state.current_step_index,
            key="step_selector_viz",
            on_change=on_selectbox_change
        )
    
    with col2:
        if selected_step != "None" and selected_step in step_to_node_map:
            step_info = step_to_node_map[selected_step]
            pred_info = step_info.get('pred_info')
            if pred_info and pred_info['is_correct']:
                st.success("✅")
            elif pred_info:
                st.error("❌")
    
    # Show step details if selected
    if selected_step != "None" and selected_step in step_to_node_map:
        step_info = step_to_node_map[selected_step]
        pred_info = step_info.get('pred_info')
        
        # Show currently selected step prominently
        st.markdown(f"### 🎯 Currently Selected: `{selected_step}`")
        
        if pred_info:
            st.markdown("#### 📊 Prediction Details")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Decision", pred_info['decision'].upper())
            with col2:
                st.metric("Confidence", f"{pred_info['confidence']:.3f}")
            with col3:
                match_icon = "✅ Match" if pred_info['is_correct'] else "❌ Mismatch"
                st.metric("Result", match_icon)
            
            # Load original data to get thinking block and screenshot
            st.markdown("---")
            st.markdown("#### 🖼️ Step Context")
            
            # Try to load the original step data
            original_data = None
            if step_info['type'] == 'train':
                # Load from training report (contains thinking blocks)
                if train_report and 'comparison_results' in train_report:
                    for result in train_report['comparison_results']:
                        if result['conversation_id'] == step_info['conv_id'] and result['step_index'] == step_info['step_idx']:
                            original_data = result
                            break
            else:
                # Load from test report
                if test_report and 'test_results' in test_report:
                    for result in test_report['test_results']:
                        if result['step_index'] == step_info.get('step_idx'):
                            original_data = result
                            break
            
            # Show thinking block if available
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**💭 Thinking/Reasoning:**")
                # Try getting from step_info first (cached from loading)
                think_text = step_info.get('think_text')
                
                if think_text:
                    st.text_area("Thinking Block", value=think_text, height=150, key=f"think_{selected_step}", label_visibility="collapsed")
                else:
                    st.warning("💡 No thinking block found\n\nRegenerate reports:\n```\npython3 train.py\npython3 test.py\n```")
            
            with col2:
                st.markdown("**📸 Screenshot:**")
                # Load conversation data and extract screenshot
                try:
                    conversations = load_conversation_data()
                    screenshot_found = False
                    debug_info = []
                    
                    if step_info['type'] == 'train':
                        # For training data, use conversation_id and step_index
                        conv_id = step_info.get('conv_id')
                        step_idx = step_info.get('step_idx')
                        
                        debug_info.append(f"Looking for: conv_id={conv_id}, step_idx={step_idx}")
                        debug_info.append(f"Total conversations loaded: {len(conversations)}")
                        
                        if conv_id in conversations:
                            conv_obj = conversations[conv_id]
                            messages = conv_obj.get('conversation', [])
                            debug_info.append(f"Found conv with {len(messages)} messages")
                            
                            # Parse messages using data_parser (same as train.py)
                            from src.data_parser import StreamingConversationParser
                            parser = StreamingConversationParser()
                            parser.reset(conversation_id=conv_id)
                            
                            # Process messages to extract steps
                            for message in messages:
                                step = parser.add_message(message)
                                if step and step.step_index == step_idx:
                                    target_screenshot = step.image_data
                                    if target_screenshot and len(str(target_screenshot)) > 50:
                                        # Display base64 image
                                        if target_screenshot.startswith('data:image'):
                                            target_screenshot = target_screenshot.split(',', 1)[1]
                                        
                                        image_data = base64.b64decode(target_screenshot)
                                        image = Image.open(BytesIO(image_data))
                                        st.image(image, width=700)
                                        screenshot_found = True
                                        break
                            
                            if not screenshot_found:
                                debug_info.append(f"Screenshot not found for step {step_idx}")
                        else:
                            debug_info.append(f"Conv ID {conv_id} not in conversations")
                    else:
                        # For test data - not implemented yet
                        step_idx = step_info.get('step_idx')
                        debug_info.append(f"Test step_idx={step_idx}")
                        st.info("Test data screenshot loading - to be implemented")
                    
                    if not screenshot_found:
                        st.warning("📷 Screenshot not found")
                        with st.expander("🔍 Debug Info"):
                            for info in debug_info:
                                st.text(info)
                        
                except Exception as e:
                    st.error(f"Error loading screenshot: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
            
            # Show actions side by side
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🔮 Predicted Action:**")
                st.code(pred_info['predicted'], language="text")
            with col2:
                st.markdown("**✓ Correct Action:**")
                st.code(pred_info['correct'], language="text")
        else:
            st.info("No prediction data available for this step")
    
    st.markdown("---")

    # Determine which point to highlight
    highlight_id = None
    if selected_step != "None" and selected_step in step_to_node_map:
        step_info = step_to_node_map[selected_step]
        if step_info['type'] == 'train':
            highlight_id = f"Train_{step_info['node_id']}"
        else:
            highlight_id = f"Test_{step_info['test_index']}"
        
        # Debug: Show which point should be highlighted
        with st.expander("🔍 Highlight Debug Info"):
            st.write(f"**Selected step:** {selected_step}")
            st.write(f"**Step type:** {step_info['type']}")
            st.write(f"**Highlight ID:** {highlight_id}")
            st.write(f"**Node ID:** {step_info.get('node_id', 'N/A')}")
            st.write(f"**Test Index:** {step_info.get('test_index', 'N/A')}")
            
            # Check if highlight_id exists in dataframe
            if highlight_id:
                matching = df[df['id'] == highlight_id]
                st.write(f"**Found in graph:** {'Yes' if not matching.empty else 'No'}")
                if not matching.empty:
                    st.write(f"**Position:** x={matching.iloc[0]['x']:.2f}, y={matching.iloc[0]['y']:.2f}")
    
    # Add highlight column
    df['is_highlighted'] = df['id'] == highlight_id
    
    # Create position lookup for edges (only for training nodes)
    # Always create this for edge rendering
    train_df = df[df['type'] == 'Training']
    test_df = df[df['type'] == 'Test'] if has_test_data and show_test_data else pd.DataFrame()
    node_pos = {row['node_id']: (row['x'], row['y']) for _, row in train_df.iterrows() if row['node_id'] is not None}

    # Create plot based on color option
    if color_by == "Data Type" and has_test_data and show_test_data:
        # Separate training and test data
        train_df = df[df['type'] == 'Training']
        test_df = df[df['type'] == 'Test']

        fig = go.Figure()

        # Split training data into normal and highlighted
        train_normal = train_df[~train_df['is_highlighted']]
        train_highlight = train_df[train_df['is_highlighted']]
        
        # Normal training nodes
        if not train_normal.empty:
            fig.add_trace(go.Scatter(
                x=train_normal['x'],
                y=train_normal['y'],
                mode='markers',
                name='Training Nodes',
                marker=dict(
                    color='#1f77b4',  # Blue
                    size=train_normal['visits'] if size_by == "Visit Count" else 8,
                    sizemode='area',
                    sizeref=2.*max(train_df['visits'])/(40.**2) if size_by == "Visit Count" else None,
                    sizemin=4,
                    line=dict(width=1, color='white')
                ),
                text=[f"Node {row['node_id']}<br>Visits: {row['visits']}<br>Success: {row['success_rate']}" for _, row in train_normal.iterrows()],
                hoverinfo='text',
                showlegend=True
            ))
        
        # Highlighted training node
        if not train_highlight.empty:
            fig.add_trace(go.Scatter(
                x=train_highlight['x'],
                y=train_highlight['y'],
                mode='markers',
                name='⭐ SELECTED STEP',
                marker=dict(
                    color='#FFD700',  # Gold
                    size=35,  # Much larger!
                    symbol='star',
                    line=dict(width=3, color='red')  # Red outline for visibility
                ),
                text=[f"⭐⭐⭐ SELECTED ⭐⭐⭐<br>Node {row['node_id']}<br>Visits: {row['visits']}<br>Success: {row['success_rate']}" for _, row in train_highlight.iterrows()],
                hoverinfo='text',
                showlegend=True
            ))

        # Split test data into normal and highlighted
        test_normal = test_df[~test_df['is_highlighted']]
        test_highlight = test_df[test_df['is_highlighted']]
        
        # Normal test points
        if not test_normal.empty:
            fig.add_trace(go.Scatter(
                x=test_normal['x'],
                y=test_normal['y'],
                mode='markers',
                name='Test Data',
                marker=dict(
                    color='#ff7f0e',  # Orange
                    size=6,
                    symbol='diamond',
                    line=dict(width=1, color='white')
                ),
                text=[f"Test Step {row['step_index']}" for _, row in test_normal.iterrows()],
                hoverinfo='text',
                showlegend=True
            ))
        
        # Highlighted test point
        if not test_highlight.empty:
            fig.add_trace(go.Scatter(
                x=test_highlight['x'],
                y=test_highlight['y'],
                mode='markers',
                name='⭐ SELECTED TEST',
                marker=dict(
                    color='#FFD700',  # Gold
                    size=30,  # Much larger!
                    symbol='star-diamond',
                    line=dict(width=3, color='red')  # Red outline for visibility
                ),
                text=[f"⭐⭐⭐ SELECTED ⭐⭐⭐<br>Test Step {row['step_index']}" for _, row in test_highlight.iterrows()],
                hoverinfo='text',
                showlegend=True
            ))

        fig.update_layout(
            title="Semantic Graph - Training vs Test Data",
            xaxis_title="MDS Dimension 1",
            yaxis_title="MDS Dimension 2",
            showlegend=True
        )

    elif color_by == "Has Action":
        # Use all data if test data is hidden, otherwise just training for action coloring
        plot_df = df[~df['is_highlighted']]  # Non-highlighted points
        plot_highlight = df[df['is_highlighted']]  # Highlighted point

        # Simple hover: node_id, visits, success_rate only
        fig = px.scatter(
            plot_df,
            x='x',
            y='y',
            color='has_action',
            size='visits' if size_by == "Visit Count" else None,
            hover_data={'node_id': True, 'visits': True, 'success_rate': True, 'x': False, 'y': False},
            color_discrete_map={'Yes': '#00ff00', 'No': '#4a90e2', 'Test': '#ff7f0e'},
            labels={'has_action': 'Has Action'},
            title="Semantic Graph - MDS Projection"
        )
        
        # Add highlighted point if exists
        if not plot_highlight.empty:
            fig.add_trace(go.Scatter(
                x=plot_highlight['x'],
                y=plot_highlight['y'],
                mode='markers',
                name='⭐ Selected',
                marker=dict(
                    color='#FFD700',
                    size=20,
                    symbol='star',
                    line=dict(width=2, color='white')
                ),
                text=[f"⭐ SELECTED ⭐<br>Node {row['node_id']}<br>Visits: {row['visits']}" for _, row in plot_highlight.iterrows()],
                hoverinfo='text',
                showlegend=True
            ))
            
    else:  # Color by visit count
        # Use all data (test points will show visit count = 1)
        plot_df = df[~df['is_highlighted']]  # Non-highlighted points
        plot_highlight = df[df['is_highlighted']]  # Highlighted point

        # Simple hover: node_id, visits, success_rate only
        fig = px.scatter(
            plot_df,
            x='x',
            y='y',
            color='visits',
            size='visits' if size_by == "Visit Count" else None,
            hover_data={'node_id': True, 'visits': True, 'success_rate': True, 'x': False, 'y': False},
            color_continuous_scale='Viridis',
            labels={'visits': 'Visit Count'},
            title="Semantic Graph - MDS Projection (Colored by Visit Count)"
        )
    
        # Add highlighted point if exists
        if not plot_highlight.empty:
            fig.add_trace(go.Scatter(
                x=plot_highlight['x'],
                y=plot_highlight['y'],
                mode='markers',
                name='⭐ Selected',
                marker=dict(
                    color='#FFD700',
                    size=20,
                    symbol='star',
                    line=dict(width=2, color='white')
                ),
                text=[f"⭐ SELECTED ⭐<br>Node {row['node_id']}<br>Visits: {row['visits']}" for _, row in plot_highlight.iterrows()],
                hoverinfo='text',
                showlegend=True
            ))
    
    # Add edges if requested (show training graph edges)
    if show_edges and graph.edges and len(node_pos) > 0:
        edge_x = []
        edge_y = []
        edge_count = 0
        
        for (from_id, to_id), edge in graph.edges.items():
            if from_id in node_pos and to_id in node_pos:
                x0, y0 = node_pos[from_id]
                x1, y1 = node_pos[to_id]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_count += 1
        
        # Add edge traces
        if len(edge_x) > 0:
            fig.add_trace(go.Scatter(
                x=edge_x,
                y=edge_y,
                mode='lines',
                line=dict(color='rgba(128, 128, 128, 0.3)', width=0.5),
                hoverinfo='skip',
                showlegend=False,
                name=f'Edges ({edge_count})'
            ))
            
            # Move edges to back (only if using plotly express figures)
            if color_by != "Data Type":
                # Move scatter traces to the front
                fig.data = fig.data[-1:] + fig.data[:-1]
    
    fig.update_layout(
        height=700,
        xaxis_title="",
        yaxis_title="",
        showlegend=True,
        plot_bgcolor='rgba(20, 20, 40, 0.9)',
        paper_bgcolor='rgba(20, 20, 40, 1)'
    )
    
    fig.update_xaxes(showticklabels=False, showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)')
    fig.update_yaxes(showticklabels=False, showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)')
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info(
        "**Interpretation:** Nearby nodes = visually similar UI screens. "
        "The distances represent semantic similarity from embeddings!"
    )


def show_comparison_details(report):
    """Show detailed comparison results."""
    st.header("📋 Comparison Details")
    
    comparisons = report['comparison_results']
    
    if not comparisons:
        st.warning("No comparison data available")
        return
    
    df = pd.DataFrame(comparisons)
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        decision_filter = st.multiselect(
            "Filter by Decision",
            options=df['decision'].unique(),
            default=df['decision'].unique()
        )
    
    with col2:
        correctness_filter = st.multiselect(
            "Filter by Correctness",
            options=[True, False],
            default=[True, False],
            format_func=lambda x: "✅ Correct" if x else "❌ Wrong"
        )
    
    with col3:
        conversation_filter = st.multiselect(
            "Filter by Conversation",
            options=sorted(df['conversation_id'].unique()),
            default=sorted(df['conversation_id'].unique())[:3]  # First 3
        )
    
    # Apply filters
    filtered_df = df[
        (df['decision'].isin(decision_filter)) &
        (df['is_correct'].isin(correctness_filter)) &
        (df['conversation_id'].isin(conversation_filter))
    ]
    
    st.markdown(f"**Showing {len(filtered_df)} / {len(df)} comparisons**")
    
    # Display table
    display_df = filtered_df[['step_index', 'conversation_id', 'decision', 'confidence', 'action_taken', 'correct_action', 'is_correct']].copy()
    display_df['decision'] = display_df['decision'].apply(lambda x: f"🔄 {x}" if x == 'replay' else f"🤖 {x}" if x == 'ai_fallback' else f"ℹ️ {x}")
    display_df['is_correct'] = display_df['is_correct'].apply(lambda x: "✅" if x else "❌")
    display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.3f}")
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "step_index": "Step",
            "conversation_id": "Conv",
            "decision": "Decision",
            "confidence": "Conf",
            "action_taken": st.column_config.TextColumn("Action Taken", width="large"),
            "correct_action": st.column_config.TextColumn("Correct Action", width="large"),
            "is_correct": "Status"
        }
    )
    
    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        "📥 Download Filtered Data (CSV)",
        csv,
        "comparison_results.csv",
        "text/csv"
    )


if __name__ == "__main__":
    main()

