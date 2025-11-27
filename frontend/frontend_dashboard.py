
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
from typing import List, Dict, Any
import numpy as np


from core.blockchain import Blockchain
from core.transaction import Transaction
from consensus.validator import Validator
from network.node import NetworkNode
from config import Tier, ValidatorConfig, PriorityConfig


st.set_page_config(
    page_title="🔗 Blockchain Dashboard",
    page_icon="⛓️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== CUSTOM CSS =====
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .success {
        color: #00ff00;
        font-weight: bold;
    }
    .danger {
        color: #ff0000;
        font-weight: bold;
    }
    .warning {
        color: #ff9800;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


# ===== SESSION STATE =====
if 'blockchain' not in st.session_state:
    st.session_state.blockchain = Blockchain()
    st.session_state.validators = []
    st.session_state.nodes = []
    st.session_state.transactions_created = 0
    st.session_state.blocks_mined = 0
    
    # Create validators
    for i in range(ValidatorConfig.TIER_1_COUNT):
        v = Validator(f"T1_Validator_{i}", 500, Tier.TIER_1)
        st.session_state.validators.append(v)
    
    for i in range(ValidatorConfig.TIER_2_COUNT):
        v = Validator(f"T2_Validator_{i}", 300, Tier.TIER_2)
        st.session_state.validators.append(v)
    
    for i in range(ValidatorConfig.TIER_3_COUNT):
        v = Validator(f"T3_Validator_{i}", 100, Tier.TIER_3)
        st.session_state.validators.append(v)


# ===== SIDEBAR =====
st.sidebar.title("🔗 Blockchain Dashboard")
page = st.sidebar.radio("Navigation", [
    "📊 Overview",
    "💱 Transactions",
    "📦 Blocks",
    "✅ Validators",
    "🌐 Network",
    "⚙️ Consensus",
    "📈 Analytics"
])


# ===== PAGE: OVERVIEW =====
if page == "📊 Overview":
    st.title("🔗 Blockchain Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Chain Length",
            len(st.session_state.blockchain.chain),
            "+1" if len(st.session_state.blockchain.chain) > 1 else "Genesis"
        )
    
    with col2:
        total_txs = sum(len(b.transactions) for b in st.session_state.blockchain.chain)
        st.metric("Total Transactions", total_txs)
    
    with col3:
        mempool_size = st.session_state.blockchain.mempool.size()
        st.metric("Pending Transactions", mempool_size)
    
    with col4:
        is_valid = st.session_state.blockchain.is_chain_valid()
        st.metric(
            "Chain Validity",
            "✓ Valid" if is_valid else "✗ Invalid",
            delta="OK" if is_valid else "ERROR"
        )
    
    # Chain visualization
    st.subheader("📈 Blockchain Growth")
    
    chain_data = {
        "Block #": list(range(len(st.session_state.blockchain.chain))),
        "Transactions": [len(b.transactions) for b in st.session_state.blockchain.chain],
        "Timestamp": [datetime.fromtimestamp(b.timestamp).strftime("%H:%M:%S") 
                     for b in st.session_state.blockchain.chain]
    }
    
    fig = px.bar(
        x=chain_data["Block #"],
        y=chain_data["Transactions"],
        title="Transactions per Block",
        labels={"x": "Block Number", "y": "Transaction Count"}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Latest blocks
    st.subheader("📦 Latest Blocks")
    
    latest_blocks = st.session_state.blockchain.chain[-5:]
    for block in reversed(latest_blocks):
        with st.expander(f"Block #{block.index} - {len(block.transactions)} txs - Hash: {block.hash[:8]}..."):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Index:** {block.index}")
                st.write(f"**Proposer:** {block.proposer[:16]}...")
                st.write(f"**Timestamp:** {datetime.fromtimestamp(block.timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
            
            with col2:
                st.write(f"**Hash:** `{block.hash}`")
                st.write(f"**Previous Hash:** `{block.previous_hash[:16]}...`")
                st.write(f"**Merkle Root:** `{block.merkle_root[:16]}...`")


# ===== PAGE: TRANSACTIONS =====
elif page == "💱 Transactions":
    st.title("💱 Transaction Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Create New Transaction")
        
        if st.session_state.validators:
            sender_idx = st.selectbox("Sender", range(len(st.session_state.validators)), 
                                     format_func=lambda i: st.session_state.validators[i].id)
            recipient_idx = st.selectbox("Recipient", range(len(st.session_state.validators)),
                                        format_func=lambda i: st.session_state.validators[i].id)
            amount = st.number_input("Amount", value=100.0, min_value=0.1)
            
            severity = st.slider("Severity (Emergency Level)", 0.0, 1.0, 0.5)
            urgency = st.slider("Urgency", 0.0, 1.0, 0.5)
            risk = st.slider("Risk Profile", 0.0, 1.0, 0.5)
            
            if st.button("✓ Create & Sign Transaction"):
                sender = st.session_state.validators[sender_idx]
                recipient = st.session_state.validators[recipient_idx]
                
                tx = Transaction(
                    sender=sender.public_key,
                    recipient=recipient.public_key,
                    amount=amount,
                    data={
                        "severity": severity,
                        "urgency": urgency,
                        "risk": risk
                    }
                )
                
                tx.sign(sender.private_key)
                
                if st.session_state.blockchain.add_transaction(tx):
                    st.success(f"✓ Transaction created: {tx.id}")
                    st.session_state.transactions_created += 1
                else:
                    st.error("✗ Failed to add transaction")
    
    with col2:
        st.subheader("Mempool Status")
        
        mempool_stats = st.session_state.blockchain.mempool.get_stats()
        
        st.metric("Total Pending", mempool_stats["size"])
        st.metric("Avg Priority", f"{mempool_stats['avg_priority']:.3f}")
        st.metric("Max Priority", f"{mempool_stats['max_priority']:.3f}")
        st.metric("Min Priority", f"{mempool_stats['min_priority']:.3f}")
        
        # Priority distribution
        st.subheader("Priority Distribution")
        if mempool_stats["size"] > 0:
            priorities = [tx.priority for tx in st.session_state.blockchain.mempool.transactions]
            
            fig = px.histogram(
                x=priorities,
                nbins=10,
                title="Transaction Priority Distribution",
                labels={"x": "Priority Score", "count": "Count"}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # All transactions in mempool
    st.subheader("Pending Transactions")
    
    if st.session_state.blockchain.mempool.size() > 0:
        tx_data = []
        for tx in st.session_state.blockchain.mempool.peek_transactions(10):
            tx_data.append({
                "ID": tx.id[:8],
                "From": tx.sender[:8] + "...",
                "To": tx.recipient[:8] + "...",
                "Amount": tx.amount,
                "Priority": f"{tx.priority:.3f}",
                "Signed": "✓" if tx.signature else "✗"
            })
        
        df = pd.DataFrame(tx_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No pending transactions")


# ===== PAGE: BLOCKS =====
elif page == "📦 Blocks":
    st.title("📦 Block Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Mine New Block")
        
        if st.session_state.validators:
            proposer_idx = st.selectbox("Proposer (Miner)", 
                                       range(len(st.session_state.validators)),
                                       format_func=lambda i: st.session_state.validators[i].id)
            tx_count = st.slider("Transactions to Include", 1, 10, 5)
            
            if st.button("⛏️ Mine Block"):
                proposer = st.session_state.validators[proposer_idx]
                
                block = st.session_state.blockchain.mine_block(
                    proposer=proposer.public_key,
                    tx_count=tx_count
                )
                
                if block:
                    if st.session_state.blockchain.add_block(block):
                        st.success(f"✓ Block #{block.index} mined successfully!")
                        st.session_state.blocks_mined += 1
                    else:
                        st.error("✗ Block validation failed")
                else:
                    st.warning("⚠ No transactions to mine")
    
    with col2:
        st.subheader("Chain Statistics")
        
        st.metric("Total Blocks", len(st.session_state.blockchain.chain))
        st.metric("Blocks Mined", st.session_state.blocks_mined)
        
        total_txs = sum(len(b.transactions) for b in st.session_state.blockchain.chain)
        st.metric("Total Transactions", total_txs)
        
        avg_txs = total_txs / len(st.session_state.blockchain.chain) if st.session_state.blockchain.chain else 0
        st.metric("Avg Txs/Block", f"{avg_txs:.1f}")
        
        is_valid = st.session_state.blockchain.is_chain_valid()
        st.metric("Chain Valid", "✓ Yes" if is_valid else "✗ No", 
                 delta="OK" if is_valid else "ERROR")
    
    # Block details
    st.subheader("Block Explorer")
    
    if len(st.session_state.blockchain.chain) > 1:
        block_idx = st.slider("Select Block #", 0, len(st.session_state.blockchain.chain) - 1)
        block = st.session_state.blockchain.chain[block_idx]
        
        with st.expander(f"Block #{block.index} Details", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Index:**", block.index)
                st.write("**Hash:**", f"`{block.hash}`")
                st.write("**Previous Hash:**", f"`{block.previous_hash[:16]}...`")
            
            with col2:
                st.write("**Proposer:**", f"`{block.proposer[:16]}...`")
                st.write("**Timestamp:**", datetime.fromtimestamp(block.timestamp).strftime("%Y-%m-%d %H:%M:%S"))
                st.write("**Merkle Root:**", f"`{block.merkle_root[:16]}...`")
            
            with col3:
                integrity = block.verify_integrity()
                txs_valid = block.verify_transactions()
                st.write("**Integrity:**", "✓ Valid" if integrity else "✗ Invalid")
                st.write("**Txs Valid:**", "✓ Valid" if txs_valid else "✗ Invalid")
                st.write("**Transaction Count:**", len(block.transactions))
            
            # Transactions in block
            st.write("**Transactions in Block:**")
            if block.transactions:
                tx_data = []
                for tx in block.transactions:
                    tx_data.append({
                        "ID": tx.id[:8],
                        "From": tx.sender[:8] + "...",
                        "To": tx.recipient[:8] + "...",
                        "Amount": tx.amount,
                        "Signed": "✓" if tx.verify_signature() else "✗"
                    })
                df = pd.DataFrame(tx_data)
                st.dataframe(df, use_container_width=True)
            else:
                st.write("Genesis block (no transactions)")


# ===== PAGE: VALIDATORS =====
elif page == "✅ Validators":
    st.title("✅ Validator Management")
    
    # Validator tier breakdown
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Tier-1 (Emergency)", ValidatorConfig.TIER_1_COUNT)
    with col2:
        st.metric("Tier-2 (Standard)", ValidatorConfig.TIER_2_COUNT)
    with col3:
        st.metric("Tier-3 (Audit)", ValidatorConfig.TIER_3_COUNT)
    
    # Validator table
    st.subheader("Validator Details")
    
    validator_data = []
    for v in st.session_state.validators:
        v_score = v.calculate_v_score()
        validator_data.append({
            "ID": v.id,
            "Tier": v.tier.name,
            "Stake": v.stake,
            "Reputation": f"{v.reputation:.3f}",
            "V-Score": f"{v_score:.3f}",
            "Blocks Validated": v.blocks_validated,
            "Uptime": f"{v.uptime*100:.1f}%",
            "Public Key": v.public_key[:8] + "..."
        })
    
    df = pd.DataFrame(validator_data)
    st.dataframe(df, use_container_width=True)
    
    # Score breakdown visualization
    st.subheader("Validator Scores Comparison")
    
    v_scores = {v.id: v.calculate_v_score() for v in st.session_state.validators}
    
    fig = px.bar(
        x=list(v_scores.keys()),
        y=list(v_scores.values()),
        title="Validator Scores (Higher = More Influence)",
        labels={"x": "Validator", "y": "V-Score"},
        color=list(v_scores.values()),
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Reputation tracker
    st.subheader("Reputation Trends")
    
    reputations = {v.id: v.reputation for v in st.session_state.validators}
    
    fig = px.bar(
        x=list(reputations.keys()),
        y=list(reputations.values()),
        title="Validator Reputation",
        labels={"x": "Validator", "y": "Reputation"},
        color=list(reputations.values()),
        color_continuous_scale="RdYlGn"
    )
    st.plotly_chart(fig, use_container_width=True)


# ===== PAGE: NETWORK =====
elif page == "🌐 Network":
    st.title("🌐 Network Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Validators", len(st.session_state.validators))
    with col2:
        st.metric("Active Nodes", len(st.session_state.validators))
    with col3:
        st.metric("Transactions Created", st.session_state.transactions_created)
    
    # Network visualization
    st.subheader("Network Topology")
    
    fig = go.Figure()
    
    # Add validator nodes
    x_pos = []
    y_pos = []
    labels = []
    colors = []
    
    for i, v in enumerate(st.session_state.validators):
        angle = (i / len(st.session_state.validators)) * 2 * 3.14159
        x = 10 * np.cos(angle)
        y = 10 * np.sin(angle)
        
        x_pos.append(x)
        y_pos.append(y)
        labels.append(v.id)
        
        if v.tier == Tier.TIER_1:
            colors.append("red")
        elif v.tier == Tier.TIER_2:
            colors.append("blue")
        else:
            colors.append("green")
    
    fig.add_trace(go.Scatter(
        x=x_pos, y=y_pos,
        mode='markers+text',
        text=labels,
        textposition="top center",
        marker=dict(size=15, color=colors),
        name="Validators"
    ))
    
    fig.update_layout(
        title="Network Topology",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Network stats
    st.subheader("Network Statistics")
    
    stats_data = {
        "Metric": ["Total Blocks", "Total Transactions", "Pending Transactions", "Chain Valid"],
        "Value": [
            len(st.session_state.blockchain.chain),
            sum(len(b.transactions) for b in st.session_state.blockchain.chain),
            st.session_state.blockchain.mempool.size(),
            "✓ Yes" if st.session_state.blockchain.is_chain_valid() else "✗ No"
        ]
    }
    
    df = pd.DataFrame(stats_data)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ===== PAGE: CONSENSUS =====
elif page == "⚙️ Consensus":
    st.title("⚙️ PBFT Consensus Algorithm")
    
    st.subheader("Consensus Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Byzantine Fault Tolerance**")
        n = len(st.session_state.validators)
        f = (n - 1) // 3
        st.write(f"Total Validators (n): {n}")
        st.write(f"Max Byzantine (f): {f}")
        st.write(f"Quorum Needed: {2*f + 1}")
    
    with col2:
        st.write("**PBFT Phases**")
        st.write("1. **Pre-Prepare**: Primary proposes block")
        st.write("2. **Prepare**: Validators validate proposal")
        st.write("3. **Commit**: Validators finalize block")
    
    with col3:
        st.write("**Quorum Requirements**")
        st.write(f"Emergency: {int(0.51*n)} validators")
        st.write(f"Normal: {int(0.67*n)} validators")
        st.write(f"Routine: {int(0.75*n)} validators")
    
    # CAMTC Algorithm
    st.subheader("CAMTC Algorithm - Transaction Priority Scoring")
    
    st.write(f"**Priority Formula:** P(tx) = α·severity + β·urgency + γ·resources + δ·risk")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("α (Severity)", PriorityConfig.ALPHA)
    with col2:
        st.metric("β (Urgency)", PriorityConfig.BETA)
    with col3:
        st.metric("γ (Resources)", PriorityConfig.GAMMA)
    with col4:
        st.metric("δ (Risk)", PriorityConfig.DELTA)
    
    # Priority categories
    st.write("**Priority Categories:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"🔴 **Emergency**: P > {PriorityConfig.EMERGENCY_THRESHOLD}")
        st.write(f"Block Size: {PriorityConfig.BLOCK_SIZE_EMERGENCY} txs")
    
    with col2:
        st.write(f"🟡 **Normal**: {PriorityConfig.NORMAL_THRESHOLD} < P ≤ {PriorityConfig.EMERGENCY_THRESHOLD}")
        st.write(f"Block Size: {PriorityConfig.BLOCK_SIZE_NORMAL} txs")
    
    with col3:
        st.write(f"🟢 **Routine**: P ≤ {PriorityConfig.NORMAL_THRESHOLD}")
        st.write(f"Block Size: {PriorityConfig.BLOCK_SIZE_ROUTINE} txs")
    
    # Consensus flow diagram
    st.subheader("Consensus Flow")
    st.write("""
    ```
    Transaction → Mempool (Priority Queue) → Primary Proposes Block
         ↓
    All Validators Receive Proposal (Pre-Prepare)
         ↓
    Validators Validate Block (Prepare Phase)
         ↓
    If 2f+1 Validators Agree → Commit Phase
         ↓
    Block Added to Chain (FINALIZED)
    ```
    """)


# ===== PAGE: ANALYTICS =====
elif page == "📈 Analytics":
    st.title("📈 Analytics & Performance")
    
    # Performance metrics
    st.subheader("Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_txs = sum(len(b.transactions) for b in st.session_state.blockchain.chain)
    blocks_count = len(st.session_state.blockchain.chain)
    
    with col1:
        st.metric("Total Blocks", blocks_count)
    with col2:
        st.metric("Total Transactions", total_txs)
    with col3:
        avg_block_time = (st.session_state.blockchain.chain[-1].timestamp - 
                         st.session_state.blockchain.chain[0].timestamp) / (blocks_count - 1) if blocks_count > 1 else 0
        st.metric("Avg Block Time", f"{avg_block_time:.1f}s")
    with col4:
        tps = total_txs / (st.session_state.blockchain.chain[-1].timestamp - 
                          st.session_state.blockchain.chain[0].timestamp + 1)
        st.metric("TPS", f"{tps:.2f}")
    
    # Blockchain growth
    st.subheader("Blockchain Growth Over Time")
    
    block_times = [datetime.fromtimestamp(b.timestamp) for b in st.session_state.blockchain.chain]
    block_indices = list(range(len(st.session_state.blockchain.chain)))
    tx_counts = [len(b.transactions) for b in st.session_state.blockchain.chain]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=block_times,
        y=block_indices,
        mode='lines+markers',
        name='Blocks',
        fill='tozeroy'
    ))
    
    fig.update_layout(
        title="Block Growth Timeline",
        xaxis_title="Time",
        yaxis_title="Block #",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Transaction distribution
    st.subheader("Transaction Distribution Across Blocks")
    
    fig = px.bar(
        x=block_indices,
        y=tx_counts,
        title="Transactions per Block",
        labels={"x": "Block #", "y": "Transaction Count"}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Validator performance
    st.subheader("Validator Performance")
    
    perf_data = []
    for v in st.session_state.validators:
        win_rate = (v.correct_votes / (v.blocks_validated + 1)) * 100
        perf_data.append({
            "Validator": v.id,
            "Score": v.calculate_v_score(),
            "Reputation": v.reputation,
            "Win Rate": win_rate,
            "Uptime": v.uptime * 100
        })
    
    df = pd.DataFrame(perf_data)
    st.dataframe(df, use_container_width=True)
    
    # Export options
    st.subheader("Export Data")
    
    if st.button("📥 Export Chain to JSON"):
        chain_data = st.session_state.blockchain.get_chain()
        st.json(chain_data[-1] if chain_data else {})  # Show last block
    
    if st.button("📥 Export Validators to CSV"):
        df = pd.DataFrame(perf_data)
        st.dataframe(df)


# Import numpy for network visualization
import numpy as np

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <p>🔗 <b>Production Blockchain Dashboard</b> | CAMTC + PBFT Consensus | Multi-Tier Validators</p>
        <p>Status: <span class="success">✓ Running</span></p>
    </div>
""", unsafe_allow_html=True)