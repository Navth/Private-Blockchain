import asyncio
import json
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from core.blockchain import Blockchain
from core.transaction import Transaction
from core.block import Block
from consensus.validator import Validator
from consensus.pbft import PBFTConsensus
from config import NetworkConfig, ValidatorConfig

@dataclass
class PeerInfo:
    """Information about a peer node."""
    node_id: str
    host: str
    port: int
    public_key: str
    is_connected: bool = False
    last_heartbeat: float = 0.0

class NetworkNode:
    
    def __init__(self, node_id: str, host: str, port: int, validator: Validator):
        """Initialize network node."""
        self.node_id = node_id
        self.host = host
        self.port = port
        self.validator = validator
        
        self.blockchain = Blockchain()
        
        self.pbft: Optional[PBFTConsensus] = None
        
        self.peers: Dict[str, PeerInfo] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        
        self.blocks_mined = 0
        self.txs_processed = 0
        self.start_time = time.time()
    
    def add_peer(self, peer_id: str, host: str, port: int, public_key: str):
        """Register a peer node."""
        self.peers[peer_id] = PeerInfo(
            node_id=peer_id,
            host=host,
            port=port,
            public_key=public_key
        )
    
    def get_all_validators(self) -> List[Validator]:
        """Get list of all validators (from peers)."""
        return [self.validator]
    
    async def init_consensus(self):
        """Initialize PBFT consensus."""
        validators = self.get_all_validators()
        self.pbft = PBFTConsensus(self.validator, validators)
    
    async def broadcast_transaction(self, tx: Transaction):
        """Broadcast transaction to all peers."""
        msg = {
            "type": "TRANSACTION",
            "tx": tx.to_dict(),
            "from": self.node_id
        }
        
        print(f"[{self.node_id}] Broadcasting transaction: {tx.id}")
        
        self.blockchain.add_transaction(tx)
        
        await self.message_queue.put(msg)
    
    async def propose_block(self) -> Optional[Block]:
        """Propose new block for consensus."""
        print(f"[{self.node_id}] Proposing new block...")
        
        block = self.blockchain.mine_block(
            proposer=self.validator.public_key,
            tx_count=5
        )
        
        if not block:
            print(f"[{self.node_id}] No transactions to mine")
            return None
        
        print(f"[{self.node_id}] Mined block #{block.index} with {len(block.transactions)} txs")
        return block
    
    async def consensus_round(self):
        """Run one consensus round."""
        block = await self.propose_block()
        
        if not block:
            await asyncio.sleep(1)
            return
        
        if self.pbft:
            success = await self.pbft.run_consensus(block)
            
            if success:
                if self.blockchain.add_block(block):
                    self.blocks_mined += 1
                    print(f"[{self.node_id}] ✓ Block #{block.index} added to chain")
                    
                    await self.broadcast_block(block)
                else:
                    print(f"[{self.node_id}] ✗ Block rejected")
    
    async def broadcast_block(self, block: Block):
        """Broadcast finalized block to peers."""
        msg = {
            "type": "BLOCK",
            "block": block.to_dict(),
            "from": self.node_id
        }
        
        print(f"[{self.node_id}] Broadcasting block #{block.index}")
        
        await self.message_queue.put(msg)
    
    async def sync_with_peer(self, peer_id: str):
        """Sync blockchain state with a peer."""
        if peer_id not in self.peers:
            return
        
        peer = self.peers[peer_id]
        print(f"[{self.node_id}] Syncing with {peer_id} at {peer.host}:{peer.port}")
        
        peer.is_connected = True
        peer.last_heartbeat = time.time()
    
    async def handle_message(self, msg: Dict[str, Any]):
        """Handle incoming message from network."""
        msg_type = msg.get("type")
        
        if msg_type == "TRANSACTION":
            tx_dict = msg.get("tx")
            print(f"[{self.node_id}] Received transaction from {msg.get('from')}")
            self.txs_processed += 1
        
        elif msg_type == "BLOCK":
            block_dict = msg.get("block")
            print(f"[{self.node_id}] Received block #{block_dict.get('index')} from {msg.get('from')}")
        
        elif msg_type == "SYNC_REQUEST":
            print(f"[{self.node_id}] Sync request from {msg.get('from')}")
    
    async def run(self):
        """Run the node."""
        print(f"\n[{self.node_id}] Starting node at {self.host}:{self.port}")
        print(f"[{self.node_id}] Validator: {self.validator.id}")
        print(f"[{self.node_id}] Public key: {self.validator.public_key[:16]}...")
        
        await self.init_consensus()
        
        try:
            while True:
                await self.consensus_round()
                await asyncio.sleep(2)
        
        except KeyboardInterrupt:
            print(f"\n[{self.node_id}] Shutting down...")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get node statistics."""
        uptime = time.time() - self.start_time
        
        return {
            "node_id": self.node_id,
            "host": f"{self.host}:{self.port}",
            "chain_length": len(self.blockchain.chain),
            "blocks_mined": self.blocks_mined,
            "txs_processed": self.txs_processed,
            "mempool_size": self.blockchain.mempool.size(),
            "uptime_seconds": round(uptime, 1),
            "validator_score": round(self.validator.calculate_v_score(), 3),
            "peers_connected": sum(1 for p in self.peers.values() if p.is_connected)
        }
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return (f"Node({self.node_id} | "
                f"Chain={stats['chain_length']} | "
                f"Mined={stats['blocks_mined']} | "
                f"Peers={stats['peers_connected']})")
