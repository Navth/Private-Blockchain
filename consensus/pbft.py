import asyncio
import time
from typing import List, Dict, Set, Optional, Any
from core.block import Block
from core.transaction import Transaction
from consensus.validator import Validator
from config import ConsensusConfig, ValidatorConfig

class PBFTMessage:
    """PBFT message for network communication."""
    
    def __init__(self, msg_type: str, content: Dict[str, Any], sender: str):
        self.type = msg_type
        self.content = content
        self.sender = sender
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict:
        return {
            "type": self.type,
            "content": self.content,
            "sender": self.sender,
            "timestamp": self.timestamp
        }

class PBFTConsensus:
    
    def __init__(self, validator: Validator, all_validators: List[Validator]):
        """Initialize PBFT consensus."""
        self.validator = validator
        self.all_validators = all_validators
        self.n = len(all_validators)
        self.f = (self.n - 1) // 3
        self.quorum = 2 * self.f + 1
        
        self.round = 0
        self.view = 0
        self.current_block: Optional[Block] = None
        
        self.pre_prepare_log: Dict[int, PBFTMessage] = {}
        self.prepare_log: Dict[str, Set[str]] = {}
        self.commit_log: Dict[str, Set[str]] = {}
        
        self.timers: Dict[str, asyncio.Task] = {}
    
    def select_primary(self) -> Validator:
        """Select primary validator for this view (round-robin)."""
        primary_idx = self.view % self.n
        return self.all_validators[primary_idx]
    
    async def pre_prepare_phase(self, block: Block) -> bool:
        primary = self.select_primary()
        
        if self.validator.public_key != primary.public_key:
            return await self._wait_pre_prepare(block)
        
        print(f"[PBFT] Primary {self.validator.id}: Pre-prepare block #{block.index}")
        
        msg = PBFTMessage(
            "PRE_PREPARE",
            {
                "block_index": block.index,
                "block_hash": block.hash,
                "view": self.view
            },
            self.validator.public_key
        )
        
        self.pre_prepare_log[block.index] = msg
        self.current_block = block
        
        return True
    
    async def prepare_phase(self, block: Block) -> bool:
        print(f"[PBFT] {self.validator.id}: Prepare phase for block #{block.index}")
        
        if not block.verify_all():
            print(f"[PBFT] {self.validator.id}: Block validation failed!")
            return False
        
        msg = PBFTMessage(
            "PREPARE",
            {
                "block_hash": block.hash,
                "view": self.view
            },
            self.validator.public_key
        )
        
        if block.hash not in self.prepare_log:
            self.prepare_log[block.hash] = set()
        
        self.prepare_log[block.hash].add(self.validator.public_key)
        
        vote_count = len(self.prepare_log[block.hash])
        
        print(f"[PBFT] {self.validator.id}: Prepare votes: {vote_count}/{self.n}")
        
        if vote_count < self.quorum:
            print(f"[PBFT] {self.validator.id}: Not enough prepare votes, waiting...")
            await asyncio.sleep(ConsensusConfig.PREPARE_TIMEOUT)
            vote_count = len(self.prepare_log.get(block.hash, set()))
        
        return vote_count >= self.quorum
    
    async def commit_phase(self, block: Block) -> bool:
        print(f"[PBFT] {self.validator.id}: Commit phase for block #{block.index}")
        
        msg = PBFTMessage(
            "COMMIT",
            {
                "block_hash": block.hash,
                "view": self.view
            },
            self.validator.public_key
        )
        
        if block.hash not in self.commit_log:
            self.commit_log[block.hash] = set()
        
        self.commit_log[block.hash].add(self.validator.public_key)
        
        commit_count = len(self.commit_log[block.hash])
        
        print(f"[PBFT] {self.validator.id}: Commit votes: {commit_count}/{self.n}")
        
        if commit_count < self.quorum:
            await asyncio.sleep(ConsensusConfig.COMMIT_TIMEOUT)
            commit_count = len(self.commit_log.get(block.hash, set()))
        
        return commit_count >= self.quorum
    
    async def run_consensus(self, block: Block) -> bool:
        print(f"\n[PBFT] Starting consensus for block #{block.index}")
        print(f"[PBFT] View: {self.view}, Round: {self.round}")
        
        if not await self.pre_prepare_phase(block):
            print(f"[PBFT] ✗ Pre-prepare failed")
            return False
        
        if not await self.prepare_phase(block):
            print(f"[PBFT] ✗ Prepare failed")
            return False
        
        if not await self.commit_phase(block):
            print(f"[PBFT] ✗ Commit failed")
            return False
        
        print(f"[PBFT] ✓ Block #{block.index} FINALIZED!")
        self.round += 1
        return True
    
    async def _wait_pre_prepare(self, block: Block) -> bool:
        """Wait for pre-prepare message from primary."""
        start = time.time()
        timeout = ConsensusConfig.PRE_PREPARE_TIMEOUT
        
        while time.time() - start < timeout:
            if block.index in self.pre_prepare_log:
                return True
            await asyncio.sleep(0.1)
        
        return False

class AdaptiveQuorum:
    
    @staticmethod
    def calculate(priority: float, validator_scores: List[float]) -> float:
        total_score = sum(validator_scores)
        
        if priority > 0.8:
            quorum_pct = ConsensusConfig.QUORUM_EMERGENCY
        elif priority > 0.3:
            quorum_pct = ConsensusConfig.QUORUM_NORMAL
        else:
            quorum_pct = ConsensusConfig.QUORUM_ROUTINE
        
        return quorum_pct * total_score
