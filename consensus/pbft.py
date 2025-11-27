
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
        self.type = msg_type  # PRE_PREPARE, PREPARE, COMMIT
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
    """
    PBFT Consensus Engine.
    
    Tolerates up to f Byzantine validators where n = 3f + 1
    For 7 validators: f = 2 (tolerate 2 malicious)
    """
    
    def __init__(self, validator: Validator, all_validators: List[Validator]):
        """Initialize PBFT consensus."""
        self.validator = validator
        self.all_validators = all_validators
        self.n = len(all_validators)
        self.f = (self.n - 1) // 3  # Byzantine fault tolerance
        self.quorum = 2 * self.f + 1
        
        # State tracking
        self.round = 0
        self.view = 0
        self.current_block: Optional[Block] = None
        
        # Message logs
        self.pre_prepare_log: Dict[int, PBFTMessage] = {}
        self.prepare_log: Dict[str, Set[str]] = {}  # block_hash -> set of validators
        self.commit_log: Dict[str, Set[str]] = {}   # block_hash -> set of validators
        
        # Timers for timeout
        self.timers: Dict[str, asyncio.Task] = {}
    
    def select_primary(self) -> Validator:
        """Select primary validator for this view (round-robin)."""
        primary_idx = self.view % self.n
        return self.all_validators[primary_idx]
    
    async def pre_prepare_phase(self, block: Block) -> bool:
        """
        Phase 1: Primary proposes block.
        
        Returns True if quorum accepts block.
        """
        primary = self.select_primary()
        
        if self.validator.public_key != primary.public_key:
            # We're not primary, wait for message
            return await self._wait_pre_prepare(block)
        
        # We ARE primary, broadcast proposal
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
        """
        Phase 2: All validators vote on block.
        
        Need 2f + 1 votes (including self).
        """
        print(f"[PBFT] {self.validator.id}: Prepare phase for block #{block.index}")
        
        # Validate block
        if not block.verify_all():
            print(f"[PBFT] {self.validator.id}: Block validation failed!")
            return False
        
        # Vote
        msg = PBFTMessage(
            "PREPARE",
            {
                "block_hash": block.hash,
                "view": self.view
            },
            self.validator.public_key
        )
        
        # Track votes
        if block.hash not in self.prepare_log:
            self.prepare_log[block.hash] = set()
        
        self.prepare_log[block.hash].add(self.validator.public_key)
        
        # Simulate collecting votes from other validators
        # In real network, messages would arrive
        vote_count = len(self.prepare_log[block.hash])
        
        print(f"[PBFT] {self.validator.id}: Prepare votes: {vote_count}/{self.n}")
        
        # Need quorum to proceed
        if vote_count < self.quorum:
            print(f"[PBFT] {self.validator.id}: Not enough prepare votes, waiting...")
            # In real system, wait for messages
            await asyncio.sleep(ConsensusConfig.PREPARE_TIMEOUT)
            vote_count = len(self.prepare_log.get(block.hash, set()))
        
        return vote_count >= self.quorum
    
    async def commit_phase(self, block: Block) -> bool:
        """
        Phase 3: Validators commit block.
        
        Once 2f + 1 commits, block is final.
        """
        print(f"[PBFT] {self.validator.id}: Commit phase for block #{block.index}")
        
        # Commit vote
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
        
        # Need quorum
        if commit_count < self.quorum:
            await asyncio.sleep(ConsensusConfig.COMMIT_TIMEOUT)
            commit_count = len(self.commit_log.get(block.hash, set()))
        
        return commit_count >= self.quorum
    
    async def run_consensus(self, block: Block) -> bool:
        """
        Run full PBFT consensus.
        
        Returns True if block is finalized.
        """
        print(f"\n[PBFT] Starting consensus for block #{block.index}")
        print(f"[PBFT] View: {self.view}, Round: {self.round}")
        
        # Phase 1: Pre-prepare
        if not await self.pre_prepare_phase(block):
            print(f"[PBFT] ✗ Pre-prepare failed")
            return False
        
        # Phase 2: Prepare
        if not await self.prepare_phase(block):
            print(f"[PBFT] ✗ Prepare failed")
            return False
        
        # Phase 3: Commit
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
            # In real system, check message queue
            if block.index in self.pre_prepare_log:
                return True
            await asyncio.sleep(0.1)
        
        return False


class AdaptiveQuorum:
    """
    Adaptive quorum calculation based on transaction priority.
    
    CAMTC Innovation: Higher priority = lower quorum needed
    """
    
    @staticmethod
    def calculate(priority: float, validator_scores: List[float]) -> float:
        """
        Calculate required quorum score.
        
        Args:
            priority: Transaction priority (0-1)
            validator_scores: List of validator scores
        
        Returns:
            Required total score for quorum
        """
        total_score = sum(validator_scores)
        
        # Adaptive quorum based on priority
        if priority > 0.8:  # Emergency
            quorum_pct = ConsensusConfig.QUORUM_EMERGENCY
        elif priority > 0.3:  # Normal
            quorum_pct = ConsensusConfig.QUORUM_NORMAL
        else:  # Routine
            quorum_pct = ConsensusConfig.QUORUM_ROUTINE
        
        return quorum_pct * total_score
 