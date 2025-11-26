
import time
from typing import Optional, Dict, Any
from utils.crypto import generate_keypair, sha256_hash
from config import ValidatorScoringConfig, Tier, TIER_CERTIFICATION, TIER_LATENCY_TARGET

class Validator:
    
    def __init__(self, validator_id: str, stake: float, tier: Tier):
        self.id = validator_id
        self.stake = stake
        self.tier = tier
        
        self.public_key, self.private_key = generate_keypair()
        
        self.reputation = 1.0
        self.blocks_validated = 0
        self.correct_votes = 0
        self.incorrect_votes = 0
        self.last_active = time.time()
        
        self.latency_ms = TIER_LATENCY_TARGET[tier]
        self.uptime = 0.99
        
        self.certification = TIER_CERTIFICATION[tier]
    
    def update_reputation(self, performance: float):
        decay = 0.7
        self.reputation = decay * self.reputation + (1 - decay) * performance
    
    def record_vote(self, correct: bool):
        if correct:
            self.correct_votes += 1
            self.update_reputation(1.0)
        else:
            self.incorrect_votes += 1
            self.update_reputation(0.0)
        
        self.blocks_validated += 1
        self.last_active = time.time()
    
    def calculate_v_score(self) -> float:
        max_stake = 1_000_000
        stake_norm = min(self.stake / max_stake, 1.0)
        
        latency_inv = 1.0 / (1 + self.latency_ms)
        
        w = ValidatorScoringConfig
        v_score = (
            w.W_STAKE * stake_norm +
            w.W_REPUTATION * self.reputation +
            w.W_LATENCY * latency_inv +
            w.W_CERTIFICATION * self.certification +
            w.W_UPTIME * self.uptime
        )
        
        return min(max(v_score, 0.0), 1.0)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "public_key": self.public_key,
            "stake": self.stake,
            "tier": self.tier.name,
            "reputation": round(self.reputation, 3),
            "v_score": round(self.calculate_v_score(), 3),
            "blocks_validated": self.blocks_validated,
            "correct_votes": self.correct_votes,
            "incorrect_votes": self.incorrect_votes,
            "uptime": round(self.uptime, 3),
            "latency_ms": self.latency_ms
        }
    
    def __repr__(self) -> str:
        v_score = self.calculate_v_score()
        return (f"Validator({self.id} | "
                f"Tier={self.tier.name} | "
                f"Stake={self.stake} | "
                f"Rep={self.reputation:.2f} | "
                f"Score={v_score:.2f})")