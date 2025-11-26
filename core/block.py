import time
import json
from typing import List, Dict, Any, Optional
from core.transaction import Transaction
from utils.crypto import sha256_hash
from utils.hashing import merkle_root

class Block:
    
    def __init__(self,
                 index: int,
                 previous_hash: str,
                 transactions: List[Transaction],
                 proposer: str,
                 timestamp: Optional[float] = None):
        self.index = index
        self.previous_hash = previous_hash
        self.transactions = transactions
        self.proposer = proposer
        self.timestamp = timestamp or time.time()
        
        self.merkle_root = self._calculate_merkle_root()
        self.hash = self._calculate_hash()
    
    def _calculate_merkle_root(self) -> str:
        if not self.transactions:
            return sha256_hash("empty")
        
        tx_dicts = [tx.to_dict() for tx in self.transactions]
        return merkle_root(tx_dicts)
    
    def _calculate_hash(self) -> str:
        block_data = {
            "index": self.index,
            "previous_hash": self.previous_hash,
            "merkle_root": self.merkle_root,
            "timestamp": self.timestamp,
            "proposer": self.proposer
        }
        block_json = json.dumps(block_data, sort_keys=True)
        return sha256_hash(block_json)
    
    def verify_integrity(self) -> bool:
        recalculated_hash = self._calculate_hash()
        return recalculated_hash == self.hash
    
    def verify_transactions(self) -> bool:
        for tx in self.transactions:
            if not tx.verify_signature():
                return False
        return True
    
    def verify_all(self) -> bool:
        return self.verify_integrity() and self.verify_transactions()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "hash": self.hash,
            "previous_hash": self.previous_hash,
            "merkle_root": self.merkle_root,
            "timestamp": self.timestamp,
            "proposer": self.proposer,
            "transactions": [tx.to_dict() for tx in self.transactions]
        }
    
    def __repr__(self) -> str:
        status = "✓" if self.verify_integrity() else "✗"
        return (f"Block#{self.index} {status} | "
                f"Hash: {self.hash[:8]}... | "
                f"Txs: {len(self.transactions)} | "
                f"Proposer: {self.proposer[:8]}...")