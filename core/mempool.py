from typing import List, Optional, Dict, Any
from core.transaction import Transaction
from config import PriorityConfig

class Mempool:
    
    def __init__(self):
        self.transactions: List[Transaction] = []
        self.tx_by_id: Dict[str, Transaction] = {}
        
    def calculate_priority(self, tx: Transaction) -> float:
        
        severity = tx.data.get('severity', 0.5)
        urgency = tx.data.get('urgency', 0.5)
        risk = tx.data.get('risk', 0.5)
        
        resource_avail = 0.8 
        
        alpha = PriorityConfig.ALPHA
        beta = PriorityConfig.BETA
        gamma = PriorityConfig.GAMMA
        delta = PriorityConfig.DELTA

        priority = (
            alpha * severity + 
            beta * urgency + 
            gamma * resource_avail + 
            delta * risk
        )

        return max(0.0, min(1.0, priority))
    
    def _sort_by_priority(self):
        
        for tx in self.transactions:
            if not hasattr(tx, 'priority'):
                tx.priority = self.calculate_priority(tx)

        self.transactions.sort(key=lambda tx: tx.priority, reverse=True)
        
    def add_transaction(self, tx: Transaction) -> bool:
        tx_id = tx.id
    
        if tx_id in self.tx_by_id:
            return False
        
        tx.priority = self.calculate_priority(tx)
        self.transactions.append(tx)
        self.transactions.sort(key=lambda x: x.priority, reverse=True)
        
        self.tx_by_id[tx_id] = tx
        
        return True
        
    def get_transactions(self, count: int = 10) -> List[Transaction]:
        
        top_txs = self.transactions[:count]
        
        self.transactions = self.transactions[count:]
        
        for tx in top_txs:
            del self.tx_by_id[tx.id]
            
        return top_txs
    
    def peek_transactions(self, count: int = 10) -> List[Transaction]:
        
        return self.transactions[:count]
        
    def remove_transaction(self, tx_id: str) -> bool:
        
        if tx_id not in self.tx_by_id:
            return False
            
        tx_to_remove = self.tx_by_id[tx_id]
        
        del self.tx_by_id[tx_id]
        
        self.transactions.remove(tx_to_remove)

        return True
        
    def size(self) -> int:
        return len(self.transactions)
        
    def get_transaction(self, tx_id: str) -> Optional[Transaction]:
        return self.tx_by_id.get(tx_id)
        
    def clear(self):
        self.transactions.clear()
        self.tx_by_id.clear()
        
    def get_stats(self) -> Dict[str, Any]:
        if not self.transactions:
            return {"size": 0, "avg_priority": 0}
            
        priorities = [tx.priority for tx in self.transactions if hasattr(tx, 'priority')]
        avg_priority = sum(priorities) / len(priorities) if priorities else 0
        
        return {
            "size": len(self.transactions),
            "avg_priority": round(avg_priority, 3),
            "max_priority": round(max(priorities), 3) if priorities else 0,
            "min_priority": round(min(priorities), 3) if priorities else 0
        }
        
    def __repr__(self) -> str:
        stats = self.get_stats()
        return f"Mempool(size={stats['size']}, avg_priority={stats['avg_priority']})"
    