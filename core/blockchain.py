from typing import List, Optional, Dict, Any
from core.block import Block
from core.transaction import Transaction
from core.mempool import Mempool
from config import BlockchainConfig, ValidatorConfig

class Blockchain:

    def __init__(self):
        self.chain: List[Block] = []
        self.mempool = Mempool()
        self.create_genesis_block()
    
    def create_genesis_block(self):
        genesis = Block(
            index=0,
            previous_hash=BlockchainConfig.GENESIS_HASH,
            transactions=[],
            proposer="system"
        )
        self.chain.append(genesis)
    
    def get_latest_block(self) -> Block:
        return self.chain[-1]
    
    def add_transaction(self, tx: Transaction) -> bool:
        if not tx.verify_signature():
            return False
        return self.mempool.add_transaction(tx)
    
    def mine_block(self, proposer: str, tx_count: int = 10) -> Optional[Block]:
        last_block = self.get_latest_block()
        
        transactions = self.mempool.get_transactions(tx_count)
        
        if not transactions:
            return None
        
        new_block = Block(
            index=last_block.index + 1,
            previous_hash=last_block.hash,
            transactions=transactions,
            proposer=proposer
        )
        
        return new_block
    
    def add_block(self, block: Block) -> bool:
        last_block = self.get_latest_block()
        
        if block.previous_hash != last_block.hash:
            print(f"❌ Block references wrong previous hash!")
            return False
        
        if block.index != last_block.index + 1:
            print(f"❌ Block index incorrect!")
            return False
        
        if not block.verify_integrity():
            print(f"❌ Block has been tampered with!")
            return False
        
        if not block.verify_transactions():
            print(f"❌ Block contains invalid transactions!")
            return False
        
        self.chain.append(block)
        return True
    
    def is_chain_valid(self) -> bool:
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]
            
            if not current.verify_integrity():
                print(f"❌ Block #{i} has invalid hash!")
                return False
            
            if current.previous_hash != previous.hash:
                print(f"❌ Chain broken at block #{i}!")
                return False
        
        return True
    
    def get_chain(self) -> List[Dict]:
        return [block.to_dict() for block in self.chain]
    
    def get_block(self, index: int) -> Optional[Block]:
        if 0 <= index < len(self.chain):
            return self.chain[index]
        return None
    
    def get_transaction(self, tx_id: str) -> Optional[Transaction]:
        for block in self.chain:
            for tx in block.transactions:
                if tx.id == tx_id:
                    return tx
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        total_txs = sum(len(block.transactions) for block in self.chain)
        return {
            "length": len(self.chain),
            "total_transactions": total_txs,
            "mempool_size": self.mempool.size(),
            "is_valid": self.is_chain_valid()
        }
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return (f"Blockchain(Length={stats['length']}, "
                f"Txs={stats['total_transactions']}, "
                f"Valid={stats['is_valid']})")
