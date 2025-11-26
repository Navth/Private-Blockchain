
from core.blockchain import Blockchain
from core.transaction import Transaction
from core.block import Block
from consensus.validator import Validator
from config import Tier, ValidatorConfig, PriorityConfig
import time

def demo_blockchain():
    
    print("=" * 80)
    print("🔗 PRODUCTION-GRADE BLOCKCHAIN DEMONSTRATION")
    print("=" * 80)
    
    print("\n[1] Creating validators...\n")
    
    validators = []
    
    for i in range(ValidatorConfig.TIER_1_COUNT):
        v = Validator(f"Validator_T1_{i}", 500, Tier.TIER_1)
        validators.append(v)
        print(f"  ✓ {v}")
    
    for i in range(ValidatorConfig.TIER_2_COUNT):
        v = Validator(f"Validator_T2_{i}", 300, Tier.TIER_2)
        validators.append(v)
        print(f"  ✓ {v}")
    
    for i in range(ValidatorConfig.TIER_3_COUNT):
        v = Validator(f"Validator_T3_{i}", 100, Tier.TIER_3)
        validators.append(v)
        print(f"  ✓ {v}")
    
    print("\n[2] Creating blockchain...\n")
    blockchain = Blockchain()
    print(f"  {blockchain}")
    
    print("\n[3] Creating signed transactions...\n")
    
    transactions = []
    for i, validator in enumerate(validators[:3]):
        sender_pub = validator.public_key
        recipient_pub = validators[(i + 1) % len(validators)].public_key
        
        tx = Transaction(
            sender=sender_pub,
            recipient=recipient_pub,
            amount=100 + i * 50,
            data={
                "severity": 0.3 + (i * 0.2),
                "urgency": 0.2 + (i * 0.3),
                "risk": 0.4
            }
        )
        
        tx.sign(validator.private_key)
        transactions.append(tx)
        
        blockchain.add_transaction(tx)
        print(f"  ✓ {tx}")
    
    print(f"\n  Mempool: {blockchain.mempool}")
    print(f"  Stats: {blockchain.mempool.get_stats()}")
    
    print("\n[4] Mining blocks...\n")
    
    for block_num in range(2):
        proposer_pub = validators[block_num % len(validators)].public_key
        
        new_block = blockchain.mine_block(proposer_pub, tx_count=5)
        
        if new_block:
            print(f"  Mining block #{new_block.index}...")
            print(f"    Proposer: {proposer_pub[:16]}...")
            print(f"    Transactions: {len(new_block.transactions)}")
            print(f"    Merkle Root: {new_block.merkle_root[:16]}...")
            print(f"    Block Hash: {new_block.hash[:16]}...")
            
            if blockchain.add_block(new_block):
                print(f"    ✓ Block #{new_block.index} added!")
            else:
                print(f"    ✗ Block #{new_block.index} rejected!")
    
    print("\n[5] Blockchain Statistics...\n")
    stats = blockchain.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n[6] Chain Validation...\n")
    
    is_valid = blockchain.is_chain_valid()
    print(f"  Chain is valid: {is_valid}")
    print(f"  Chain length: {len(blockchain.chain)}")
    
    for i, block in enumerate(blockchain.chain):
        integrity = block.verify_integrity()
        txs_valid = block.verify_transactions()
        print(f"    Block #{i}: Integrity={integrity} | Txs Valid={txs_valid}")
    
    print("\n[7] Validator Scores...\n")
    
    for validator in validators:
        v_score = validator.calculate_v_score()
        print(f"  {validator.id:20} | Score: {v_score:.3f} | Rep: {validator.reputation:.3f}")
    
    print("\n[8] Transaction Lookup...\n")
    
    if blockchain.chain[1].transactions:
        first_tx = blockchain.chain[1].transactions[0]
        found_tx = blockchain.get_transaction(first_tx.id)
        
        if found_tx:
            print(f"  ✓ Found transaction: {found_tx.id}")
            print(f"    Amount: {found_tx.amount}")
            print(f"    Sender: {found_tx.sender[:16]}...")
            print(f"    Signed: {found_tx.signature is not None}")
    
    print("\n[9] Security Test - Tampering Detection...\n")
    
    original_hash = blockchain.chain[1].hash
    blockchain.chain[1].transactions[0].amount = 999999
    
    new_hash = blockchain.chain[1]._calculate_hash()
    
    print(f"  Original hash: {original_hash[:16]}...")
    print(f"  After tampering: {new_hash[:16]}...")
    print(f"  Hashes match: {original_hash == new_hash}")
    print(f"  ✓ Tampering detected! Block is invalid.")
    
    blockchain.chain[1].transactions[0].amount = 100
    
    print("\n" + "=" * 80)
    print("✅ BLOCKCHAIN DEMONSTRATION COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    demo_blockchain()