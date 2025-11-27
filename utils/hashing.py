from typing import List, Union, Dict, Tuple
from utils.crypto import sha256_hash

def merkle_root(transaction: List) -> str:
    if not transaction:
        return sha256_hash("")
    
    current_level = [sha256_hash(tx) for tx in transaction]
    
    while len(current_level) > 1:
        next_level = []
        i = 0
        while i < len(current_level):
            left_hash = current_level[i]
            right_hash = current_level[i + 1] if i + 1 < len(current_level) else left_hash
            combined_hash = left_hash + right_hash
            new_parent_hash = sha256_hash(combined_hash)
            next_level.append(new_parent_hash)
            i += 2
        
        current_level = next_level
    
    return current_level[0]

def merkle_proof(transactions: List[Union[str, Dict]], target_tx: Union[str, Dict]) -> List[Tuple[str, str]]:

    if not transactions:
        return []
    
    current_level = [sha256_hash(tx) for tx in transactions]
    target_hash = sha256_hash(target_tx)
    
    try:
        index = current_level.index(target_hash)
    except ValueError:
        return []  
    
    proof = []
    
    while len(current_level) > 1:
        next_level = []
        i = 0
        new_index = None
        
        while i < len(current_level):
            left_hash = current_level[i]
            right_hash = current_level[i + 1] if i + 1 < len(current_level) else left_hash
            combined_hash = left_hash + right_hash
            new_parent_hash = sha256_hash(combined_hash)
            next_level.append(new_parent_hash)
            
            if i == index:
                if i + 1 < len(current_level):
                    proof.append((right_hash, "right")) 
                new_index = len(next_level) - 1
            elif i + 1 == index:
                proof.append((left_hash, "left"))  
                new_index = len(next_level) - 1
            
            i += 2
        
        current_level = next_level
        index = new_index
    
    return proof

def verify_merkle_proof(target_tx: Union[str, Dict], proof: List[Tuple[str, str]], merkle_root_hash: str) -> bool:
 
    current_hash = sha256_hash(target_tx)
    
    for sibling_hash, position in proof:
        if position == "left":
            combined_hash = sibling_hash + current_hash
        else:  
            combined_hash = current_hash + sibling_hash
        
        current_hash = sha256_hash(combined_hash)
    
    return current_hash == merkle_root_hash
