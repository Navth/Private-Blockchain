import time
import json
from typing import Dict, Optional, Any
from utils.crypto import sha256_hash, sign_message, verify_signature

class Transaction:
    def __init__(self,
                 sender: str,
                 recipient: str,
                 amount: float,
                 data: Optional[Dict[str, Any]]=None,
                 nonce: int=0):
        self.sender = sender
        self.recipient = recipient
        self.amount = amount
        self.data = data or {}
        self.timestamp = time.time()
        self.nonce = nonce
        self.signature = None 
        self.id = self._calculate_id() 
        self.priority = 0.5
        
    def get_message_data(self) -> str:
        tx_dict = {
            "sender" : self.sender,
            "recipient" : self.recipient,
            "amount" : self.amount,
            "data" : self.data,
            "timestamp" : self.timestamp,
            "nonce" : self.nonce
        }
        return json.dumps(tx_dict, sort_keys=True)
        
    def _calculate_id(self) -> str:
        tx_json_string = self.get_message_data()
        return sha256_hash(tx_json_string) [:16]

    def sign(self, private_key: str) -> str:
        message_data = self.get_message_data()
        
        self.signature = sign_message(message_data, private_key) 
        return self.signature
        
    def verify_signature(self) -> bool:
        if not self.signature:
            return False
            
        message_data = self.get_message_data()
        
        return verify_signature(message_data, self.signature, self.sender) 
        
    def to_dict(self) -> Dict:
        return {
            "sender" : self.sender,
            "recipient" : self.recipient,
            "amount" : self.amount,
            "data" : self.data,
            "timestamp" : self.timestamp,
            "nonce" : self.nonce,
            "signature" : self.signature,
            "id" : self.id
        }
        
    def __repr__(self) -> str:
        return f"Transaction(id={self.id[:8]}..., sender={self.sender[:8]}..., recipient={self.recipient[:8]}..., amount={self.amount})"
