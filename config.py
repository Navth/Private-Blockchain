from enum import Enum
from typing import Dict, List
import os
from dotenv import load_dotenv

load_dotenv()

class NetworkConfig:
    LISTEN_PORT = int(os.getenv("LISTEN_PORT", 8000))
    BOOTSTRAP_PEERS = os.getenv("BOOTSTRAP_PEERS", "").split(",") if os.getenv("BOOTSTRAP_PEERS") else []
    MAX_PEER_CONNECTIONS = 100
    MESSAGE_TIMEOUT = 30
    HEARTBEAT_INTERVAL = 5
    REQUEST_RETRIES = 3

class BlockchainConfig:
    GENESIS_HASH = "0"
    BLOCK_MAX_SIZE = 1024 * 1024
    BLOCK_TIMEOUT = 30
    TRANSACTION_TIMEOUT = 3600
    MAX_TX_PER_BLOCK = 1000
    MIN_TX_PER_BLOCK = 1
    CONFIRMATIONS_REQUIRED = 3

class ValidatorConfig:
    TOTAL_VALIDATORS = 7
    TIER_1_COUNT = 2
    TIER_2_COUNT = 3
    TIER_3_COUNT = 2
    
    MIN_STAKE = 100
    MAX_STAKE = 1_000_000
    
    REPUTATION_DECAY = 0.7
    MIN_REPUTATION = 0.1
    REPUTATION_WINDOW = 100
    
    MAX_VALIDATORS_PER_BLOCK = 20
    MIN_VALIDATORS_REQUIRED = 4

class ConsensusConfig:
    PRE_PREPARE_TIMEOUT = 2
    PREPARE_TIMEOUT = 2
    COMMIT_TIMEOUT = 2
    
    QUORUM_EMERGENCY = 0.51
    QUORUM_NORMAL = 0.67
    QUORUM_ROUTINE = 0.75
    
class PriorityConfig:
    ALPHA = 0.3
    BETA = 0.4
    GAMMA = 0.15
    DELTA = 0.15
    
    EMERGENCY_THRESHOLD = 0.8
    NORMAL_THRESHOLD = 0.3
    
    BLOCK_SIZE_EMERGENCY = 100
    BLOCK_SIZE_NORMAL = 500
    BLOCK_SIZE_ROUTINE = 1000

class ValidatorScoringConfig:
    W_STAKE = 0.25
    W_REPUTATION = 0.35
    W_LATENCY = 0.15
    W_CERTIFICATION = 0.15
    W_UPTIME = 0.1
    
    assert abs(W_STAKE + W_REPUTATION + W_LATENCY + W_CERTIFICATION + W_UPTIME - 1.0) < 0.01, \
        "Validator weights must sum to 1.0"
    
    LATENCY_GOOD = 50
    LATENCY_OK = 200
    LATENCY_BAD = 500

class StorageConfig:
    DB_PATH = os.getenv("DB_PATH", "./blockchain_data")
    CACHE_SIZE = 10000
    BATCH_SIZE = 100
    SYNC_INTERVAL = 5

class LoggingConfig:
    LEVEL = os.getenv("LOG_LEVEL", "INFO")
    FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    ROTATION = "100 MB"
    RETENTION = "10 days"
    LOG_FILE = os.getenv("LOG_FILE", "blockchain.log")

class MetricsConfig:
    ENABLED = os.getenv("METRICS_ENABLED", "true").lower() == "true"
    PROMETHEUS_PORT = 9090
    COLLECTION_INTERVAL = 10

class CryptoConfig:
    SIGNATURE_ALGORITHM = "Ed25519"
    HASH_ALGORITHM = "SHA256"
    NONCE_LENGTH = 32

class IoTConfig:
    SENSOR_INTERVAL = 1
    NUM_SHIPMENTS = 10
    
    NORMAL_TEMP_MIN = 2
    NORMAL_TEMP_MAX = 8
    TEMP_VARIANCE = 0.5

class Tier(Enum):
    TIER_1 = 1
    TIER_2 = 2
    TIER_3 = 3

TIER_CERTIFICATION: Dict[Tier, float] = {
    Tier.TIER_1: 1.0,
    Tier.TIER_2: 0.7,
    Tier.TIER_3: 0.4,
}

TIER_LATENCY_TARGET: Dict[Tier, float] = {
    Tier.TIER_1: 50,
    Tier.TIER_2: 200,
    Tier.TIER_3: 500,
}

class RuntimeFlags:
    ENABLE_CAMTC = True
    ENABLE_ML_OPTIMIZATION = False
    ENABLE_CONCURRENT_CONSENSUS = True
    ENABLE_MERKLE_PROOFS = True
    DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"

class PerformanceTargets:
    TARGET_TPS = 1000
    TARGET_EMERGENCY_LATENCY = 0.5
    TARGET_NORMAL_LATENCY = 5
    TARGET_ROUTINE_LATENCY = 12
    TARGET_FINALITY = 3

if __name__ == "__main__":
    print("=== BLOCKCHAIN CONFIGURATION ===\n")
    print(f"Network Port: {NetworkConfig.LISTEN_PORT}")
    print(f"Validators: {ValidatorConfig.TOTAL_VALIDATORS} "
          f"({ValidatorConfig.TIER_1_COUNT}/{ValidatorConfig.TIER_2_COUNT}/{ValidatorConfig.TIER_3_COUNT})")
    print(f"Storage: {StorageConfig.DB_PATH}")
    print(f"CAMTC Enabled: {RuntimeFlags.ENABLE_CAMTC}")
    print(f"\nPriority Weights: α={PriorityConfig.ALPHA} β={PriorityConfig.BETA} "
          f"γ={PriorityConfig.GAMMA} δ={PriorityConfig.DELTA}")
    print(f"Validator Weights: stake={ValidatorScoringConfig.W_STAKE} "
          f"rep={ValidatorScoringConfig.W_REPUTATION} "
          f"latency={ValidatorScoringConfig.W_LATENCY}")
