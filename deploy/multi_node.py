import asyncio
import sys
from typing import List, Dict, Optional
from network.node import NetworkNode
from consensus.validator import Validator
from config import Tier, ValidatorConfig

class NetworkCluster:
    
    def __init__(self, cluster_name: str):
        self.cluster_name = cluster_name
        self.nodes: Dict[str, NetworkNode] = {}
        self.validators: List[Validator] = []
    
    def create_validators(self):
        """Create validators for the network."""
        print(f"\n[{self.cluster_name}] Creating validators...\n")
        
        for i in range(ValidatorConfig.TIER_1_COUNT):
            v = Validator(f"T1_Node_{i}", 500, Tier.TIER_1)
            self.validators.append(v)
            print(f"  ✓ {v}")
        
        for i in range(ValidatorConfig.TIER_2_COUNT):
            v = Validator(f"T2_Node_{i}", 300, Tier.TIER_2)
            self.validators.append(v)
            print(f"  ✓ {v}")
        
        for i in range(ValidatorConfig.TIER_3_COUNT):
            v = Validator(f"T3_Node_{i}", 100, Tier.TIER_3)
            self.validators.append(v)
            print(f"  ✓ {v}")
    
    def create_nodes(self, 
                     base_port: int = 8000,
                     node_count: Optional[int] = None) -> List[NetworkNode]:
        if node_count is None:
            node_count = len(self.validators)
        
        print(f"\n[{self.cluster_name}] Creating {node_count} nodes...\n")
        
        nodes = []
        for i, validator in enumerate(self.validators[:node_count]):
            node = NetworkNode(
                node_id=f"Node_{i}",
                host="localhost",
                port=base_port + i,
                validator=validator
            )
            
            self.nodes[f"Node_{i}"] = node
            nodes.append(node)
            print(f"  ✓ Node_{i} on port {base_port + i}")
        
        return nodes
    
    def connect_peers(self):
        """Connect nodes to each other."""
        print(f"\n[{self.cluster_name}] Connecting peers...\n")
        
        nodes_list = list(self.nodes.values())
        base_port = 8000
        
        for i, node in enumerate(nodes_list):
            for j, other_node in enumerate(nodes_list):
                if i != j:
                    node.add_peer(
                        peer_id=other_node.node_id,
                        host=other_node.host,
                        port=other_node.port,
                        public_key=other_node.validator.public_key
                    )
            
            print(f"  ✓ {node.node_id} connected to {len(node.peers)} peers")
    
    async def run_all_nodes(self):
        """Run all nodes concurrently."""
        print(f"\n[{self.cluster_name}] Starting all nodes...\n")
        
        tasks = []
        for node in self.nodes.values():
            task = asyncio.create_task(node.run())
            tasks.append(task)
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            print(f"\n[{self.cluster_name}] Stopping all nodes...")
            for task in tasks:
                task.cancel()
    
    async def monitor_nodes(self, interval: int = 5):
        """Monitor and print node statistics."""
        print(f"\n[{self.cluster_name}] Starting monitor (interval: {interval}s)...\n")
        
        while True:
            try:
                print("\n" + "=" * 80)
                print(f"[{self.cluster_name}] Network Status")
                print("=" * 80)
                
                for node in self.nodes.values():
                    stats = node.get_stats()
                    print(f"\n{node}")
                    for key, value in stats.items():
                        if key != "node_id":
                            print(f"  {key}: {value}")
                
                total_blocks = sum(len(node.blockchain.chain) for node in self.nodes.values())
                total_txs = sum(node.txs_processed for node in self.nodes.values())
                print(f"\nCluster: Total blocks={total_blocks}, Total txs={total_txs}")
                
                await asyncio.sleep(interval)
            
            except KeyboardInterrupt:
                break
    
    async def start(self, monitor: bool = True):
        """Start the entire cluster."""
        print(f"\n{'=' * 80}")
        print(f"🔗 BLOCKCHAIN CLUSTER: {self.cluster_name}")
        print(f"{'=' * 80}\n")
        
        self.create_validators()
        self.create_nodes()
        self.connect_peers()
        
        if monitor:
            node_task = asyncio.create_task(self.run_all_nodes())
            monitor_task = asyncio.create_task(self.monitor_nodes(interval=5))
            
            await asyncio.gather(node_task, monitor_task)
        else:
            await self.run_all_nodes()

class MultiMachineSetup:
    
    @staticmethod
    def get_node_config(machine_id: int) -> Dict:
        """Get configuration for a specific machine."""
        configs = {
            0: {
                "nodes": 3,
                "base_port": 8000,
                "validators": [0, 1, 2]
            },
            1: {
                "nodes": 3,
                "base_port": 8000,
                "validators": [3, 4, 5]
            },
            2: {
                "nodes": 2,
                "base_port": 8000,
                "validators": [6, 7]
            }
        }
        
        return configs.get(machine_id, configs[0])

async def run_single_machine_cluster():
    """Run a 7-node cluster on single machine (for testing)."""
    cluster = NetworkCluster("SingleMachine-7Nodes")
    await cluster.start(monitor=True)

async def run_multi_machine_node(machine_id: int):
    config = MultiMachineSetup.get_node_config(machine_id)
    
    cluster = NetworkCluster(f"Machine-{machine_id}")
    cluster.create_validators()
    
    validators_for_machine = [cluster.validators[i] for i in config["validators"]]
    
    nodes = []
    for i, validator in enumerate(validators_for_machine):
        node = NetworkNode(
            node_id=f"Machine{machine_id}_Node{i}",
            host="0.0.0.0",
            port=config["base_port"] + i,
            validator=validator
        )
        cluster.nodes[node.node_id] = node
        nodes.append(node)
    
    for node in nodes:
        print(f"Node {node.node_id} listening on :{node.port}")
    
    await cluster.run_all_nodes()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        machine_id = int(sys.argv[1])
        asyncio.run(run_multi_machine_node(machine_id))
    else:
        asyncio.run(run_single_machine_cluster())
