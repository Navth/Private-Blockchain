import { motion } from "framer-motion";
import { Activity, Play, Plus, RefreshCw } from "lucide-react";
import { useRef, useState } from "react";

import { DomainSwitcher } from "./components/DomainSwitcher";
import { NetworkGraph } from "./components/NetworkGraph";
import { BlockchainTimeline } from "./components/BlockchainTimeline";
import { BlockDetails } from "./components/BlockDetails";
import { buildDomainPayload } from "./data/domains";
import { SimulationLayer } from "./services/SimulationLayer";
import type {
  BlockSummary,
  ConsensusPhase,
  DomainKey,
  DomainPayload,
  NetworkNode,
  NodeStatus,
  SimulationMessage,
  SimulationStep,
} from "./types/blockchain";

type NodeBlueprint = Pick<NetworkNode, "id" | "label" | "position">;

const POSITION_PRESETS: Array<NodeBlueprint["position"]> = [
  { x: 18, y: 24 },
  { x: 52, y: 15 },
  { x: 82, y: 28 },
  { x: 26, y: 74 },
  { x: 78, y: 68 },
  { x: 12, y: 52 },
  { x: 50, y: 86 },
  { x: 88, y: 52 },
  { x: 12, y: 32 },
  { x: 88, y: 32 },
];

const INITIAL_BLUEPRINTS: NodeBlueprint[] = [
  { id: "node-nyc", label: "Node NYC", position: POSITION_PRESETS[0]! },
  { id: "node-zrh", label: "Node Zurich", position: POSITION_PRESETS[1]! },
  { id: "node-sgp", label: "Node Singapore", position: POSITION_PRESETS[2]! },
  { id: "node-blr", label: "Node Bengaluru", position: POSITION_PRESETS[3]! },
  { id: "node-sao", label: "Node São Paulo", position: POSITION_PRESETS[4]! },
];

const buildNodes = (
  blueprints: NodeBlueprint[],
  leaderId: string,
  statusMap: Record<string, NodeStatus> = {},
): NetworkNode[] =>
  blueprints.map((blueprint) => ({
    id: blueprint.id,
    label: blueprint.label,
    position: blueprint.position,
    role: blueprint.id === leaderId ? "leader" : "validator",
    status: statusMap[blueprint.id] ?? "idle",
  }));

const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);

const generateFallbackPosition = (index: number): NodeBlueprint["position"] => {
  const angle = ((index * 137.5) * Math.PI) / 180;
  const radius = 28 + (index % 5) * 7;
  const x = 50 + radius * Math.cos(angle);
  const y = 50 + radius * Math.sin(angle);
  return {
    x: clamp(Math.round(x), 10, 90),
    y: clamp(Math.round(y), 10, 90),
  };
};

const createBlueprint = (existing: NodeBlueprint[]): NodeBlueprint => {
  const index = existing.length;
  const position = POSITION_PRESETS[index] ?? generateFallbackPosition(index);
  const baseId = `node-${index + 1}`;
  let candidateId = baseId;
  let attempt = 1;
  while (existing.some((bp) => bp.id === candidateId)) {
    candidateId = `${baseId}-${attempt}`;
    attempt += 1;
  }

  return {
    id: candidateId,
    label: `Node ${index + 1}`,
    position,
  };
};

const genesisBlock: BlockSummary = {
  height: 0,
  hash: "0xGENESIS",
  previousHash: "0x0",
  timestamp: new Date(Date.now() - 1000 * 60 * 5).toISOString(),
  domain: "supply-chain",
  payload: { status: "Network bootstrapped" },
  proposer: "controller",
  votes: [],
  latencyMs: 0,
  fallback: true,
};

function App() {
  const simulationLayerRef = useRef(new SimulationLayer());

  const initialLeaderId = INITIAL_BLUEPRINTS[0]?.id ?? "node-nyc";
  const [nodeBlueprints, setNodeBlueprints] = useState<NodeBlueprint[]>(INITIAL_BLUEPRINTS);
  const [domain, setDomain] = useState<DomainKey>("supply-chain");
  const [leaderId, setLeaderId] = useState<string>(initialLeaderId);
  const [nodes, setNodes] = useState<NetworkNode[]>(() => buildNodes(INITIAL_BLUEPRINTS, initialLeaderId));
  const [messages, setMessages] = useState<SimulationMessage[]>([]);
  const [blocks, setBlocks] = useState<BlockSummary[]>([genesisBlock]);
  const [selectedBlock, setSelectedBlock] = useState<BlockSummary | null>(genesisBlock);
  const [activePhase, setActivePhase] = useState<ConsensusPhase>("idle");
  const [currentStep, setCurrentStep] = useState<SimulationStep | null>(null);
  const [statusCopy, setStatusCopy] = useState<string>("Ready to simulate consensus.");
  const [payloadPreview, setPayloadPreview] = useState<DomainPayload | null>(null);
  const [isSimulating, setIsSimulating] = useState(false);

  const nextLeaderIndexRef = useRef(1);

  const handleSimulate = async () => {
    if (isSimulating || nodeBlueprints.length === 0) {
      return;
    }

    const leaderIndex = nextLeaderIndexRef.current % nodeBlueprints.length;
    nextLeaderIndexRef.current = (leaderIndex + 1) % nodeBlueprints.length;

    const leaderBlueprint = nodeBlueprints[leaderIndex] ?? nodeBlueprints[0];
    const validatorIds = nodeBlueprints
      .filter((blueprint) => blueprint.id !== leaderBlueprint.id)
      .map((blueprint) => blueprint.id);

    if (!leaderBlueprint) {
      return;
    }

    const payload = buildDomainPayload(domain);
    setPayloadPreview(payload);

    const previousBlock = blocks[blocks.length - 1];
    const context = {
      domain,
      leaderId: leaderBlueprint.id,
      validatorIds,
      previousHash: previousBlock?.hash ?? "0xGENESIS",
      height: (previousBlock?.height ?? 0) + 1,
      payload,
    } satisfies Parameters<SimulationLayer["execute"]>[0];

    setLeaderId(leaderBlueprint.id);
    setIsSimulating(true);
    setActivePhase("proposal");
    setStatusCopy(`${leaderBlueprint.label} is aggregating transactions.`);
    setCurrentStep(null);
    setNodes(buildNodes(nodeBlueprints, leaderBlueprint.id, { [leaderBlueprint.id]: "validating" }));
    setMessages([]);

    try {
      const result = await simulationLayerRef.current.execute(context, {
        onPhaseChange: (phase, step) => {
          setActivePhase(phase);
          setCurrentStep(step);
          setStatusCopy(step.description);
        },
        onNodeStatus: (statusMap) => {
          setNodes(buildNodes(nodeBlueprints, leaderBlueprint.id, statusMap));
        },
        onMessages: (stepMessages) => {
          setMessages(stepMessages);
        },
      });

      setBlocks((prev) => [...prev, result.block]);
      setSelectedBlock(result.block);
      setStatusCopy(
        result.fallback
          ? "Simulation layer committed the block. Validators are in agreement."
          : "Live endpoint committed the block successfully.",
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : "Simulation failed unexpectedly.";
      setStatusCopy(message);
    } finally {
      setIsSimulating(false);
      setTimeout(() => {
        setMessages([]);
        setNodes(buildNodes(nodeBlueprints, leaderBlueprint.id));
        setActivePhase("idle");
        setCurrentStep(null);
        setPayloadPreview(null);
      }, 800);
    }
  };

  const handleAddNode = () => {
    if (isSimulating) {
      return;
    }

    setNodeBlueprints((prev) => {
      const nextBlueprint = createBlueprint(prev);
      const updated = [...prev, nextBlueprint];
      const statusMap = nodes.reduce<Record<string, NodeStatus>>((acc, node) => {
        acc[node.id] = node.status;
        return acc;
      }, {});
      setNodes(buildNodes(updated, leaderId, statusMap));
      return updated;
    });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-950 to-slate-900 text-slate-100">
      <main className="mx-auto flex min-h-screen max-w-7xl flex-col gap-10 px-6 py-10">
        <header className="space-y-6">
          <div>
            <span className="inline-flex items-center gap-2 rounded-full border border-slate-700/70 bg-slate-900/80 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-slate-300">
              <Activity className="h-3.5 w-3.5" /> Consensus Visualizer
            </span>
            <h1 className="mt-3 text-3xl font-semibold tracking-tight text-slate-50">
              Private Blockchain Control Room
            </h1>
            <p className="mt-2 max-w-2xl text-sm text-slate-400">
              Observe how the simulated leader election, validation votes, and commit broadcasts evolve
              across multiple domains while the SimulationLayer bridges missing backend endpoints.
            </p>
          </div>

          <div className="space-y-3">
            <p className="text-xs uppercase tracking-wide text-slate-400">Domain presets</p>
            <DomainSwitcher value={domain} onChange={setDomain} disabled={isSimulating} />
          </div>
        </header>

        <section className="grid flex-1 gap-8 lg:grid-cols-[1.6fr_1fr]">
          <div className="space-y-6">
            <NetworkGraph
              nodes={nodes}
              activePhase={activePhase}
              currentStep={currentStep}
              messages={messages}
            />

            <motion.div
              layout
              className="rounded-3xl border border-slate-800/80 bg-slate-900/70 p-6"
            >
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="text-xs uppercase tracking-wide text-slate-400">Simulation status</p>
                  <p className="mt-1 text-base font-semibold text-slate-100">{statusCopy}</p>
                </div>
                <div className="flex flex-col items-end gap-2 sm:flex-row">
                  <button
                    type="button"
                    onClick={handleAddNode}
                    disabled={isSimulating}
                    className="flex items-center gap-2 rounded-2xl border border-slate-700/70 bg-slate-800/60 px-3 py-2 text-xs font-semibold text-slate-200 transition hover:border-slate-600/70 hover:bg-slate-800/80 disabled:cursor-not-allowed disabled:border-slate-700/50 disabled:bg-slate-800/50 disabled:text-slate-500"
                    title="Add a validator node"
                  >
                    <Plus className="h-4 w-4" /> Add Node
                  </button>
                  <button
                    type="button"
                    onClick={handleSimulate}
                    disabled={isSimulating || nodeBlueprints.length < 2}
                    className="flex items-center gap-2 rounded-2xl border border-emerald-500/30 bg-emerald-500/10 px-4 py-2 text-sm font-semibold text-emerald-200 transition hover:border-emerald-400/50 hover:bg-emerald-500/20 disabled:cursor-not-allowed disabled:border-slate-700/60 disabled:bg-slate-800/60 disabled:text-slate-500"
                  >
                    <Play className="h-4 w-4" />
                    {isSimulating ? "Running" : "Simulate Consensus"}
                  </button>
                </div>
              </div>

              <div className="mt-4 grid gap-3 sm:grid-cols-2">
                <div className="rounded-2xl border border-slate-800/80 bg-slate-950/50 p-4">
                  <p className="text-xs uppercase tracking-wide text-slate-400">Leader</p>
                  <p className="mt-1 text-lg font-semibold text-slate-100">
                    {nodes.find((node) => node.id === leaderId)?.label ?? leaderId}
                  </p>
                  <p className="mt-1 text-xs text-slate-400">
                    Rotates every run to showcase proposer election.
                  </p>
                </div>

                <div className="rounded-2xl border border-slate-800/80 bg-slate-950/50 p-4">
                  <p className="text-xs uppercase tracking-wide text-slate-400">Active phase</p>
                  <p className="mt-1 text-lg font-semibold capitalize text-slate-100">{activePhase}</p>
                  <p className="mt-1 text-xs text-slate-400">
                    {activePhase === "idle"
                      ? "Awaiting the next simulation round."
                      : currentStep?.description ?? "Progressing through consensus steps."}
                  </p>
                </div>
              </div>

              {payloadPreview && (
                <motion.div
                  layout
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 12 }}
                  className="mt-4 rounded-2xl border border-slate-800/70 bg-slate-950/60 p-4"
                >
                  <div className="flex items-center justify-between">
                    <p className="text-xs uppercase tracking-wide text-slate-400">Payload Preview</p>
                    <RefreshCw className="h-4 w-4 text-slate-500" />
                  </div>
                  <div className="mt-2 grid gap-2 text-xs text-slate-300">
                    {Object.entries(payloadPreview).map(([key, value]) => (
                      <div key={key} className="flex items-center justify-between">
                        <span className="font-medium text-slate-300">{key}</span>
                        <span className="text-slate-200">{value}</span>
                      </div>
                    ))}
                  </div>
                </motion.div>
              )}
            </motion.div>
          </div>

          <aside className="space-y-6">
            <div className="rounded-3xl border border-slate-800/80 bg-slate-900/70 p-6">
              <div className="flex items-center justify-between">
                <p className="text-sm font-semibold text-slate-100">Blockchain</p>
                <span className="text-xs text-slate-500">{blocks.length} blocks</span>
              </div>
              <p className="mt-1 text-xs text-slate-400">
                Click a block to inspect hashes and validator signatures.
              </p>
              <div className="mt-4 max-h-[280px] space-y-3 overflow-y-auto pr-1">
                <BlockchainTimeline
                  blocks={[...blocks].sort((a, b) => b.height - a.height)}
                  selectedHash={selectedBlock?.hash ?? null}
                  onSelect={setSelectedBlock}
                />
              </div>
            </div>

            <BlockDetails block={selectedBlock} />
          </aside>
        </section>
      </main>
    </div>
  );
}

export default App;
