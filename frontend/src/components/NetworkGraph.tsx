import { AnimatePresence, motion } from "framer-motion";
import type {
  ConsensusPhase,
  NetworkNode,
  SimulationMessage,
  SimulationStep,
} from "../types/blockchain";

interface NetworkGraphProps {
  nodes: NetworkNode[];
  activePhase: ConsensusPhase;
  currentStep?: SimulationStep | null;
  messages: SimulationMessage[];
}

const statusClassMap = {
  idle: "border-blue-400/40 bg-blue-500/10 text-blue-100",
  validating: "border-amber-400/60 bg-amber-400/15 text-amber-100",
  committed: "border-emerald-400/60 bg-emerald-400/10 text-emerald-100",
} as const;

const statusLabel: Record<keyof typeof statusClassMap, string> = {
  idle: "Idle",
  validating: "Validating",
  committed: "Committed",
};

export function NetworkGraph({
  nodes,
  activePhase,
  currentStep,
  messages,
}: NetworkGraphProps) {
  const positionMap = new Map(nodes.map((node) => [node.id, node.position]));
  const viewBoxSize = 100;

  const leaderNode = nodes.find((node) => node.role === "leader");
  const edgeSet = new Set<string>();
  const edges: Array<[string, string]> = [];

  const registerEdge = (fromId: string, toId: string) => {
    const sortedKey = [fromId, toId].sort().join("::");
    if (fromId === toId || edgeSet.has(sortedKey)) {
      return;
    }
    edgeSet.add(sortedKey);
    edges.push([fromId, toId]);
  };

  if (leaderNode) {
    nodes
      .filter((node) => node.id !== leaderNode.id)
      .forEach((node) => registerEdge(leaderNode.id, node.id));
  }

  const orderedNodes = [...nodes].sort((a, b) => a.label.localeCompare(b.label));
  for (let index = 0; index < orderedNodes.length; index += 1) {
    const current = orderedNodes[index];
    const next = orderedNodes[(index + 1) % orderedNodes.length];
    if (!current || !next) {
      continue;
    }
    registerEdge(current.id, next.id);
  }

  return (
    <div className="relative h-[360px] overflow-hidden rounded-3xl border border-slate-800/80 bg-slate-950/80 shadow-2xl">
      <svg
        className="absolute inset-0 h-full w-full"
        viewBox={`0 0 ${viewBoxSize} ${viewBoxSize}`}
        preserveAspectRatio="none"
      >
        <defs>
          <radialGradient id="mesh" cx="50%" cy="50%" r="75%">
            <stop offset="0%" stopColor="rgba(30, 64, 175, 0.35)" />
            <stop offset="100%" stopColor="rgba(8, 15, 35, 0.95)" />
          </radialGradient>
        </defs>
        <rect width="100%" height="100%" fill="url(#mesh)" />
        {edges.map(([fromId, toId]) => {
          const source = positionMap.get(fromId);
          const target = positionMap.get(toId);
          if (!source || !target) {
            return null;
          }

          return (
            <line
              key={`${fromId}-${toId}`}
              x1={source.x}
              y1={source.y}
              x2={target.x}
              y2={target.y}
              stroke="rgba(100, 116, 139, 0.35)"
              strokeWidth={1.5}
              strokeDasharray="4 8"
            />
          );
        })}
      </svg>

      <div className="relative h-full w-full">
        {nodes.map((node) => {
          const variantKey = node.status;

          return (
            <motion.div
              key={node.id}
              className={`absolute flex h-24 w-24 flex-col items-center justify-center rounded-2xl border backdrop-blur ${statusClassMap[variantKey]} ${
                node.role === "leader" ? "ring-2 ring-blue-300/70" : ""
              }`}
              style={{
                left: `${node.position.x}%`,
                top: `${node.position.y}%`,
                transform: "translate(-50%, -50%)",
              }}
              animate={{
                boxShadow:
                  variantKey === "committed"
                    ? "0 0 30px rgba(34, 197, 94, 0.55)"
                    : variantKey === "validating"
                    ? "0 0 20px rgba(245, 158, 11, 0.45)"
                    : "0 0 12px rgba(59, 130, 246, 0.35)",
                scale: variantKey === "idle" ? 1 : variantKey === "validating" ? 1.04 : 1.08,
              }}
              transition={{ type: "spring", stiffness: 140, damping: 14 }}
            >
              <div className="text-xs uppercase text-slate-300/80">
                {node.role === "leader" ? "Leader" : "Validator"}
              </div>
              <div className="text-lg font-semibold tracking-tight text-slate-50">
                {node.label}
              </div>
              <span className="text-[11px] font-medium text-slate-300/80">
                {statusLabel[variantKey]}
              </span>
            </motion.div>
          );
        })}

        <AnimatePresence>
          {messages.map((message) => {
            const source = positionMap.get(message.source);
            const target = positionMap.get(message.target);

            if (!source || !target) {
              return null;
            }

            const transitionDuration = (currentStep?.duration ?? 1000) / 1000;

            return (
              <motion.div
                key={message.id}
                initial={{
                  left: `${source.x}%`,
                  top: `${source.y}%`,
                  opacity: 0,
                  scale: 0.6,
                }}
                animate={{
                  left: `${target.x}%`,
                  top: `${target.y}%`,
                  opacity: 1,
                  scale: 1,
                }}
                exit={{ opacity: 0 }}
                transition={{ duration: transitionDuration, ease: "easeInOut" }}
                className="pointer-events-none absolute flex h-10 w-36 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-xl border border-slate-700/80 bg-slate-900/80 text-xs font-medium text-slate-200"
              >
                {message.label}
              </motion.div>
            );
          })}
        </AnimatePresence>

        <div className="absolute bottom-4 left-1/2 flex -translate-x-1/2 items-center gap-2 rounded-full border border-slate-700/60 bg-slate-900/90 px-4 py-1 text-xs font-medium uppercase tracking-wide text-slate-200">
          <span className="h-2 w-2 rounded-full bg-emerald-400" />
          {activePhase === "idle" ? "Idle" : `${activePhase} phase`}
        </div>
      </div>
    </div>
  );
}
