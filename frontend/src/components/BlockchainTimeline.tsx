import { motion } from "framer-motion";
import type { BlockSummary } from "../types/blockchain";

interface BlockchainTimelineProps {
  blocks: BlockSummary[];
  selectedHash: string | null;
  onSelect: (block: BlockSummary) => void;
}

export function BlockchainTimeline({ blocks, selectedHash, onSelect }: BlockchainTimelineProps) {
  return (
    <div className="space-y-3">
      {blocks.map((block) => {
        const isActive = selectedHash === block.hash;

        return (
          <motion.button
            key={block.hash}
            type="button"
            whileHover={{ scale: 1.01 }}
            whileTap={{ scale: 0.99 }}
            onClick={() => onSelect(block)}
            className={`w-full rounded-2xl border border-slate-800/80 bg-slate-900/80 p-4 text-left transition-colors ${
              isActive ? "border-emerald-400/70 ring-1 ring-emerald-400/40" : "hover:border-slate-700/80"
            }`}
          >
            <div className="flex items-center justify-between">
              <p className="text-sm font-semibold text-slate-100">Block #{block.height}</p>
              <span className="rounded-full border border-slate-700/80 px-3 py-1 text-[11px] uppercase tracking-wide text-slate-300">
                {block.domain.replace("-", " ")}
              </span>
            </div>
            <div className="mt-2 flex items-center justify-between text-xs text-slate-400">
              <span>{new Date(block.timestamp).toLocaleTimeString()}</span>
              <span>{block.fallback ? "Simulated" : "Live"}</span>
            </div>
          </motion.button>
        );
      })}
    </div>
  );
}
