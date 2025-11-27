import type { ComponentType, SVGProps } from "react";
import { Fingerprint, Link2, ShieldCheck } from "lucide-react";
import type { BlockSummary } from "../types/blockchain";

interface BlockDetailsProps {
  block: BlockSummary | null;
}

export function BlockDetails({ block }: BlockDetailsProps) {
  if (!block) {
    return (
      <div className="flex h-full min-h-[240px] items-center justify-center rounded-3xl border border-dashed border-slate-700/60 bg-slate-900/40">
        <p className="text-sm text-slate-400">Select a block to inspect consensus details.</p>
      </div>
    );
  }

  return (
    <div className="space-y-4 rounded-3xl border border-slate-800/80 bg-slate-900/80 p-6">
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-500/15 text-emerald-300">
          <ShieldCheck className="h-5 w-5" />
        </div>
        <div>
          <p className="text-sm font-semibold text-slate-100">Block #{block.height}</p>
          <p className="text-xs text-slate-400">Latency {block.latencyMs}ms · {block.votes.length} votes</p>
        </div>
      </div>

      <div className="space-y-3">
        <DetailRow title="Hash" icon={Fingerprint} value={block.hash} />
        <DetailRow title="Previous Hash" icon={Link2} value={block.previousHash} />
      </div>

      <section>
        <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-300">Domain Payload</h4>
        <div className="mt-2 grid gap-2 rounded-2xl border border-slate-800/70 bg-slate-950/60 p-4 text-sm text-slate-200">
          {Object.entries(block.payload).map(([key, value]) => (
            <div key={key} className="flex items-center justify-between text-xs">
              <span className="font-medium text-slate-300">{key}</span>
              <span className="text-slate-200/90">{value}</span>
            </div>
          ))}
        </div>
      </section>

      <section>
        <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-300">Validator Signatures</h4>
        {block.votes.length === 0 ? (
          <p className="mt-2 rounded-2xl border border-dashed border-slate-700/60 bg-slate-950/60 px-3 py-2 text-xs text-slate-400">
            Awaiting validator signatures.
          </p>
        ) : (
          <div className="mt-2 space-y-2 text-xs">
            {block.votes.map((vote) => (
              <div
                key={vote.validatorId}
                className="flex items-center justify-between rounded-2xl border border-emerald-500/20 bg-emerald-500/5 px-3 py-2 text-emerald-200"
              >
                <span>{vote.validatorId}</span>
                <span className="font-mono tracking-tight">{vote.signature}</span>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

interface DetailRowProps {
  title: string;
  value: string;
  icon: ComponentType<SVGProps<SVGSVGElement>>;
}

function DetailRow({ title, value, icon: Icon }: DetailRowProps) {
  return (
    <div className="rounded-2xl border border-slate-800/80 bg-slate-950/70 px-3 py-2">
      <div className="flex items-center gap-3 text-xs text-slate-300">
        <Icon className="h-4 w-4" />
        <span className="uppercase tracking-wide">{title}</span>
      </div>
      <p className="mt-1 truncate font-mono text-sm text-slate-200" title={value}>
        {value}
      </p>
    </div>
  );
}
