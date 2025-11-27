import type { LucideIcon } from "lucide-react";

export type DomainKey = "supply-chain" | "healthcare" | "finance";

export interface DomainOption {
  key: DomainKey;
  label: string;
  description: string;
  icon: LucideIcon;
  accent: string;
}

export type NodeRole = "leader" | "validator";
export type NodeStatus = "idle" | "validating" | "committed";
export type ConsensusPhase = "idle" | "proposal" | "validation" | "commit";

export interface NetworkNode {
  id: string;
  label: string;
  role: NodeRole;
  status: NodeStatus;
  position: {
    x: number;
    y: number;
  };
}

export interface SimulationMessage {
  id: string;
  source: string;
  target: string;
  label: string;
  phase: Exclude<ConsensusPhase, "idle">;
}

export interface SimulationStep {
  phase: ConsensusPhase;
  duration: number;
  description: string;
  statusByNode: Record<string, NodeStatus>;
  messages: SimulationMessage[];
}

export type DomainPayload = Record<string, string | number>;

export interface ConsensusVote {
  validatorId: string;
  approved: boolean;
  signature: string;
}

export interface BlockSummary {
  height: number;
  hash: string;
  previousHash: string;
  timestamp: string;
  domain: DomainKey;
  payload: DomainPayload;
  proposer: string;
  votes: ConsensusVote[];
  latencyMs: number;
  fallback: boolean;
}

export interface MineContext {
  domain: DomainKey;
  leaderId: string;
  validatorIds: string[];
  previousHash: string;
  height: number;
  payload: DomainPayload;
}

export interface SimulationResult {
  block: BlockSummary;
  steps: SimulationStep[];
  fallback: boolean;
}

export interface SimulationCallbacks {
  onPhaseChange?: (phase: ConsensusPhase, step: SimulationStep) => void;
  onNodeStatus?: (statusMap: Record<string, NodeStatus>) => void;
  onMessages?: (messages: SimulationMessage[]) => void;
  onComplete?: (result: SimulationResult) => void;
  onError?: (error: Error) => void;
}
