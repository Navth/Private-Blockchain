import type {
  MineContext,
  NodeStatus,
  SimulationResult,
  SimulationStep,
} from "../types/blockchain";
import {
  randomHex,
  randomNumber,
  randomSignature,
} from "../utils/random";

export class BlockchainService {
  private readonly latencyRange = { min: 650, max: 1600 };
  private readonly requestTimeout = 2500;

  async mineBlock(context: MineContext): Promise<SimulationResult> {
    const realResult = await this.tryRealEndpoint(context);

    if (realResult) {
      return realResult;
    }

    return this.simulateConsensus(context);
  }

  private async tryRealEndpoint(
    context: MineContext,
  ): Promise<SimulationResult | null> {
    if (typeof fetch === "undefined") {
      return null;
    }

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.requestTimeout);

    try {
      const response = await fetch("/mine", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(context),
        signal: controller.signal,
      });

      if (!response.ok) {
        return null;
      }

      const payload = await response.json();
      if (!payload?.block) {
        return null;
      }

      const normalizedBlock = {
        height: payload.block.height ?? context.height,
        hash: payload.block.hash ?? `0x${randomHex(32)}`,
        previousHash: payload.block.previousHash ?? context.previousHash,
        timestamp: payload.block.timestamp ?? new Date().toISOString(),
        domain: payload.block.domain ?? context.domain,
        payload: payload.block.payload ?? context.payload,
        proposer: payload.block.proposer ?? context.leaderId,
        votes:
          payload.block.votes ??
          context.validatorIds.map((validatorId) => ({
            validatorId,
            approved: true,
            signature: `0x${randomSignature(20)}`,
          })),
        latencyMs: payload.block.latencyMs ?? randomNumber(650, 1250),
        fallback: false,
      } satisfies SimulationResult["block"];

      return {
        block: normalizedBlock,
        steps: payload.steps ?? [],
        fallback: false,
      } satisfies SimulationResult;
    } catch (error: unknown) {
      if (error instanceof DOMException && error.name === "AbortError") {
        console.warn("/mine request timed out; using simulation layer fallback");
      }

      return null;
    } finally {
      clearTimeout(timer);
    }
  }

  private async simulateConsensus(context: MineContext): Promise<SimulationResult> {
    const latencyMs = randomNumber(this.latencyRange.min, this.latencyRange.max);
    await this.delay(latencyMs * 0.2);

    const now = new Date().toISOString();
    const hash = `0x${randomHex(32)}`;
    const block = {
      height: context.height,
      hash,
      previousHash: context.previousHash,
      timestamp: now,
      domain: context.domain,
      payload: context.payload,
      proposer: context.leaderId,
      votes: context.validatorIds.map((validatorId) => ({
        validatorId,
        approved: true,
        signature: `0x${randomSignature(20)}`,
      })),
      latencyMs,
      fallback: true,
    } satisfies SimulationResult["block"];

    const steps: SimulationStep[] = [
      {
        phase: "proposal",
        duration: 900,
        description: `${context.leaderId} batches domain payload and broadcasts proposal`,
        statusByNode: this.composeStatusMap(context, {
          [context.leaderId]: "validating",
        }),
        messages: context.validatorIds.map((validatorId) => ({
          id: `proposal-${validatorId}`,
          source: context.leaderId,
          target: validatorId,
          label: "Proposal Broadcast",
          phase: "proposal",
        })),
      },
      {
        phase: "validation",
        duration: 1100,
        description: "Validators execute checks and vote",
        statusByNode: this.composeStatusMap(context, {
          [context.leaderId]: "validating",
          ...context.validatorIds.reduce<Record<string, "validating">>((acc, id) => {
            acc[id] = "validating";
            return acc;
          }, {}),
        }),
        messages: context.validatorIds.map((validatorId) => ({
          id: `vote-${validatorId}`,
          source: validatorId,
          target: context.leaderId,
          label: "Validation Vote",
          phase: "validation",
        })),
      },
      {
        phase: "commit",
        duration: 850,
        description: "Leader broadcasts commit and finalizes block",
        statusByNode: this.composeStatusMap(
          context,
          context.validatorIds.concat(context.leaderId).reduce<Record<string, NodeStatus>>((acc, id) => {
            acc[id] = "committed";
            return acc;
          }, {}),
        ),
        messages: context.validatorIds.map((validatorId) => ({
          id: `commit-${validatorId}`,
          source: context.leaderId,
          target: validatorId,
          label: "Commit Proof",
          phase: "commit",
        })),
      },
    ];

    return {
      block,
      steps,
      fallback: true,
    } satisfies SimulationResult;
  }

  private composeStatusMap(
    context: MineContext,
    overrides: Record<string, NodeStatus>,
  ): Record<string, NodeStatus> {
    const map: Record<string, NodeStatus> = {};
    [...context.validatorIds, context.leaderId].forEach((id) => {
      map[id] = overrides[id] ?? "idle";
    });
    return map;
  }

  private delay(duration: number): Promise<void> {
    return new Promise((resolve) => {
      setTimeout(resolve, duration);
    });
  }
}

export const blockchainService = new BlockchainService();
