import type { MineContext, SimulationCallbacks, SimulationResult } from "../types/blockchain";
import { blockchainService, BlockchainService } from "./BlockchainService";

export class SimulationLayer {
  private readonly service: BlockchainService;

  constructor(service: BlockchainService = blockchainService) {
    this.service = service;
  }

  async execute(
    context: MineContext,
    callbacks?: SimulationCallbacks,
  ): Promise<SimulationResult> {
    try {
      const result = await this.service.mineBlock(context);

      for (const step of result.steps) {
        callbacks?.onPhaseChange?.(step.phase, step);
        callbacks?.onNodeStatus?.(step.statusByNode);
        callbacks?.onMessages?.(step.messages);
        await this.delay(step.duration);
      }

      callbacks?.onMessages?.([]);
      callbacks?.onComplete?.(result);

      return result;
    } catch (error) {
      const normalizedError =
        error instanceof Error ? error : new Error("Simulation execution failed");

      callbacks?.onError?.(normalizedError);
      throw normalizedError;
    }
  }

  private delay(duration: number): Promise<void> {
    return new Promise((resolve) => {
      setTimeout(resolve, duration);
    });
  }
}

export const simulationLayer = new SimulationLayer();
