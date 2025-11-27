# Consensus Algorithm Visualizer (Frontend)

A React + Vite dashboard that demonstrates how a private blockchain reaches consensus across multiple business domains. The UI highlights leader election, validation votes, and commit propagation while the `SimulationLayer` bridges missing backend endpoints by falling back to a deterministic mock.

## Key Features

- **Interactive Network Mesh** – Framer Motion drives animated state transitions (Idle → Validating → Committed) for each validator node, including vote/commit message flows.
- **Dynamic Validators** – Introduce additional nodes after each round to visualise how the mesh adapts and redistributes consensus roles.
- **Domain Switcher** – Toggle between Supply Chain, Healthcare, and Finance to see domain-specific payload structures injected into new blocks.
- **Blockchain Timeline** – Inspect hashes, previous links, latency, and validator signatures per block in a scrollable timeline.
- **SimulationLayer Fallback** – Attempts to POST to `/mine`; on failure it simulates latency, generates hashes/signatures, and streams phase updates to the UI.
- **Tailwind UI Shell** – Tailwind CSS powers a responsive control room aesthetic with Lucide icons for quick status recognition.

## Getting Started

```bash
# Install dependencies
npm install

# Start the Vite dev server (http://localhost:5173)


# Type-check and build production assets
npm run build

# Optional: run ESLint
npm run lint
```

The project also includes a VS Code task named `npm: dev` that launches the development server.

## Project Structure

```
src/
  components/       // UI building blocks (graph, timeline, details, switcher)
  data/             // Domain presets + payload generators
  services/         // BlockchainService + SimulationLayer orchestrator
  types/            // Shared TypeScript contracts for the simulation
  utils/            // Random data helpers (hex/signature generators)
  App.tsx           // Dashboard composition & state management
```

### Simulation Flow
1. `SimulationLayer.execute()` requests a block via `BlockchainService.mineBlock()`.
2. `BlockchainService` first tries the real `/mine` endpoint (with timeout safeguards).
3. On failure it synthesizes a block, builds step metadata (`proposal`, `validation`, `commit`), and streams node/message updates back to the UI.
4. The UI adds the new block to the timeline and presents the domain payload + validator signatures.

## Customisation Ideas

- Wire the service to the real `/mine` API once available (retain fallback as a resilience layer).
- Extend `DOMAIN_OPTIONS` with additional industries or payload templates.
- Record consensus history in local storage to persist simulations between sessions.
- Surface validator health metrics (latency histograms, failed vote scenarios) using the existing messaging hooks.

## Tech Stack

- **React 19 + TypeScript** via Vite
- **Tailwind CSS** for layout and theming
- **Framer Motion** for animated edges/messages
- **Lucide Icons** to label domains and statuses

---

Built to accompany the Private Blockchain backend while development of real consensus endpoints is in progress.
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
