# Changelog

All notable changes to the Robust-Continual-Flow project.

## [1.0.0] - 2026-02-26

### Added

#### Core Components
- **TrajectoryLogger** — Graph-based trajectory logging with node/edge tracking
- **UncertaintyEstimator** — CATTS-inspired normalized entropy calculation from votes/probabilities
- **ComputeAllocator** — Dynamic compute budget scaling based on uncertainty thresholds
- **TrajectoryPruner** — WebClipper-style cycle detection and branch pruning
- **FailureMonitor** — Reasoning fatigue and suggestion hijacking detection
- **NavigationAgent** — Multi-step orchestration with integrated monitoring

#### Environment & Benchmarking
- **AdversarialEnvironment** — Multi-turn adversarial test environment
- **BenchmarkRunner** — Integrated accuracy vs. token efficiency benchmarking
- **ScalingFatigueAnalyzer** — Compute-fatigue correlation analysis
- **MockModel** — Simulated model for testing without LLM calls

#### CLI Interface
- `benchmark` command — Run full benchmark suite
- `analyze` command — Scaling vs. fatigue analysis with JSON output
- `dashboard` command — Real-time ASCII visualization
- `evaluate` command — Domain-specific evaluation

### Test Coverage
- 109 tests across all modules
- Full coverage of core algorithms
- Integration tests for agent loop

### Documentation
- Comprehensive README with architecture diagram
- API documentation in docstrings
- Usage examples for programmatic access

---

## Development History

### Step 1: Project Scaffold
Initial project structure with `src/`, `tests/`, and `requirements.txt`.

### Step 2: Trajectory Graph Logger
Implemented `TrajectoryLogger` class for capturing agent actions as directed graph.

### Step 3: Uncertainty Estimator (CATTS)
Added `UncertaintyEstimator` with normalized entropy calculation from vote distributions.

### Step 4: Dynamic Compute Allocator
Implemented `ComputeAllocator` for threshold-based compute scaling.

### Step 5: Graph-based Trajectory Pruner (WebClipper)
Added `TrajectoryPruner` with DFS-based cycle detection and uncertainty-based branch pruning.

### Step 6: Failure Mode Monitor
Implemented `FailureMonitor` for detecting Reasoning Fatigue and Suggestion Hijacking.

### Step 7: Multi-Step Navigation Agent
Created `NavigationAgent` orchestrating all components in integrated loop.

### Step 8: Adversarial Simulation Environment
Added `AdversarialEnvironment` with multi-turn adversarial scenarios.

### Step 9: Integrated Benchmark Run
Implemented `BenchmarkRunner` with `MockModel` for accuracy vs. token efficiency measurement.

### Step 10: Scaling vs. Fatigue Analysis
Added `ScalingFatigueAnalyzer` for computing correlation between test-time scaling and fatigue onset.

### Step 11: CLI & Real-time Dashboard
Created CLI interface with ASCII dashboard visualization.

### Step 12: Final Documentation & README
Comprehensive documentation, architecture diagram, and usage examples.

---

## Research Foundation

This project implements techniques from:

- **WebClipper** (Wang et al., 2026) — Graph-based trajectory pruning
- **CATTS** (Lee et al., 2026) — Test-time compute scaling
- **Multi-Turn Attack Consistency** (Li et al., 2026) — Failure mode detection
