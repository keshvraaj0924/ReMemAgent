# Paired repeated-seed ownership

Paired ALFWorld/WebShop experiments use one explicit repeated seed sequence for both baseline and treatment conditions. Each `ExternalBenchmarkSpec.seed` must therefore remain `None`.

The paired runner now validates repeated-run seed ownership for **both** conditions before resolving external callables, constructing environments, probing model policies, or executing measured episodes. This prevents a malformed treatment specification from consuming baseline benchmark compute before failing.

The validation also reuses the repeated-benchmark episode-seed isolation contract. Because episode seeds are currently derived from each run seed and episode index, overlapping derived seed ranges are rejected before either condition begins.

This is an orchestration correctness guarantee only. It does not demonstrate benchmark effectiveness, environment determinism, model reproducibility, or statistical significance. Those claims require actual controlled multi-seed runs against the upstream ALFWorld/WebShop environments and persisted evidence artifacts.
