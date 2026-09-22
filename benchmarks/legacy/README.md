# Historical scaling experiments

These scripts are preserved from the working project's ExampleCode/ablationStudy directory. They benchmark simplified, separately implemented architectures; they do not import the packaged sams model.

Original CSV, JSON and PNG outputs are in [results/reference](../results/reference). The provenance file records source location and SHA-256 for each preserved artifact.

The records are useful for tracing the architecture exploration, but lack complete hardware/environment provenance. Do not use their ratios as current packaged-model speedup claims. They were not rerun as part of package validation.

For a new measurement, use [scripts/benchmark.py](../../scripts/benchmark.py), which imports the actual model and records all individual timings. Install the plots extra to produce a chart.
