# AI4Antenna — Deep Learning for Antenna Forward Prediction & Inverse Design

This project is an actively maintained research and engineering codebase that applies deep learning to both forward prediction and inverse design of antennas. The goal is to provide a complete, closed-loop pipeline covering dataset generation, model training, inference, and simulation-based validation.

Key goals

- Provide reproducible scripts to generate antenna datasets (geometries, S-parameters, fields).

- Train and evaluate deep learning models for forward prediction (geometry -> performance) and inverse design (spec -> geometry).

- Integrate model-driven designs with Antenna Toolbox simulations for verification.

- Benchmark parallel simulation workloads and provide helper utilities for batch experiments.

Repository structure (high level)

- `pct_benchmark/` — Parallel Computing Toolbox based benchmark and experiment scripts (contains multiple benchmark and optimizer variants).

- `Single_ant/` — Single-antenna examples and small demos (pixelated patch, probe-fed patch, GA-based miniaturization examples).

- `debug_script/` — Local debugging and plotting helpers (kept out of README usage examples).

Note: the repository contains many MATLAB example scripts that use Antenna Toolbox and related toolboxes. Some folders contain generated figures used for documentation and analysis.

Requirements

- MATLAB (with a recent release recommended).

- Antenna Toolbox.

- RF/PCB related toolboxes (e.g. RF Toolbox or RF PCB Toolbox) for `pcbStack` and related commands.

- Parallel Computing Toolbox for running the included parallel benchmarks.

Quick start

1. Open MATLAB in the repository directory (or add the folder to the path).

1. Confirm required toolboxes and licenses are available in MATLAB.

1. Run a lightweight example to verify environment (recommended):

```powershell
matlab -batch "single_pixel_antenna"
```

1. Run a benchmark (example):

```powershell
matlab -batch "pct_benchmark_pixel_antenna"
```

Notes and tips

- Many scripts perform heavy meshing and electromagnetic simulations; expect significant CPU, memory, and time usage for higher-resolution experiments.

- If running parallel experiments, configure `parpool` according to your system resources (number of physical cores, available memory).

- Some scripts include license checks and will exit early if required toolboxes are not present.

Contributing

- This repository is actively maintained. Contributions should follow the repository style and include small, well-tested commits. Please open issues for discussion before implementing large changes.

License

- Add or update a license file as needed for your project. No license is included by default in this repository.
