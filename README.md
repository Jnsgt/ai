# Cross-Language DL Unified

A cross-language ONNX inference consistency testing toolkit for **Python**, **Node.js**, and **Java**.

This project helps you run the same ONNX model across multiple language runtimes, compare their outputs, and quickly identify numerical inconsistencies in cross-language deep learning deployment.

## Features

- Run ONNX inference with Python, Node.js, and Java backends
- Export results to JSON for reproducible comparison
- Compute pairwise consistency reports such as max absolute difference and mean absolute difference
- Package the main workflow as a Python CLI: `cldltest`
- Provide example models, inputs, and sample comparison outputs for quick demos

## Repository Structure

```text
cross-lang-dl-unified/
├─ README.md
├─ LICENSE
├─ .gitignore
├─ pyproject.toml
├─ package.json
├─ package-lock.json
├─ requirements.txt
├─ cldltest/                # Core Python package and runners
├─ docs/                    # Project documents
├─ examples/                # Example inputs, models, and demo outputs
├─ models/                  # Model source code and ONNX export scripts
├─ scripts/                 # GUI and auxiliary scripts
└─ tests/                   # Testing and comparison scripts
```

## Installation

### Python

```bash
pip install -r requirements.txt
pip install -e .
```

### Node.js

```bash
npm install
```

### Java

Make sure Java and Maven are installed and available in your `PATH` if you want to run the Java backend.

## Quick Start

Run a benchmark on the example model and input:

```bash
cldltest benchmark   --model examples/models/linear_model.onnx   --input examples/test_input.json   --outdir outputs   --backends py js
```

After running, the tool will generate backend outputs and pairwise comparison reports in the output directory.

## Example Assets

- Example inputs: `examples/cases/`
- Example ONNX model: `examples/models/`
- Demo outputs: `examples/outputs/`

## Development Notes

- Do not commit `node_modules/`, `__pycache__/`, or generated benchmark outputs.
- Keep sample assets in `examples/` small and focused.
- Prefer putting reusable code in `cldltest/` and runnable helpers in `scripts/`.

## Roadmap

See `docs/roadmap.md` for planned improvements.

## License

This repository is released under the MIT License. See `LICENSE` for details.
