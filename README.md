# symseq

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Build Status](https://img.shields.io/travis/zbarni/symseq.svg)](https://travis-ci.com/zbarni/symseq)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen)](https://symseq.readthedocs.io)

<!-- [![PyPI version](https://img.shields.io/badge/PyPI-v0.1.0-blue)](https://pypi.org/project/neurolytics/) -->

! **Development Status:** 3 – Alpha: The codebase is still under active development, with several features missing
features and potential API modifications. Use at your own risk. ! 

**SymSeq** is a Python library for defining, generating, and analyzing symbolic sequences. It contains core data structures
for grammar and sequence generation, a diverse library of artificial language generators, interfaces for parsing
user-supplied data, and a comprehensive suite of analysis metrics spanning multiple structural and linguistic scales.

Together with [**SeqBench**](https://github.com/symseqbench/seqbench), **SymSeq** forms a unified framework for symbolic
sequence processing and benchmarking. A high-level overview of the combined framework and its capabilities can be found
in the accompanying [**SymSeqBench paper**](https://arxiv.org/abs/2512.24977).

* **Free software:** MIT license
* **Documentation:** [https://symseq.readthedocs.io/](https://symseq.readthedocs.io/)

---

## Features

* **Symbolic Sequence Generation:** Core algorithms for creating rule-based symbolic sequences.
* **Configurable Parameters:** Flexible options to control sequence properties.
* **Metrics & Analysis:** Tools to analyze generated sequences (e.g., complexity, statistical properties).
* **Extensible Task Framework:** Define custom tasks or experiments.
* **Command-Line Interface:** Utilities for generating and managing sequences from the terminal.

---

## Installation

### Using uv (recommended)

[uv](https://github.com/astral-sh/uv) is an extremely fast Python package installer and resolver that greatly simplifies
installation and dependency management.

1. Install `uv`:

```bash
pip install uv
# or
pipx install uv
```

2. Install `symseq`:

```bash
uv pip install symseq
```

---

### Development installation

1. Clone the repository:

```bash
git clone https://github.com/symseqbench/symseq.git
cd symseq
```

2. Create and activate a virtual environment:

```bash
uv venv
source .venv/bin/activate  # Unix/macOS
# or
.venv\\Scripts\\activate     # Windows
```

3. Install development dependencies:

```bash
uv pip install -e ".[dev]"
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code follows appropriate coding standards and includes appropriate tests.

## Citation
--------

If you use the ``SymSeq`` library, please cite our [paper](https://arxiv.org/abs/2512.24977).

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage) project template.
