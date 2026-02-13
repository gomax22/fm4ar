# MST-FMPE: Multi-Scale Transformer for High-Resolution Atmospheric Retrieval of Exoplanets

![Python 3.10](https://img.shields.io/badge/python-3.10+-blue)
[![Data availability](https://img.shields.io/badge/Data-Available_on_Edmond-31705e)](https://doi.org/10.17617/3.LYSSVN)

This repository contains the code for the research paper:

> M. Giordano Orsini, A. Ferone, L. Inno, A. Maratea, A. Casolaro, P. Giacobbe, L. Pino, A. S. Bonomo (2026). 
> "MST-FMPE: Multi-Scale Transformer for High-Resolution Atmospheric Retrieval of Exoplanets".
> Submitted to _Journal of Computational Physics._

---


## 🚀 Quickstart

Installation should generally work by checking out this repository and running `pip install` on it:

```bash
git clone git@github.com:gomax22/fm4ar.git ;
cd fm4ar ;
pip install -e .
```

## 🏕 Setting up the environment

The code in here relies on some environmental variables that you need to set:

```bash
export FM4AR_DATASETS_DIR=/path/to/datasets ;
export FM4AR_EXPERIMENTS_DIR=/path/to/experiments ;
```

You might want to add these lines to your `.bashrc` or `.zshrc` file.

Generally, these folders can be subfolders of this repository; however, there may exists scenarios where this is not desirable (e.g., on a cluster).


## 📜 Citation and credits
If you use find this code useful, please cite our work:

```bibtex
@article{GiordanoOrsini_2026,
  author     = {Giordano Orsini, Massimiliano and Ferone, Alessio and Inno, Laura and Maratea, Antonio and Casolaro, Angelo and Giacobbe, Paolo and Pino, Lorenzo and Bonomo, Aldo S.}, 
  title      = {MST-FMPE: Multi-Scale Transformer for High-Resolution Atmospheric Retrieval of Exoplanets},
  year       = 2026,
  journal    = {Journal of Computational Physics},
  addendum   = {(Submitted)},
}
```


and the original work:

```bibtex
@article{Gebhard_2024,
  author     = {Gebhard, Timothy D. and Wildberger, Jonas and Dax, Maximilian and Angerhausen, Daniel and Quanz, Sascha P. and Schölkopf, Bernhard},
  title      = {Flow Matching for Atmospheric Retrieval of Exoplanets: Where Reliability meets Adaptive Noise Levels},
  year       = 2024,
  journal    = {Astronomy \& Astrophysics},
  eprint     = {2410.21477},
  eprinttype = {arXiv},
  addendum   = {(Accepted)},
}
```

## ⚖️ License and copyright
This repository is a fork of [fm4ar](https://github.com/timothygebhard/fm4ar), adapted to our research.

The code in this repository was written by [Massimiliano Giordano Orsini](https://github.com/gomax22), and is owned by the [Computer Vision and Pattern Recognition "Alfredo Petrosino" Laboratory (CVPRLAB), Department of Science and Technology, Parthenope University of Naples](https://cvprlab.uniparthenope.it)

The original code in this repository was written by [Timothy Gebhard](https://github.com/timothygebhard), with contributions from [Jonas Wildberger](https://github.com/jonaswildberger) and [Maximilian Dax](https://github.com/max-dax), and is owned by the [Max Planck Society](https://www.mpg.de/en).
The original code is released under the **BSD-3 Clause License** (see [LICENSE](LICENSE) for details). All modifications in this repository are released under the same license terms.

