<!-- PROJECT SHIELDS -->
[![arXiv][arxiv-shield]][arxiv-url]
[![MIT License][license-shield]][license-url]
[![ReseachGate][researchgate-shield]][researchgate-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
[![GIT][git-shield]][git-url]
<!-- [![finalpaper][finalpaper-shield]][finalpaper-url] -->
<!-- [![Scholar][scholar-shield]][scholar-url] -->
<!-- [![Webpage][webpage-shield]][webpage-url] -->

# Online Convex Optimization and Integral Quadratic Constraints: A new approach to regret analysis

This repository contains the code from our paper "Online Convex Optimization and Integral Quadratic Constraints: A new approach to regret analysis". The preprint is accessible on [arXiv](https://arxiv.org/abs/2503.23600).

## Installation
The code was developed with Python 3.11. All relevant packages can be installed with
```bash 
pip install -r requirements.txt
```

All SDPs are solved with the commericial solver MOSEK.
```bash 
pip install Mosek
```
An academic license can be requested [here](https://www.mosek.com/products/academic-licenses/). Other open-source solvers might work as well (e.g. cvxopt), however, we observed best numerical stability with MOSEK.

## Running Experiments
All experiments can be replicated in an associated notebook ``oco_iqc.ipynb``.

## Contact

🧑‍💻 Fabian Jakob

📧 [fabian.jakob@ist.uni-stuttgart.de](mailto:fabian.jakob@ist.uni-stuttgart.de)

[git-shield]: https://img.shields.io/badge/Github-fjakob-white?logo=github
[git-url]: https://github.com/fjakob
[license-shield]: https://img.shields.io/badge/License-MIT-T?style=flat&color=blue
[license-url]: https://github.com/col-tasas/2025-oco-with-iqcs/blob/main/LICENSE
<!-- [webpage-shield]: https://img.shields.io/badge/Webpage-Fabian%20Jakob-T?style=flat&logo=codementor&color=green
[webpage-url]: https://www.ist.uni-stuttgart.de/institute/team/Jakob-00004/ add personal webpage -->
[arxiv-shield]: https://img.shields.io/badge/arXiv-2503.23600-t?style=flat&logo=arxiv&logoColor=white&color=red
[arxiv-url]: https://arxiv.org/abs/2503.23600
<!-- [finalpaper-shield]: https://img.shields.io/badge/SIAM-Paper-T?style=flat&color=red
[finalpaper-url]: https://google.com -->
[researchgate-shield]: https://img.shields.io/badge/ResearchGate-Fabian%20Jakob-T?style=flat&logo=researchgate&color=darkgreen
[researchgate-url]: https://www.researchgate.net/profile/Fabian-Jakob-4
[linkedin-shield]: https://img.shields.io/badge/Linkedin-Fabian%20Jakob-T?style=flat&logo=linkedin&logoColor=blue&color=blue
[linkedin-url]: https://www.linkedin.com/in/fabian-jakob/
