# spme-comparison

A comparison of the terminal voltage predicted by the SPMe [1], DFN, SP, and cSP [2] models.

The script uses the open-source battery modelling software [PyBaMM](https://github.com/pybamm-team/PyBaMM). To install the appropriate version of PyBaMM, run:
```
pip install -e git+https://github.com/pybamm-team/pybamm.git@csp-updated#egg=pybamm
```

References:
>[1]  S.G. Marquis, V. Sulzer, R. Timms, C.P. Please, and S.J. Chapman. An Asymptotic Derivation of a Single Particle Model with Electrolyte.Journal of The Electrochemical Society, 166(15):A3693–A3706, 2019.

>[2]  G. Richardson, I. Korotkin, R. Ranom, M. Castle, and J.M. Foster. Generalised single particle models for high-rate operation of graded lithium-ion electrodes:  Systematic derivation and validation. Electrochimica Acta, 339:135862, 2020.
