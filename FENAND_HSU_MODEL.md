# Hsu-Calibrated FeNAND Model

Primary reference: P.-K. Hsu et al., "3-D Ferroelectric NAND In-Storage
Processing Architecture for Mass Spectrometry," IEEE Design & Test, 2026,
doi: 10.1109/MDAT.2025.3596025.

The simulator uses the numerical HEK293 operating point from the same authors'
open MEMSYS 2023 predecessor: 46,665 queries, 0.429 s, and 0.491 J.

```text
array latency/query = 0.429 / 46,665 = 9.193185 us
array energy/query  = 0.491 / 46,665 = 10.521804 uJ
```

OMS charges this cost per search stage. Applying it per spectrum to the
clustering coarse filter is a conservative cross-workload extrapolation, not a
measured clustering result. Package-link energy and MultiRAM stream bandwidth
remain separate architecture assumptions.
