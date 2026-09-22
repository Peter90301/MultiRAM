# FeNAND Coarse Filter Model for Proteomic Clustering

## Functional Role

The FeNAND stage reads compact precursor metadata and assigns each spectrum to
a precursor-m/z bucket before FeRAM similarity clustering. It does not discard
spectra. It removes only cross-bucket pair comparisons:

```text
unfiltered pairs = N * (N - 1) / 2
filtered pairs   = sum_i B_i * (B_i - 1) / 2
pair survival    = filtered pairs / unfiltered pairs
```

This makes filter selectivity dataset-derived. The default bucket width is
10 Da, matching the clustering model that was previously treated as free host
preprocessing.

## Default Parameters

| Parameter | Default | Status |
| --- | ---: | --- |
| Bucket width | 10 Da | Algorithm/model parameter |
| Metadata per spectrum | 16 B | Assumed compact precursor/index record |
| FeNAND-to-FeRAM payload | 256 B/spectrum | Assumed packed 2048-bit HV |
| FeNAND decompressed stream | 8.1 GB/s | MultiRAM architecture input |
| FeNAND-to-FeRAM package link | 256 GB/s | MultiRAM architecture upper bound |
| Stage setup latency | 50 us | Modeling assumption inherited from the existing simulator |
| Internal filter energy | 20 pJ/spectrum | Modeling assumption, not measured |
| Package-link energy | 0.8 pJ/bit | Modeling assumption, not measured |

The effective output bandwidth is:

```text
B_effective = min(8.1 GB/s, 256 GB/s) = 8.1 GB/s
```

Latency and energy are calculated as:

```text
T_filter = T_setup
         + D_metadata / B_FeNAND
         + D_HV / B_effective

E_filter = N * E_filter_per_spectrum
         + 8 * D_HV * E_package_link_per_bit

T_clustering = T_filter + T_FeRAM_core
E_clustering = E_filter + E_FeRAM_core
```

All assumptions are exposed as command-line parameters in
`run_pim_full_pipeline_timing.py` and `pim_hyperoms_estimator.py`. Use
`--disable-fenand-coarse-filter` to reproduce the previous behavior where
precursor bucketing was modeled as free host preprocessing.

## Interpretation

The pair-reduction percentage is measured from bucket occupancy for every
dataset. The filter does not claim a spectrum-rejection rate, and therefore
does not change cluster assignments by itself. Quality metrics should remain
identical for a fixed bucket width; only hardware latency and energy change.

The simulator also reports an unfiltered FeRAM baseline in which all spectra
occupy one bucket. `speedup_vs_unfiltered_feram` includes FeNAND overhead and
must be used instead of interpreting pair reduction as runtime speedup. A
filter can reduce energy while slightly increasing latency on a dataset that
is not large enough to require additional FeRAM column passes.
