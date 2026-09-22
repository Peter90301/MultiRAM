# MultiRAM Clustering Quality Optimization

## Change

The clustering path now uses:

- FeNAND coarse-filter key: MGF precursor charge + 5 Da precursor-m/z bin
- SpecHD hypervector dimension: 2048 bits
- Complete-linkage Hamming cutoff: `0.455 * 2048` bits
- MGF `CHARGE` metadata at runtime; peptide truth is never used for clustering

The previous implementation derived the cutoff from twice the largest observed
distance in each bucket. That made the cutoff depend on bucket composition.
The corrected implementation derives it from the fixed 2048-bit HV width.

## Matched b1926 Result

This comparison uses the same 5,358 high-confidence repeated-cluster spectra
available to CPU Falcon, GPU RAPIDS, and MultiRAM.

| MultiRAM configuration | Incorrect ratio | Completeness | Clustered ratio | Pair precision | Pair recall |
|---|---:|---:|---:|---:|---:|
| Previous: 10 Da, bucket-relative cutoff | 6.8122% | 95.6335% | 100.0000% | 80.4455% | 60.7902% |
| Optimized: charge + 5 Da, fixed 0.455 cutoff | **0.8772%** | **93.7884%** | **100.0000%** | **95.8899%** | 45.0567% |

The incorrect ratio decreases by 87.12% relative. The tradeoff is a 1.85
percentage-point decrease in completeness and a 15.73-point decrease in pair
recall. This is a precision-oriented operating point; it avoids claiming that
quality improved on every metric.

## Independent Validation

b1927-b1930 were not used to select the operating point.

| Sample | Evaluated spectra | Incorrect ratio | Completeness | Clustered ratio |
|---|---:|---:|---:|---:|
| b1927 | 5,874 | 0.6299% | 94.2120% | 100.0000% |
| b1928 | 4,950 | 0.5657% | 94.0536% | 100.0000% |
| b1929 | 6,944 | 0.5472% | 94.1230% | 100.0000% |
| b1930 | 5,513 | 0.9614% | 93.6942% | 100.0000% |

Across the four validation runs, 156 of 23,281 evaluated spectra are minority
peptides in their predicted cluster: 0.6701%. Spectrum-weighted completeness is
94.0292%.

## Hardware-Model Impact on Full b1926

For all 42,828 b1926 spectra, charge-aware 5 Da filtering gives:

| Metric | Result |
|---|---:|
| Candidate pairs before filtering | 917,097,378 |
| Candidate pairs after filtering | 3,436,924 |
| Candidate-pair reduction | 99.6252% |
| FeNAND filter and transfer | 1.4882 ms |
| FeRAM clustering core | 1.5807 ms |
| MultiRAM clustering total | 3.0689 ms |
| MultiRAM clustering energy | 0.10284 mJ |

The runtime model remains analytical. The quality metrics come from functional
SpecHD clustering against high-confidence PeptideProphet assignments.

## Reproduction

```bash
python3 proteomic_full_pipeline/evaluate_clustering_quality.py
python3 proteomic_full_pipeline/validate_multiram_clustering_series.py
```

Detailed artifacts:

- `proteomic_full_pipeline/clustering_quality_b1926_multiram/`
- `proteomic_full_pipeline/clustering_quality_b1926_b1930_optimized/`
