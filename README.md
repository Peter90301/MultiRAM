# MultiRAM Simulation Pipeline

This repository contains the composed MultiRAM simulator used to estimate
end-to-end latency, data movement, and transfer energy across genomics,
proteomics, and MoE stages. Hardware numbers produced by these scripts are
model outputs unless a result is explicitly labeled as measured.

## Components

| Component | Directory | Role |
| --- | --- | --- |
| Full-pipeline driver | `run_pim_full_pipeline_timing.py` | Runs selected stages, adds their modeled latencies, and writes JSON/Markdown summaries. |
| Genomic PNM | `GenDP/GenDRAM/` | Performs minigraph-like sequence-to-graph alignment and maps seeding/alignment work onto the 32 GB M3D DRAM model. |
| FeRAM clustering | `FeRAM_simulation/` and `proteomic_full_pipeline/` | Encodes spectra as hypervectors and estimates FeRAM clustering latency and energy. |
| FeNAND coarse filter | `sumukh_proteomic_test/gpu_benchmark.py` | Streams precursor metadata, assigns charge-aware 5 Da buckets, and removes cross-bucket candidate pairs before FeRAM clustering. |
| FeRAM OMS | `proteomic_full_pipeline/pim_hyperoms_estimator.py` | Runs HyperOMS-style candidate search, HDC similarity, target-decoy filtering, and reports modeled PIM cost. |
| MoE | `TRPCA-MOE-2/TRPCA_MoE/` | Runs the downstream Top-1 mixture-of-experts model or reuses a cached timing result. |
| Package transfer | Full-pipeline driver | Models host, FeNAND, FeRAM, M3D DRAM, ring, and UCIe transfers as `sum(bytes_i / bandwidth_i)`. |
| Data-movement experiments | `multiram_data_movement/` | Profiles CPU/GPU baselines and projects matched subset measurements to large NA12878 and PXD workloads. |

The M3D DRAM organization is documented in
[`M3D_DRAM_CONFIGURATION.md`](M3D_DRAM_CONFIGURATION.md): 32 GB, 8 tiers,
32 channels, one PU per channel, and eight 1 Gb logical banks per channel.
The FeNAND assumptions and latency/energy equations are documented in
[`FENAND_COARSE_FILTER_MODEL.md`](FENAND_COARSE_FILTER_MODEL.md).
The clustering-quality calibration, tradeoffs, and b1926-b1930 validation are
reported in
[`MULTIRAM_CLUSTERING_QUALITY_OPTIMIZATION.md`](MULTIRAM_CLUSTERING_QUALITY_OPTIMIZATION.md).

## Setup

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For OMS, build the included HOMSTC parser extension:

```bash
pip install -e sumukh_proteomic_test/homs-tc
```

HGA is an optional GPU baseline and is not vendored here:

```bash
git clone https://github.com/RapidsAtHKUST/hga external/hga
make -C external/hga all
```

## Quick Smoke Test

The repository includes small MHC reads/reference files. This command runs the
M3D DRAM genomic stage plus the package-transfer model without requiring the
large omics datasets:

```bash
python3 run_pim_full_pipeline_timing.py \
  --genomic-backend gendp \
  --genomic-max-queries 10 \
  --skip-proteomic \
  --skip-moe \
  --output-dir outputs/genomic_smoke
```

The main outputs are:

- `pim_full_pipeline_timing_summary.json`: machine-readable stage timing.
- `pim_full_pipeline_timing_summary.md`: compact timing table.
- `transfer_model_summary.json`: bytes, bandwidth, latency, and link energy for each transfer.
- `logs/`: raw stdout/stderr from each executed component.

## Full Pipeline

Large datasets and spectral libraries are intentionally not stored in Git.
Pass their locations explicitly:

```bash
python3 run_pim_full_pipeline_timing.py \
  --genomic-backend gendp \
  --genomic-ref-fasta /path/to/reference.fa \
  --genomic-query-fastq /path/to/reads.fastq \
  --proteomic-query /path/to/query.mgf \
  --proteomic-ref /path/to/reference.splib \
  --proteomic-config sumukh_proteomic_test/homs-tc/configs/hek293.ini \
  --proteomic-python "$(command -v python3)" \
  --proteomic-clustering-python "$(command -v python3)" \
  --reuse-moe-json \
  --transfer-include-reference \
  --output-dir outputs/full_pipeline
```

Use `--proteomic-max-queries N` for a subset experiment. Run
`python3 run_pim_full_pipeline_timing.py --help` for all bandwidth, backend,
candidate-cap, and stage-selection options.

## Timing Definition

The reported composed latency is:

```text
T_total = T_genomic + T_clustering + T_OMS + T_transfer + T_MoE
```

`latency_sec` is the modeled or parsed hardware-stage latency. `wall_time_sec`
is the time spent running the Python/native simulator and should not be
reported as PIM hardware latency. The package-link energy defaults to an
explicit modeling assumption of 0.8 pJ/bit, not a measured value.

## Dataset Policy

The included FASTA, FASTQ, and MGF files are smoke-test subsets only. Full
Platinum Genomes NA12878, HMP2/IBDMDB, PXD024364, model checkpoints, generated
indexes, binaries, and experiment outputs are excluded by `.gitignore`.
