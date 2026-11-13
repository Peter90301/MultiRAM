# Source-release validation

Validated on Linux with existing Python 3.10 environments:

- Six fixture-free regression functions passed: coarse filtering, monotonic
  candidate count, deterministic Top-K ties, metrics, batched encoding
  time/energy, and driver import/default input paths.
- Four preserved unittest cases passed: precursor charge parsing, charge-aware
  buckets, fixed-dimension Hamming cutoff, and observed pair-reduction math.
- All repository Python source files parsed successfully.
- The included HOMSTC Cython parser built and its OMS imports succeeded.
  The build emitted legacy Cython language-level and NumPy API warnings.
- The genomic smoke test ran 10 queries and package-transfer accounting;
  both stages reported status ok.
- Four new/updated analytical and sensitivity entry points accepted --help.
- git diff --check passed. Selected text files were screened for common
  credential patterns and machine-local paths.

pytest was not installed in the existing validation environments; the six
fixture-free test functions were invoked directly, and the four original
cases were run with unittest. requirements-dev.txt provides pytest for users.

Not rerun: full omics datasets, GPU baselines, sensitivity sweeps, MoE training,
thermal solver, and full paper reproduction. Passing these checks is not a
validation of the paper's modeled speedup/energy assumptions. No benchmark
results were regenerated for this release.
