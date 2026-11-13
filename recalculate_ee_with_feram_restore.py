#!/usr/bin/env python3
"""Recalculate MultiRAM EE with explicit M3D, FeRAM, and FeNAND energy.

The full data-dependent script and generated reports are maintained in the
experiment workspace. This repository copy documents the reproducible formula:

  E_restore = candidate_reads * HV_bits * 39.7 fJ/bit
  T_restore = sum(ceil(candidates_per_query / resident_columns)) * 50 ns
  endurance_fraction = restore_passes / 1e12
  E_M3D = M3D_array_bytes * 8 * 0.429 pJ/bit
"""

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-reads", type=int, required=True)
    parser.add_argument("--restore-passes", type=int, required=True)
    parser.add_argument("--query-stages", type=int, default=0)
    parser.add_argument("--fenand-candidates", type=int, default=0)
    parser.add_argument("--m3d-array-bytes", type=int, default=0)
    parser.add_argument("--old-pim-energy-j", type=float, required=True)
    parser.add_argument("--a100-energy-j", type=float, required=True)
    parser.add_argument("--package-link-energy-j", type=float, default=0.0)
    parser.add_argument("--pim-active-power-w", type=float, default=45.0)
    parser.add_argument("--hv-bits", type=int, default=8192)
    parser.add_argument("--restore-energy-fj-per-bit", type=float, default=39.7)
    parser.add_argument("--feram-read-energy-fj-per-bit", type=float, default=2.0)
    parser.add_argument("--feram-adc-digital-fj-per-candidate", type=float, default=250.0)
    parser.add_argument("--feram-dac-fj-per-query-bit", type=float, default=20.0)
    parser.add_argument("--fenand-energy-pj-per-candidate", type=float, default=20.0)
    parser.add_argument("--m3d-energy-pj-per-bit", type=float, default=0.429)
    parser.add_argument("--restore-latency-ns", type=float, default=50.0)
    parser.add_argument("--endurance-cycles", type=float, default=1e12)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    restore_energy_j = (
        args.candidate_reads * args.hv_bits * args.restore_energy_fj_per_bit * 1e-15
    )
    restore_latency_s = args.restore_passes * args.restore_latency_ns * 1e-9
    feram_read_energy_j = (
        args.candidate_reads * args.hv_bits * args.feram_read_energy_fj_per_bit * 1e-15
    )
    feram_peripheral_energy_j = (
        args.candidate_reads * args.feram_adc_digital_fj_per_candidate * 1e-15
        + args.query_stages * args.hv_bits * args.feram_dac_fj_per_query_bit * 1e-15
    )
    fenand_array_energy_j = (
        args.fenand_candidates * args.fenand_energy_pj_per_candidate * 1e-12
    )
    m3d_array_energy_j = args.m3d_array_bytes * 8 * args.m3d_energy_pj_per_bit * 1e-12
    added_active_energy_j = args.pim_active_power_w * restore_latency_s
    new_energy_j = (
        args.old_pim_energy_j
        + args.package_link_energy_j
        + restore_energy_j
        + feram_read_energy_j
        + feram_peripheral_energy_j
        + fenand_array_energy_j
        + m3d_array_energy_j
        + added_active_energy_j
    )
    result = {
        "restore_energy_j": restore_energy_j,
        "feram_read_energy_j": feram_read_energy_j,
        "feram_peripheral_energy_j": feram_peripheral_energy_j,
        "fenand_array_energy_j": fenand_array_energy_j,
        "m3d_array_energy_j": m3d_array_energy_j,
        "restore_latency_s": restore_latency_s,
        "added_active_energy_j": added_active_energy_j,
        "new_pim_energy_j": new_energy_j,
        "ee_vs_a100_x": args.a100_energy_j / new_energy_j,
        "endurance_fraction": args.restore_passes / args.endurance_cycles,
    }
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
