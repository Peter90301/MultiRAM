#!/usr/bin/env python3
"""Recalculate MultiRAM package-transfer energy and the Table V EE ratio."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
JSON_OUT = ROOT / "package_link_energy_ee_recalculation.json"
MD_OUT = ROOT / "PACKAGE_LINK_ENERGY_EE_RECALCULATION.md"

LINK_ENERGY_PJ_PER_BIT = 0.8
BITS_PER_BYTE = 8
LINK_ENERGY_PJ_PER_BYTE = LINK_ENERGY_PJ_PER_BIT * BITS_PER_BYTE

# Decimal SI units match the GB/s and MB labels in the paper table.
GB = 1_000_000_000
MB = 1_000_000

TRANSFERS = [
    ("Pangenome graph loading", "FeNAND -> 3D DRAM", 1.00 * GB, 256.0),
    ("Spectral/HV reference loading", "FeNAND -> FeRAM", 0.77 * GB, 256.0),
    ("Query dispatch", "Host/FeNAND -> Compute", 32 * MB, 256.0),
    ("Proteomic feature forwarding", "FeRAM -> 3D DRAM", 64 * MB, 512.0),
    ("Genomic feature forwarding", "3D DRAM local", 8 * MB, 2000.0),
    ("MoE token preparation", "3D DRAM local", 5 * MB, 2000.0),
]

WARM_TRANSFER_NAMES = {
    "Query dispatch",
    "Proteomic feature forwarding",
    "Genomic feature forwarding",
    "MoE token preparation",
}

# Existing Table V values. The package term is replaced by the recalculated
# energy in joules, following the arithmetic used by the current draft.
A100_PIPELINE_ENERGY_UNITS = 4.300
MULTIRAM_NON_PACKAGE_ENERGY_UNITS = {
    "genomic_alignment": 0.0025,
    "proteomic_clustering": 0.012,
    "proteomic_oms": 0.016,
    "moe_inference": 0.043,
}


def transfer_energy_j(num_bytes: float) -> float:
    return num_bytes * LINK_ENERGY_PJ_PER_BYTE * 1e-12


def build_results() -> dict:
    rows = []
    for name, direction, num_bytes, bandwidth_gbps in TRANSFERS:
        rows.append(
            {
                "transfer_path": name,
                "direction": direction,
                "bytes": int(num_bytes),
                "data_size_gb": num_bytes / GB,
                "bandwidth_gbps": bandwidth_gbps,
                "latency_ms": num_bytes / (bandwidth_gbps * GB) * 1e3,
                "modeled_transfer_energy_mj": transfer_energy_j(num_bytes) * 1e3,
                "warm_per_batch": name in WARM_TRANSFER_NAMES,
            }
        )

    cold_bytes = sum(row["bytes"] for row in rows)
    cold_latency_ms = sum(row["latency_ms"] for row in rows)
    cold_energy_j = sum(row["modeled_transfer_energy_mj"] for row in rows) / 1e3
    warm_rows = [row for row in rows if row["warm_per_batch"]]
    warm_bytes = sum(row["bytes"] for row in warm_rows)
    warm_latency_ms = sum(row["latency_ms"] for row in warm_rows)
    warm_energy_j = sum(row["modeled_transfer_energy_mj"] for row in warm_rows) / 1e3

    non_package = sum(MULTIRAM_NON_PACKAGE_ENERGY_UNITS.values())
    multiram_pipeline_energy = non_package + cold_energy_j
    modeled_ee = A100_PIPELINE_ENERGY_UNITS / multiram_pipeline_energy

    return {
        "model": {
            "package_link_energy_pj_per_bit": LINK_ENERGY_PJ_PER_BIT,
            "package_link_energy_pj_per_byte": LINK_ENERGY_PJ_PER_BYTE,
            "command_latency_ns": 0.0,
            "command_latency_note": "not separately modeled",
            "provenance": "assumed package-link energy; not measured",
            "energy_equation": "E_transfer = bytes * 8 * 0.8 pJ/bit",
        },
        "transfer_rows": rows,
        "totals": {
            "cold_start_bytes": cold_bytes,
            "cold_start_data_size_gb": cold_bytes / GB,
            "cold_start_latency_ms": cold_latency_ms,
            "cold_start_transfer_energy_mj": cold_energy_j * 1e3,
            "warm_per_batch_bytes": warm_bytes,
            "warm_per_batch_data_size_mb": warm_bytes / MB,
            "warm_per_batch_latency_ms": warm_latency_ms,
            "warm_per_batch_transfer_energy_mj": warm_energy_j * 1e3,
        },
        "table_v_recalculation": {
            "scope": "draft Table V normalized-composition arithmetic only",
            "a100_pipeline_energy_units": A100_PIPELINE_ENERGY_UNITS,
            "multiram_non_package_energy_units": MULTIRAM_NON_PACKAGE_ENERGY_UNITS,
            "multiram_non_package_total": non_package,
            "multiram_package_energy_entry": cold_energy_j,
            "multiram_pipeline_energy": multiram_pipeline_energy,
            "modeled_pipeline_energy_efficiency_vs_a100_x": modeled_ee,
            "draft_table_v_previous_ee_vs_a100_x": 54.8,
            "relative_change_percent": (modeled_ee / 54.8 - 1.0) * 100.0,
            "caveat": (
                "This reproduces the draft Table V composition. It is a modeled "
                "normalized ratio, not a matched joule-level measurement and not "
                "a recalculation of the HMP2/IBDMDB 20.15x claim."
            ),
        },
        "hmp2_ibdmdb_recalculation": {
            "original_ee_vs_a100_x": 20.15,
            "status": "requires original absolute A100 and MultiRAM energy values",
            "cold_start_if_package_was_previously_excluded": (
                "EE_new = E_A100 / (E_MultiRAM_old + 0.0120256 J)"
            ),
            "cold_start_if_old_3_67_mj_package_value_was_included": (
                "EE_new = E_A100 / (E_MultiRAM_old + 0.0083556 J)"
            ),
            "equivalent_using_original_ratio_if_package_was_excluded": (
                "EE_new = 20.15 / (1 + 0.0120256 / E_MultiRAM_old)"
            ),
            "note": (
                "Adding positive package-link energy can only reduce EE relative "
                "to 20.15x; it cannot increase it to 50.28x."
            ),
        },
    }


def write_markdown(result: dict) -> None:
    totals = result["totals"]
    table_v = result["table_v_recalculation"]
    lines = [
        "# Package-Link Energy and EE Recalculation",
        "",
        "## Modeling Assumption",
        "",
        "- Package-link energy: **0.8 pJ/bit = 6.4 pJ/byte**.",
        "- Provenance: **assumed package-link energy; not measured**.",
        "- Command latency: `t_cmd = 0 ns` because it is not separately modeled.",
        "- Energy equation: `E_transfer = transferred bytes * 6.4 pJ/byte`.",
        "- Decimal SI units are used: `1 GB = 10^9 bytes`, `1 MB = 10^6 bytes`.",
        "",
        "## Recalculated Table IV",
        "",
        "| Transfer path | Direction | Data | Bandwidth | Latency | Transfer energy |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in result["transfer_rows"]:
        size = (
            f"{row['bytes'] / GB:.3f} GB"
            if row["bytes"] >= GB
            else f"{row['bytes'] / MB:.0f} MB"
        )
        bandwidth = (
            f"{row['bandwidth_gbps'] / 1000:.0f} TB/s"
            if row["bandwidth_gbps"] >= 1000
            else f"{row['bandwidth_gbps']:.0f} GB/s"
        )
        lines.append(
            f"| {row['transfer_path']} | {row['direction']} | {size} | "
            f"{bandwidth} | {row['latency_ms']:.4f} ms | "
            f"{row['modeled_transfer_energy_mj']:.4f} mJ |"
        )

    lines.extend(
        [
            f"| **Total, cold-start** | All rows | **{totals['cold_start_data_size_gb']:.3f} GB** | - | **{totals['cold_start_latency_ms']:.4f} ms** | **{totals['cold_start_transfer_energy_mj']:.4f} mJ** |",
            f"| **Total, per-batch warm** | Last four rows | **{totals['warm_per_batch_data_size_mb']:.0f} MB** | - | **{totals['warm_per_batch_latency_ms']:.4f} ms** | **{totals['warm_per_batch_transfer_energy_mj']:.4f} mJ** |",
            "",
            "The six listed transfers sum to 1.879 GB, not 1.867 GB. The previous",
            "energy values implied 2.0 pJ/byte (0.25 pJ/bit); the new assumption",
            "therefore increases every link-energy term by 3.2x.",
            "",
            "## Draft Table V Recalculation (Not the HMP2 Result)",
            "",
            "| Pipeline component | A100 pipeline | MultiRAM |",
            "|---|---:|---:|",
            "| Genomic alignment energy | 1.000 | 0.0025 |",
            "| Proteomic clustering energy | 1.000 | 0.0120 |",
            "| Proteomic OMS energy | 1.000 | 0.0160 |",
            f"| Package-level transfer energy | 0.300 | **{table_v['multiram_package_energy_entry']:.6f}** |",
            "| MoE inference energy | 1.000 | 0.0430 |",
            f"| **Modeled pipeline energy** | **{table_v['a100_pipeline_energy_units']:.3f}** | **{table_v['multiram_pipeline_energy']:.6f}** |",
            f"| **Modeled pipeline EE vs A100** | **1.00x** | **{table_v['modeled_pipeline_energy_efficiency_vs_a100_x']:.2f}x** |",
            "",
            f"The modeled full-pipeline EE changes from the draft's 54.8x to **{table_v['modeled_pipeline_energy_efficiency_vs_a100_x']:.2f}x** ",
            f"({table_v['relative_change_percent']:.2f}% relative change). Runtime and speedup are unchanged because bandwidth was not changed.",
            "",
            "> Methodological note: Table V mixes normalized component-energy terms with",
            "> a package-energy term expressed in joules. The 50.28x value reproduces that",
            "> existing composition for consistency; it must be labeled modeled normalized",
            "> EE, not a measured system-level joule comparison.",
            "",
            "## HMP2/IBDMDB 20.15x Claim",
            "",
            "The 50.28x result above is **not** an update of the HMP2/IBDMDB",
            "20.15x full-pipeline claim. It uses a different Table V baseline and",
            "different normalized component-energy inputs.",
            "",
            "For the HMP2 result, adding the cold-start package energy gives:",
            "",
            "```text",
            "EE_new = E_A100 / (E_MultiRAM_old + 0.0120256 J)",
            "       = 20.15 / (1 + 0.0120256 / E_MultiRAM_old)",
            "```",
            "",
            "If the old MultiRAM energy already included the previous 3.67 mJ",
            "package estimate, replace only the difference:",
            "",
            "```text",
            "EE_new = E_A100 / (E_MultiRAM_old + 0.0083556 J)",
            "```",
            "",
            "The original absolute A100 and MultiRAM joule values are required for",
            "an exact revised HMP2 number. In all cases the result must be below",
            "20.15x, although the change may be very small if compute energy is much",
            "larger than 12.0256 mJ.",
            "",
            "## LaTeX Values",
            "",
            "```latex",
            r"Package Link & \multicolumn{3}{c|}{FeNAND$\rightarrow$FeRAM: 256 GB/s; FeNAND$\rightarrow$M3D: 256 GB/s; FeRAM$\rightarrow$M3D: 512 GB/s} \\ \hline",
            r"Link Model & \multicolumn{3}{c|}{$t_{\mathrm{cmd}}=0$ ns (not separately modeled); $e_{\mathrm{link}}=0.8$ pJ/bit (assumed, not measured)} \\ \hline",
            "```",
            "",
        ]
    )
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    result = build_results()
    JSON_OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_markdown(result)
    print(json.dumps(result, indent=2))
    print(f"Wrote {JSON_OUT}")
    print(f"Wrote {MD_OUT}")


if __name__ == "__main__":
    main()
