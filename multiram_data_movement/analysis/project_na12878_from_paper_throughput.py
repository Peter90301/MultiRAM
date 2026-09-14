#!/usr/bin/env python3
"""Project NA12878 using the paper's normalized 100-bp throughput factors."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/home/tsl012/multiomic/multiram_data_movement")
RESULT_DIR = ROOT / "results_hga_e2e_subgraph_comparison"
CPU_PIM_PROJECTION = (
    ROOT
    / "results_matched_mhc_minigraph_hga_pim/"
    "PLATINUM_NA12878_CPU_HGA_PIM_PROJECTION.json"
)
VALID_HGA_PROJECTION = (
    RESULT_DIR / "PLATINUM_NA12878_FROM_VALID_HGA_E2E_PROJECTION.json"
)

# The target dataset averages 101.0 bases per FASTQ sequence, so use the
# 100-bp group from the supplied component-throughput figure.
THROUGHPUT_FACTORS_100BP = {
    "cpu_minigraph": 1.0,
    "cpu_amx_minigraph": 1.2,
    "a100_hga": 51.6,
    "h100_hga": 65.1,
    "segram": 145.0,
    "multiram": 283.0,
}


def duration(seconds: float) -> str:
    if seconds >= 86400:
        return f"{seconds / 86400:.2f} days"
    if seconds >= 3600:
        return f"{seconds / 3600:.2f} h"
    if seconds >= 60:
        return f"{seconds / 60:.2f} min"
    return f"{seconds:.2f} s"


def main() -> int:
    cpu_pim = json.loads(CPU_PIM_PROJECTION.read_text(encoding="utf-8"))
    valid_hga = json.loads(
        VALID_HGA_PROJECTION.read_text(encoding="utf-8")
    )

    cpu_compute_s = float(cpu_pim["cpu_minigraph"]["target_compute_s"])
    cpu_movement_s = float(
        cpu_pim["cpu_minigraph"]["movement_storage_service_s"]
    )
    cpu_overall_s = cpu_compute_s + cpu_movement_s
    hga_movement_s = float(
        valid_hga["a100_candidate_hga"][
            "prototype_4000_read_batch_replay"
        ]["movement_s"]
    )
    pim_movement_s = [
        float(value) for value in valid_hga["multiram"]["movement_s_range"]
    ]

    compute = {
        name: cpu_compute_s / factor
        for name, factor in THROUGHPUT_FACTORS_100BP.items()
    }
    a100_additive_s = compute["a100_hga"] + hga_movement_s
    h100_additive_s = compute["h100_hga"] + hga_movement_s
    pim_additive_s = [
        compute["multiram"] + movement for movement in pim_movement_s
    ]

    output = {
        "status": "projection_from_paper_component_throughput_factors",
        "target": valid_hga["target"],
        "throughput_factors_100bp": THROUGHPUT_FACTORS_100BP,
        "selection_reason": (
            "NA12878 averages approximately 101 bases per FASTQ sequence, "
            "so the 100-bp figure group is used instead of the across-length "
            "average group."
        ),
        "compute_only": {
            name: {
                "time_s": compute[name],
                "speedup_vs_cpu_minigraph_x": factor,
            }
            for name, factor in THROUGHPUT_FACTORS_100BP.items()
        },
        "conservative_additive_overall": {
            "cpu_minigraph": {
                "compute_s": cpu_compute_s,
                "movement_s": cpu_movement_s,
                "overall_s": cpu_overall_s,
                "movement_fraction_percent": (
                    100.0 * cpu_movement_s / cpu_overall_s
                ),
            },
            "a100_hga": {
                "compute_s": compute["a100_hga"],
                "movement_s": hga_movement_s,
                "overall_s": a100_additive_s,
                "movement_fraction_percent": (
                    100.0 * hga_movement_s / a100_additive_s
                ),
                "overall_speedup_vs_cpu_x": (
                    cpu_overall_s / a100_additive_s
                ),
            },
            "h100_hga": {
                "compute_s": compute["h100_hga"],
                "movement_s": hga_movement_s,
                "overall_s": h100_additive_s,
                "movement_fraction_percent": (
                    100.0 * hga_movement_s / h100_additive_s
                ),
                "overall_speedup_vs_cpu_x": (
                    cpu_overall_s / h100_additive_s
                ),
            },
            "multiram": {
                "compute_s": compute["multiram"],
                "movement_s_range": pim_movement_s,
                "overall_s_range": pim_additive_s,
                "movement_fraction_percent_range": [
                    100.0 * movement / overall
                    for movement, overall in zip(
                        pim_movement_s, pim_additive_s
                    )
                ],
                "overall_speedup_vs_cpu_x_range": [
                    cpu_overall_s / overall for overall in pim_additive_s
                ],
            },
        },
        "perfect_overlap_overall": {
            "cpu_minigraph_s": max(cpu_compute_s, cpu_movement_s),
            "a100_hga_s": max(compute["a100_hga"], hga_movement_s),
            "h100_hga_s": max(compute["h100_hga"], hga_movement_s),
            "multiram_s_range": [
                max(compute["multiram"], movement)
                for movement in pim_movement_s
            ],
        },
        "interpretation": [
            "The throughput factors are component-level normalized values, not measured full-dataset end-to-end runtimes.",
            "The CPU full-dataset compute projection anchors the normalized factors.",
            "The conservative overall model adds serialized movement service to compute; actual execution may overlap them.",
            "This model is distinct from the corrected public-HGA implementation projection and must not be mixed with it.",
            "The component-throughput model assumes optimized resident HGA behavior and does not include the public wrapper's dense 25-kb DP behavior.",
        ],
    }

    json_path = RESULT_DIR / "PLATINUM_NA12878_PAPER_THROUGHPUT_PROJECTION.json"
    json_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    c = output["conservative_additive_overall"]
    p = output["perfect_overlap_overall"]
    lines = [
        "# NA12878 Projection Using Paper Throughput Factors",
        "",
        "> This is a projection from the normalized component-throughput figure. "
        "It is separate from the corrected public-HGA code measurement.",
        "",
        "## Selected Throughput Group",
        "",
        f"- NA12878 average sequence length: {float(valid_hga['target']['average_read_length']):.3f} bp.",
        "- Selected figure group: **100 bp**.",
        "- A100 + HGA: **51.6x** CPU + minigraph.",
        "- H100 + HGA: **65.1x** CPU + minigraph.",
        "- SeGraM: **145x** CPU + minigraph.",
        "- MultiRAM: **283x** CPU + minigraph.",
        "",
        "## Core Compute Only",
        "",
        "| Platform | Throughput vs CPU | Projected compute time |",
        "| --- | ---: | ---: |",
        f"| CPU + minigraph | 1.0x | **{compute['cpu_minigraph']:,.1f} s ({duration(compute['cpu_minigraph'])})** |",
        f"| CPU (AMX) + minigraph | 1.2x | **{compute['cpu_amx_minigraph']:,.1f} s ({duration(compute['cpu_amx_minigraph'])})** |",
        f"| A100 + HGA | 51.6x | **{compute['a100_hga']:,.1f} s ({duration(compute['a100_hga'])})** |",
        f"| H100 + HGA | 65.1x | **{compute['h100_hga']:,.1f} s ({duration(compute['h100_hga'])})** |",
        f"| SeGraM | 145x | **{compute['segram']:,.1f} s ({duration(compute['segram'])})** |",
        f"| MultiRAM | 283x | **{compute['multiram']:,.1f} s ({duration(compute['multiram'])})** |",
        "",
        "Under the paper-throughput model, A100 + HGA is correctly faster than "
        "CPU + minigraph: 335.2 seconds versus 17,297.0 seconds for core compute.",
        "",
        "## Overall Including Movement",
        "",
        "| Platform | Compute | Movement service | Conservative overall | Movement / overall | Overall speedup vs CPU |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| CPU + minigraph | {c['cpu_minigraph']['compute_s']:,.1f} s | {c['cpu_minigraph']['movement_s']:,.1f} s | **{c['cpu_minigraph']['overall_s']:,.1f} s ({duration(c['cpu_minigraph']['overall_s'])})** | {c['cpu_minigraph']['movement_fraction_percent']:.2f}% | 1.00x |",
        f"| A100 + HGA | {c['a100_hga']['compute_s']:,.1f} s | {c['a100_hga']['movement_s']:,.1f} s | **{c['a100_hga']['overall_s']:,.1f} s ({duration(c['a100_hga']['overall_s'])})** | {c['a100_hga']['movement_fraction_percent']:.2f}% | {c['a100_hga']['overall_speedup_vs_cpu_x']:.2f}x |",
        f"| H100 + HGA | {c['h100_hga']['compute_s']:,.1f} s | {c['h100_hga']['movement_s']:,.1f} s | **{c['h100_hga']['overall_s']:,.1f} s ({duration(c['h100_hga']['overall_s'])})** | {c['h100_hga']['movement_fraction_percent']:.2f}% | {c['h100_hga']['overall_speedup_vs_cpu_x']:.2f}x |",
        f"| MultiRAM | {c['multiram']['compute_s']:,.1f} s | {c['multiram']['movement_s_range'][0]:.1f}-{c['multiram']['movement_s_range'][1]:.1f} s | **{c['multiram']['overall_s_range'][0]:.1f}-{c['multiram']['overall_s_range'][1]:.1f} s ({duration(c['multiram']['overall_s_range'][0])}-{duration(c['multiram']['overall_s_range'][1])})** | {c['multiram']['movement_fraction_percent_range'][0]:.2f}-{c['multiram']['movement_fraction_percent_range'][1]:.2f}% | {c['multiram']['overall_speedup_vs_cpu_x_range'][1]:.2f}-{c['multiram']['overall_speedup_vs_cpu_x_range'][0]:.2f}x |",
        "",
        "The GPU core is fast in this model, but its full-data projection becomes "
        "storage-bound: movement accounts for about 95% of A100 overall time. "
        "This is exactly the motivation for keeping data near the compute.",
        "",
        "## Perfect-Overlap Bound",
        "",
        f"- CPU + minigraph: {p['cpu_minigraph_s']:,.1f} s ({duration(p['cpu_minigraph_s'])}).",
        f"- A100 + HGA: {p['a100_hga_s']:,.1f} s ({duration(p['a100_hga_s'])}).",
        f"- H100 + HGA: {p['h100_hga_s']:,.1f} s ({duration(p['h100_hga_s'])}).",
        f"- MultiRAM: {p['multiram_s_range'][0]:.1f}-{p['multiram_s_range'][1]:.1f} s.",
        "",
        "## Do Not Mix the Two HGA Models",
        "",
        "- **Paper-throughput model:** optimized component throughput; A100 HGA is 51.6x faster than CPU compute at 100 bp.",
        "- **Public-code measurement:** candidate-filtered 25-kb dense DP plus four HGA processes; its corrected kernel was much slower.",
        "",
        "Use the paper-throughput table for the architecture-level projection. "
        "Use the public-code result only as a prototype implementation result or "
        "validation study; do not present both as measurements of the same HGA configuration.",
    ]
    report_path = RESULT_DIR / "PLATINUM_NA12878_PAPER_THROUGHPUT_PROJECTION.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "report": str(report_path),
        "json": str(json_path),
        "a100_hga_compute_s": compute["a100_hga"],
        "a100_hga_overall_s": a100_additive_s,
        "multiram_compute_s": compute["multiram"],
        "multiram_overall_s_range": pim_additive_s,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
