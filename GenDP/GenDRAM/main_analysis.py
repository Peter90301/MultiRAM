# # 檔名: main_analysis.py (V7 - 最終版)

# from .config_unified import CfgDRAM3D, CfgPIM
# from .simulator_unified import PIM_SoC_Simulator_V2, DatabaseManager

# def main():
#     print("======================================================")
#     print("=     ANALYZING FINAL 3D-PIM (Capacity-Aware)      =")
#     print("======================================================")
    
#     # --- 1. 初始化硬體與資料庫 ---
#     cfg_dram = CfgDRAM3D()
#     cfg_pim = CfgPIM()
    
#     # 建立資料庫管理器並對映資料
#     db_manager = DatabaseManager(cfg_dram, cfg_pim)
#     db_manager.map_genome_data(
#         ptr_size_gb=4.0, #
#         cal_size_gb=13.4, #
#         ref_genome_size_gb=3.0
#     )
    
#     # --- 2. 初始化模擬器 ---
#     sim = PIM_SoC_Simulator_V2(database_manager=db_manager)
    
#     # --- 3. 執行模擬 ---
#     time, energy = sim.run_full_pipeline(num_queries=1000)
#     power = energy.sum() / time if time > 0 else 0
    
#     # --- 4. 輸出報告 ---
#     print("\n--- Final Pipeline PPA Report ---")
#     print(f"  Total Latency (Pipelined) : {time * 1e3:.2f} ms")
#     print(f"  Total Energy              : {energy.sum() * 1e6:.2f} uJ")
#     print(f"  Average Power             : {power:.2f} W")
    
# if __name__ == '__main__':
#     main()

# 檔名: main_analysis.py (最終繞過檔案讀取版)

import argparse
import os
import re
from collections import Counter

from config_unified import CfgDRAM3D, CfgPIM
from simulator_unified import Final_PIM_Simulator, DatabaseManager
from graph_aligner_minigraph_like import (
    MinigraphLikeAligner,
    build_linear_graph_from_fasta,
    build_demo_graph,
    parse_fasta,
    parse_fastq,
    parse_gfa,
)


PAPER_TRCD_8TIERS_NS = list(CfgDRAM3D.TRCD_TIERS_NS)

# 我們不再需要這個函式，因為我們將手動提供參數
# def analyze_sequences_from_fasta(...):
#     ...

def run_seq_to_graph_alignment(
    gfa_path: str | None,
    ref_fasta: str | None,
    query_fasta: str | None,
    query_fastq: str | None,
    max_queries: int,
    target_total_bases: int = 0,
    read_length_target: int = 0,
    aligner_k: int | None = None,
    aligner_w: int | None = None,
    aligner_max_occ: int = 256,
    aligner_band: int = 96,
):
    if gfa_path:
        graph = parse_gfa(gfa_path)
        print(f"Loaded graph from GFA: {gfa_path}")
    elif ref_fasta:
        # 1KB chunks to match mapping policy: target_channel = (node_id / 1KB) mod 32.
        graph = build_linear_graph_from_fasta(ref_fasta, chunk_size=1024)
        print(f"Built linear graph from FASTA: {ref_fasta}")
    else:
        graph = build_demo_graph()
        print("Using built-in demo graph (provide --gfa or --ref-fasta for real graph)")

    if query_fastq:
        queries = parse_fastq(query_fastq)
        print(f"Loaded {len(queries)} query sequences from FASTQ: {query_fastq}")
    elif query_fasta:
        queries = parse_fasta(query_fasta)
        print(f"Loaded {len(queries)} query sequences from FASTA: {query_fasta}")
    else:
        queries = [
            "ACCGTATGGCTTACGCTTAGG",
            "ACCGTATGGCGGAACCTTAGG",
            "ACCGTATGGCTTACGCTTGG",
        ]
        print("Using built-in demo query batch (provide --query-fastq or --query-fasta for real reads)")

    if max_queries > 0 and len(queries) > max_queries:
        queries = queries[:max_queries]
        print(f"Trimmed query batch to max_queries={max_queries}")

    if read_length_target > 0:
        fixed = [q[:read_length_target] for q in queries if len(q) >= read_length_target]
        if not fixed:
            raise ValueError(
                f"No reads found with length >= read_length_target={read_length_target}"
            )
        queries = fixed
        print(
            f"Applied fixed read_length_target={read_length_target} "
            f"(kept_reads={len(queries)})"
        )

    if target_total_bases > 0:
        trimmed = []
        acc_bases = 0
        for q in queries:
            if acc_bases >= target_total_bases:
                break
            trimmed.append(q)
            acc_bases += len(q)
        if trimmed:
            queries = trimmed
            print(
                f"Trimmed query batch to target_total_bases={target_total_bases} "
                f"(actual_total_bases={acc_bases})"
            )

    min_graph_len = min(len(n.seq) for n in graph.nodes.values())
    min_query_len = min(len(q) for q in queries)
    min_len = max(1, min(min_graph_len, min_query_len))
    auto_k = max(3, min(15, min_len // 2))
    auto_w = max(4, min(16, auto_k + 3))

    use_k = aligner_k if aligner_k is not None else auto_k
    use_w = aligner_w if aligner_w is not None else auto_w
    use_k = max(3, use_k)
    use_w = max(4, use_w)

    aligner = MinigraphLikeAligner(k=use_k, w=use_w, max_occ=max(1, aligner_max_occ), band=max(8, aligner_band))
    print(
        "  - Minigraph-like params: "
        f"k={use_k}, w={use_w}, max_occ={max(1, aligner_max_occ)}, band={max(8, aligner_band)}"
    )
    aligner.build_index(graph)

    results = []
    for q in queries:
        if len(q) < aligner.k:
            continue
        results.append(aligner.align(q))

    if not results:
        raise ValueError("No valid query sequences for alignment")

    total_anchors = sum(r["num_anchors"] for r in results)
    total_chain_anchors = sum(r["chain_anchors"] for r in results)
    total_dp_cells = sum(r["dp_cells"] for r in results)
    avg_edit_distance = sum(r["edit_distance"] for r in results) / len(results)

    print("\n--- Sequence-to-Graph Alignment Summary (Minigraph-like) ---")
    print(f"  Queries aligned           : {len(results)}")
    print(f"  Avg anchors/query         : {total_anchors / len(results):.2f}")
    print(f"  Avg chain anchors/query   : {total_chain_anchors / len(results):.2f}")
    print(f"  Avg DP cells/query        : {total_dp_cells / len(results):.2f}")
    print(f"  Avg edit distance         : {avg_edit_distance:.2f}")

    # --- Mapping summary for timing estimation ---
    # Policy:
    #   target_channel = (node_id / 1KB) mod NUM_CHANNELS
    #   hot nodes on lower tiers, cold nodes on upper tiers
    node_access_counter: Counter[int] = Counter()
    per_query_paths = []

    def _extract_node_num(node_id: str) -> int:
        m = re.search(r"(\d+)$", str(node_id))
        return int(m.group(1)) if m else 0

    for r in results:
        path = r.get("graph_path", [])
        nums = [_extract_node_num(nid) for nid in path]
        per_query_paths.append(nums)
        for n in nums:
            node_access_counter[n] += 1

    unique_nodes = list(node_access_counter.keys())
    sorted_nodes = sorted(unique_nodes, key=lambda n: node_access_counter[n], reverse=True)
    total_unique = max(1, len(sorted_nodes))

    num_tiers = CfgDRAM3D.NUM_TIERS
    num_channels = CfgDRAM3D.NUM_CHANNELS

    # Rank-based tier assignment: hottest nodes -> tier 0 ... coldest tier.
    node_to_tier = {}
    for rank, node in enumerate(sorted_nodes):
        tier = int((rank * num_tiers) / total_unique)
        node_to_tier[node] = min(num_tiers - 1, max(0, tier))

    # Paper-aligned 8-tier latency table (tRCD in ns), extracted from Table/Section 5.1.3.
    # tRCD = [2.29, 3.92, 5.99, 8.50, 11.44, 14.82, 18.63, 22.88] ns.
    tier_samples = PAPER_TRCD_8TIERS_NS

    total_node_accesses = 0
    total_tier_read_ns = 0.0
    total_channel_switches = 0
    total_tier_hops = 0

    for nodes in per_query_paths:
        if not nodes:
            continue
        prev_channel = None
        prev_tier = None
        for node_num in nodes:
            tier = node_to_tier.get(node_num, num_tiers - 1)
            channel = node_num % num_channels

            total_node_accesses += 1
            total_tier_read_ns += tier_samples[tier]
            if prev_channel is not None and channel != prev_channel:
                total_channel_switches += 1
            if prev_tier is not None:
                total_tier_hops += abs(tier - prev_tier)
            prev_channel = channel
            prev_tier = tier

    denom_q = max(1, len(results))
    avg_node_accesses_per_query = total_node_accesses / denom_q
    avg_tier_read_ns = (total_tier_read_ns / total_node_accesses) if total_node_accesses > 0 else tier_samples[-1]
    avg_channel_switches_per_query = total_channel_switches / denom_q
    avg_tier_hops_per_query = total_tier_hops / denom_q

    mapping_summary = {
        "mode": "mapped",
        "policy": (
            f"target_channel=(node_id/1KB)%{num_channels}, "
            f"hot->tier0, cold->tier{num_tiers - 1}"
        ),
        "baseline_policy": "no-tier-aware + no-modulo-aware",
        "paper_tRCD_8tiers_ns": PAPER_TRCD_8TIERS_NS,
        "num_tiers": num_tiers,
        "capacity_per_tier_gb": CfgDRAM3D.CAPACITY_PER_TIER_GB,
        "num_channels": num_channels,
        "pus_per_channel": CfgDRAM3D.PUS_PER_CHANNEL,
        "bank_groups_per_channel": CfgDRAM3D.BANK_GROUPS_PER_CHANNEL,
        "banks_per_bank_group": CfgDRAM3D.BANKS_PER_BANK_GROUP,
        "banks_per_channel": CfgDRAM3D.BANKS_PER_CHANNEL,
        "bank_capacity_gbit": CfgDRAM3D.BANK_CAPACITY_GBIT,
        "avg_node_accesses_per_query": avg_node_accesses_per_query,
        "avg_tier_read_ns": avg_tier_read_ns,
        "avg_channel_switches_per_query": avg_channel_switches_per_query,
        "channel_switch_penalty_ns": 0.25,
        "avg_tier_hops_per_query": avg_tier_hops_per_query,
        # Node-path baseline (no tier-aware + no modulo-aware):
        "baseline_node_tier_read_ns": sum(PAPER_TRCD_8TIERS_NS) / len(PAPER_TRCD_8TIERS_NS),
        "baseline_node_channel_switches_per_query": max(1.0, avg_node_accesses_per_query * 0.70),
        "baseline_channel_switch_penalty_ns": 0.90,
        "baseline_node_tier_hops_per_query": max(1.0, avg_node_accesses_per_query * 3.0),
        # Seeding PTR/CAL model (paper-aligned latency-critical mapping):
        # map latency-sensitive tables to tier-0 for mapped mode.
        "ptr_cal_lookups_per_query": max(1.0, total_anchors / len(results)),
        "ptr_cal_entry_bytes": 64.0,
        "alignment_read_bytes_per_query": max(4096.0, total_dp_cells / len(results) * 0.5),
        "baseline_dram_read_inflation": 1.8,
        "mapped_ptr_cal_tier_read_ns": PAPER_TRCD_8TIERS_NS[0],
        "baseline_ptr_cal_tier_read_ns": sum(PAPER_TRCD_8TIERS_NS) / len(PAPER_TRCD_8TIERS_NS),
        "mapped_ptr_cal_channel_switches_per_query": max(1.0, total_anchors / len(results)) * 0.15,
        "baseline_ptr_cal_channel_switches_per_query": max(1.0, total_anchors / len(results)) * 0.55,
        "mapped_ptr_cal_avg_tier_hops_per_query": 0.0,
        "baseline_ptr_cal_avg_tier_hops_per_query": max(1.0, total_anchors / len(results)) * 3.5,
        "baseline_no_modulo_conflict_penalty_ns": max(1.0, total_anchors / len(results)) * 0.8,
    }

    print("\n--- Mapping Summary (8-tier + cyclic channel) ---")
    print(f"  Policy                    : {mapping_summary['policy']}")
    print(f"  Baseline policy           : {mapping_summary['baseline_policy']}")
    print(f"  Paper tRCD tiers (ns)     : {mapping_summary['paper_tRCD_8tiers_ns']}")
    print(f"  Avg node accesses/query   : {avg_node_accesses_per_query:.2f}")
    print(f"  Avg tier read latency     : {avg_tier_read_ns:.3f} ns")
    print(f"  Avg channel switches/query: {avg_channel_switches_per_query:.2f}")
    print(f"  Avg tier hops/query       : {avg_tier_hops_per_query:.2f}")

    return {
        "num_queries": len(results),
        "avg_query_len": int(sum(len(q) for q in queries) / len(queries)),
        "ops_per_seed_task": max(1, int(total_anchors / len(results) * 16)),
        "ops_per_align_task": max(1, int(total_dp_cells / len(results))),
        "mapping_summary": mapping_summary,
    }


def main():
    default_data_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "mhc_test")
    )
    default_ref_fasta = os.path.join(default_data_dir, "mhc_ref.fa")
    default_query_fastq = os.path.join(default_data_dir, "query_reads_100bp.fq")

    parser = argparse.ArgumentParser(description="GenDRAM sequence-to-graph alignment + PIM simulation")
    parser.add_argument("--gfa", type=str, default=None, help="Input variation graph in GFA format")
    parser.add_argument("--ref-fasta", type=str, default=default_ref_fasta, help="Reference FASTA used to build a linear graph when --gfa is not given")
    parser.add_argument("--query-fasta", type=str, default=None, help="Query reads in FASTA format")
    parser.add_argument("--query-fastq", type=str, default=default_query_fastq, help="Query reads in FASTQ format")
    parser.add_argument("--max-queries", type=int, default=0, help="Maximum number of query sequences to align (0 means all)")
    args = parser.parse_args()

    if args.gfa is None and not os.path.exists(args.ref_fasta):
        args.ref_fasta = None
    if args.query_fastq and not os.path.exists(args.query_fastq):
        args.query_fastq = None

    # 初始化
    cfg_dram = CfgDRAM3D()
    cfg_pim = CfgPIM()
    print("\n--- M3D DRAM Organization ---")
    print(f"  Capacity           : {cfg_dram.TOTAL_CAPACITY_GB} GB "
          f"({cfg_dram.NUM_TIERS} tiers x {cfg_dram.CAPACITY_PER_TIER_GB} GB)")
    print(f"  Physical layers    : {cfg_dram.NUM_LAYERS} "
          f"({cfg_dram.PHYSICAL_LAYERS_PER_TIER} per tier)")
    print(f"  Channels / PUs     : {cfg_dram.NUM_CHANNELS} channels, "
          f"{cfg_dram.PUS_PER_CHANNEL} PU/channel")
    print(f"  Bank organization  : {cfg_dram.BANK_GROUPS_PER_CHANNEL} bank groups/channel, "
          f"{cfg_dram.BANKS_PER_BANK_GROUP} banks/bank group, "
          f"{cfg_dram.BANKS_PER_CHANNEL} banks/channel")
    print(f"  Bank capacity      : {cfg_dram.BANK_CAPACITY_GBIT} Gb/bank "
          f"({cfg_dram.TOTAL_BANKS} banks total)")
    db_manager = DatabaseManager(cfg_dram, cfg_pim)
    db_manager.map_genome_data(
        ptr_size_gb=4.0, 
        cal_size_gb=13.4, 
        ref_genome_size_gb=3.0
    )
    
    print("\n" + "="*50)
    print("=      STARTING BIOINFORMATICS SIMULATION      =")
    print("="*50)

    bio_params = run_seq_to_graph_alignment(
        gfa_path=args.gfa,
        ref_fasta=args.ref_fasta,
        query_fasta=args.query_fasta,
        query_fastq=args.query_fastq,
        max_queries=args.max_queries,
    )
    print("\nDerived BIO workload from sequence-to-graph alignment:")
    print(f"  - num_queries       : {bio_params['num_queries']}")
    print(f"  - ops_per_seed_task : {bio_params['ops_per_seed_task']}")
    print(f"  - ops_per_align_task: {bio_params['ops_per_align_task']}")
    
    # --- 執行生物資訊學模擬 ---
    sim_bio = Final_PIM_Simulator(application='bio', database_manager=db_manager)
    
    # 修改 simulator 實例中的配置
    if hasattr(sim_bio.cfg_app, 'ops_per_seed_task'):
        sim_bio.cfg_app.ops_per_seed_task = bio_params["ops_per_seed_task"]
    if hasattr(sim_bio.cfg_app, 'ops_per_align_task'):
        sim_bio.cfg_app.ops_per_align_task = bio_params["ops_per_align_task"] 

    time_bio, energy_bio = sim_bio.run_bio_pipeline(
        num_queries=bio_params["num_queries"],
        mapping_summary=bio_params.get("mapping_summary"),
    )
    power_bio = energy_bio.sum() / time_bio if time_bio > 0 else 0
    
    print("\n--- Bioinformatics PPA Report ---")
    print(f"  Total Latency (Pipelined) : {time_bio * 1e3:.4f} ms")
    print(f"  Average Power             : {power_bio:.2f} W")

    # --- 執行 APSP 模擬 (這部分不受影響) ---
    print("\n" + "="*50)
    print("=           STARTING APSP SIMULATION           =")
    print("="*50)
    sim_apsp = Final_PIM_Simulator(application='apsp', database_manager=db_manager)
    time_apsp, energy_apsp = sim_apsp.run_apsp_algorithm(num_vertices=8192)
    power_apsp = energy_apsp.sum() / time_apsp if time_apsp > 0 else 0
    print("\n--- APSP PPA Report ---")
    print(f"  Total Latency             : {time_apsp:.4f} s")
    print(f"  Average Power             : {power_apsp:.2f} W")

if __name__ == '__main__':
    main()
