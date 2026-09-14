# 檔名: simulator_unified.py (最終能量模型修正版 V2)

import math
from dataclasses import dataclass
import random
from typing import Optional

from config_unified import CfgDRAM3D, CfgPIM, CfgBioPIM, CfgAPSP
from components_unified import LogicDie_Hardware

# ... (DatabaseManager 和 Energy class 保持不變) ...
class DatabaseManager:
    def __init__(self, cfg_dram, cfg_pim):
        self.cfg_dram = cfg_dram
        self.cfg_pim = cfg_pim
        self.tiers = [{} for _ in range(cfg_dram.NUM_TIERS)]
        print(
            f"DatabaseManager: Initialized {cfg_dram.NUM_TIERS} tiers x "
            f"{cfg_dram.CAPACITY_PER_TIER_GB} GB "
            f"({cfg_dram.NUM_CHANNELS} channels, {cfg_dram.BANKS_PER_CHANNEL} banks/channel)."
        )
    def map_genome_data(self, ptr_size_gb, cal_size_gb, ref_genome_size_gb):
        self._distribute_data("RefGenome", ref_genome_size_gb, range(0, self.cfg_pim.NUM_ALIGNMENT_TIERS))
        self._distribute_data("PTR_Table", ptr_size_gb, range(self.cfg_pim.NUM_ALIGNMENT_TIERS, self.cfg_dram.NUM_TIERS))
        self._distribute_data("CAL_Table", cal_size_gb, range(0, self.cfg_dram.NUM_TIERS))
    def _distribute_data(self, name, size_gb, tier_range):
        print(f"  Mapping {size_gb} GB of {name} to tiers {tier_range.start}-{tier_range.stop-1}...")
    def get_data_location(self, data_type, logical_address):
        if data_type == "CAL_Table": return random.randint(0, self.cfg_dram.NUM_TIERS - 1)
        elif data_type == "RefGenome": return random.randint(0, self.cfg_pim.NUM_ALIGNMENT_TIERS - 1)
        return 0

@dataclass
class Energy:
    e_dram_act: float = 0.; e_noc: float = 0.; e_compute: float = 0.; e_sram: float = 0.
    def sum(self): return self.e_dram_act + self.e_noc + self.e_compute + self.e_sram
    def reset(self):
        for field in self.__dataclass_fields__: setattr(self, field, 0.0)


class Final_PIM_Simulator:
    def __init__(self, application: str, database_manager: DatabaseManager):
        # ... (init 部分不變) ...
        self.app = application.lower()
        self.db = database_manager
        self.cfg_dram = CfgDRAM3D()
        self.cfg_pim = CfgPIM()
        self.energy = Energy()
        self.hw = LogicDie_Hardware()
        if self.app == 'bio': self.cfg_app = CfgBioPIM()
        elif self.app == 'apsp': self.cfg_app = CfgAPSP()
        else: raise ValueError("Unsupported application.")

    def _mapping_overhead_from_summary(self, num_queries: int, mapping_summary: dict) -> tuple[float, float, float]:
        """
        Estimate mapping overhead with 8-tier policy:
        - hot nodes on lower tiers (low latency)
        - cold nodes on upper tiers (high latency)
        - cyclic channel mapping: target_channel = (node_id / 1KB) mod 32
        Returns: (extra_time_s, extra_e_noc_j, avg_tier_read_ns)
        """
        if not mapping_summary:
            return 0.0, 0.0, 0.0

        mode = str(mapping_summary.get("mode", "mapped")).lower()

        avg_node_accesses = float(mapping_summary.get("avg_node_accesses_per_query", 0.0))
        if mode == "baseline":
            # Baseline: no tier-aware placement and no modulo-aware distribution.
            avg_tier_read_ns = float(mapping_summary.get("baseline_node_tier_read_ns", 11.44))
            avg_channel_switches = float(
                mapping_summary.get("baseline_node_channel_switches_per_query", avg_node_accesses * 0.7)
            )
            channel_switch_penalty_ns = float(mapping_summary.get("baseline_channel_switch_penalty_ns", 0.9))
            avg_tier_hops = float(
                mapping_summary.get("baseline_node_tier_hops_per_query", avg_node_accesses * 3.0)
            )
        else:
            avg_tier_read_ns = float(mapping_summary.get("avg_tier_read_ns", 0.0))
            avg_channel_switches = float(mapping_summary.get("avg_channel_switches_per_query", 0.0))
            channel_switch_penalty_ns = float(mapping_summary.get("channel_switch_penalty_ns", 0.25))
            avg_tier_hops = float(mapping_summary.get("avg_tier_hops_per_query", 0.0))

        # Use the configured inter-layer hop latency as the base for vertical movement.
        tier_hop_penalty_ns = self.cfg_dram.INTER_LAYER_LATENCY_PER_HOP * 1e9

        per_query_ns = (
            avg_node_accesses * avg_tier_read_ns
            + avg_channel_switches * channel_switch_penalty_ns
            + avg_tier_hops * tier_hop_penalty_ns
        )
        extra_time_s = (per_query_ns * num_queries) * 1e-9

        # Approximate vertical data movement energy for 1KB node fetches.
        bits_per_node = 1024 * 8
        extra_e_noc_j = (
            num_queries
            * avg_node_accesses
            * avg_tier_hops
            * bits_per_node
            * self.cfg_dram.INTER_LAYER_ENERGY_PER_BIT_PER_HOP
        )

        return extra_time_s, extra_e_noc_j, avg_tier_read_ns

    def _seeding_ptr_cal_overhead(
        self,
        num_queries: int,
        ops_per_seed_task: int,
        mapping_summary: dict,
    ) -> tuple[float, float, float]:
        """
        Model latency/energy for seeding PTR/CAL accesses.
        This captures the major mapping benefit in the paper: placing latency-critical
        seeding tables in fast tier-0 instead of mixed/high tiers.

        Returns: (extra_time_s, extra_energy_j, avg_ptr_cal_tier_read_ns)
        """
        mode = str(mapping_summary.get("mode", "mapped")).lower()

        lookups_per_query = float(
            mapping_summary.get("ptr_cal_lookups_per_query", max(1.0, ops_per_seed_task / 32.0))
        )
        channel_switch_penalty_ns = float(mapping_summary.get("channel_switch_penalty_ns", 0.25))
        tier_hop_penalty_ns = self.cfg_dram.INTER_LAYER_LATENCY_PER_HOP * 1e9
        entry_bytes = float(mapping_summary.get("ptr_cal_entry_bytes", 64.0))

        if mode == "baseline":
            tier_read_ns = float(mapping_summary.get("baseline_ptr_cal_tier_read_ns", 11.44))
            channel_switches_per_query = float(
                mapping_summary.get("baseline_ptr_cal_channel_switches_per_query", lookups_per_query * 0.55)
            )
            tier_hops_per_query = float(
                mapping_summary.get("baseline_ptr_cal_avg_tier_hops_per_query", lookups_per_query * 3.5)
            )
            modulo_conflict_penalty_ns = float(
                mapping_summary.get("baseline_no_modulo_conflict_penalty_ns", lookups_per_query * 0.8)
            )
        else:
            tier_read_ns = float(mapping_summary.get("mapped_ptr_cal_tier_read_ns", 2.29))
            channel_switches_per_query = float(
                mapping_summary.get("mapped_ptr_cal_channel_switches_per_query", lookups_per_query * 0.15)
            )
            tier_hops_per_query = float(
                mapping_summary.get("mapped_ptr_cal_avg_tier_hops_per_query", 0.0)
            )
            modulo_conflict_penalty_ns = 0.0

        per_query_ns = (
            lookups_per_query * tier_read_ns
            + channel_switches_per_query * channel_switch_penalty_ns
            + tier_hops_per_query * tier_hop_penalty_ns
            + modulo_conflict_penalty_ns
        )
        extra_time_s = (per_query_ns * num_queries) * 1e-9

        # Dynamic read energy for PTR/CAL entries + inter-tier movement.
        bits = entry_bytes * 8
        e_read = num_queries * lookups_per_query * bits * self.cfg_dram.energy_per_bit_int
        e_hops = (
            num_queries
            * lookups_per_query
            * max(0.0, tier_hops_per_query / max(1.0, lookups_per_query))
            * bits
            * self.cfg_dram.INTER_LAYER_ENERGY_PER_BIT_PER_HOP
        )

        return extra_time_s, (e_read + e_hops), tier_read_ns

    def _calculate_vertical_transfer_cost(self, data_size_bits, from_layer, to_layer):
        # ... (這個函式保持不變) ...
        num_hops = abs(to_layer - from_layer); latency_transfer = data_size_bits / self.cfg_pim.VERTICAL_BUS_BPS
        latency_propagation = num_hops * self.cfg_dram.INTER_LAYER_LATENCY_PER_HOP; total_latency = latency_transfer + latency_propagation
        total_energy = data_size_bits * num_hops * self.cfg_dram.INTER_LAYER_ENERGY_PER_BIT_PER_HOP
        return total_latency, total_energy

    def run_bio_pipeline(self, num_queries: int, mapping_summary: Optional[dict] = None):
        # ... (此函式與上一版相同，保持不變) ...
        if self.app != 'bio': raise Exception("Not configured for BIO pipeline")
        self.energy.reset()
        total_seeding_ops = num_queries * self.cfg_app.ops_per_seed_task
        total_seeding_cycles = math.ceil(total_seeding_ops / self.cfg_pim.TOTAL_SEARCH_PES)
        t_seeding_stage = total_seeding_cycles * self.cfg_pim.CLK_PERIOD
        total_alignment_ops = num_queries * self.cfg_app.ops_per_align_task
        total_alignment_cycles = math.ceil(total_alignment_ops / self.cfg_pim.TOTAL_COMPUTE_PES)
        t_alignment_stage = total_alignment_cycles * self.cfg_pim.CLK_PERIOD
        total_time = max(t_seeding_stage, t_alignment_stage)
        self.energy.e_compute += total_seeding_cycles * self.hw.TOTAL_SEEDING_ENERGY_PER_CYCLE
        self.energy.e_compute += total_alignment_cycles * self.hw.TOTAL_ALIGNMENT_ENERGY_PER_CYCLE
        total_ops = total_seeding_ops + total_alignment_ops
        e_sram_per_op = (0.0008876e-9 * total_seeding_ops + 0.0074776e-9 * total_alignment_ops) / total_ops
        self.energy.e_sram = total_ops * 3 * e_sram_per_op
        # DRAM read energy model:
        # - If mapping summary is available, use workload-dependent traffic.
        # - Otherwise, fall back to legacy fixed-capacity approximation.
        if mapping_summary:
            mode = str(mapping_summary.get("mode", "mapped")).lower()
            avg_node_accesses = float(mapping_summary.get("avg_node_accesses_per_query", 0.0))
            ptr_cal_lookups = float(mapping_summary.get("ptr_cal_lookups_per_query", 1.0))
            ptr_cal_entry_bytes = float(mapping_summary.get("ptr_cal_entry_bytes", 64.0))
            alignment_read_bytes_per_query = float(
                mapping_summary.get(
                    "alignment_read_bytes_per_query",
                    max(2048.0, self.cfg_app.ops_per_align_task * self.cfg_app.OP_SIZE_BITS / 16.0),
                )
            )

            bytes_per_query = (
                avg_node_accesses * 1024.0
                + ptr_cal_lookups * ptr_cal_entry_bytes
                + alignment_read_bytes_per_query
            )

            if mode == "baseline":
                # Baseline without mapping typically incurs higher off-bank traffic.
                bytes_per_query *= float(mapping_summary.get("baseline_dram_read_inflation", 1.8))

            total_read_bits = num_queries * bytes_per_query * 8.0
            self.energy.e_dram_act = total_read_bits * self.cfg_dram.energy_per_bit_int
        else:
            total_data_gb = 4.0 + 13.4 + 3.0
            e_dram_read = (total_data_gb * 1024**3 * 8 * 0.5) * self.cfg_dram.energy_per_bit_int
            self.energy.e_dram_act = e_dram_read

        # Mapping-aware overhead: tier/channel placement affects data access time.
        t_map_extra, e_map_noc, _ = self._mapping_overhead_from_summary(num_queries, mapping_summary or {})
        self.energy.e_noc += e_map_noc

        # Seeding PTR/CAL mapping model (dominant mapping-sensitive part in genomics).
        t_seed_mem, e_seed_mem, _ = self._seeding_ptr_cal_overhead(
            num_queries=num_queries,
            ops_per_seed_task=self.cfg_app.ops_per_seed_task,
            mapping_summary=mapping_summary or {},
        )
        self.energy.e_dram_act += e_seed_mem

        timing_model = str((mapping_summary or {}).get("timing_model", "conservative")).lower()
        overlap_coeff = float((mapping_summary or {}).get("seeding_memory_overlap_coeff", 0.0))
        overlap_coeff = max(0.0, min(1.0, overlap_coeff))
        effective_seed_mem_penalty = (1.0 - overlap_coeff) * (t_map_extra + t_seed_mem)
        if timing_model == "paper_like":
            # Paper-like pipeline: memory-bound seeding stage overlaps with compute-bound
            # alignment stage; end-to-end latency is stage bottleneck rather than sum.
            seeding_total = t_seeding_stage + effective_seed_mem_penalty
            alignment_total = t_alignment_stage
            total_time = max(seeding_total, alignment_total)
        else:
            # Conservative model: stage times and mapping penalties are additive.
            total_time += effective_seed_mem_penalty
        return total_time, self.energy

    def run_apsp_algorithm(self, num_vertices: int):
        if self.app != 'apsp': raise Exception("Not configured for APSP")
        self.energy.reset()
        
        B = 256
        op_size_bits = self.cfg_app.OP_SIZE_BITS
        total_ops = num_vertices ** 3
        
        # 時間計算 (保持不變)
        total_cycles = math.ceil(total_ops / self.cfg_pim.TOTAL_COMPUTE_PES)
        total_time = total_cycles * self.cfg_pim.CLK_PERIOD
        
        # --- 能量模型 V2: 階層式修正 ---
        # 1. 計算功耗 (不變)
        self.energy.e_compute = total_cycles * self.hw.TOTAL_ALIGNMENT_ENERGY_PER_CYCLE
        
        # 2. SRAM 功耗 (修正)
        # 2.1. 絕大多數操作 (N^3) 發生在 32KB Local Memory
        e_local_mem_per_access = 0.0074776e-9 # 32KB Local Memory 存取能量
        self.energy.e_sram += total_ops * 3 * e_local_mem_per_access
        
        # 2.2. 少量的區塊級資料交換會用到 256KB Shared Memory
        # Blocked FW 的 Shared Memory 存取次數約為 O(N^3 / B)
        num_shared_mem_access = (num_vertices ** 3 / B)
        e_shared_mem_per_access = 0.32063e-9 # 256KB Shared Memory 存取能量
        self.energy.e_sram += num_shared_mem_access * 3 * e_shared_mem_per_access

        # 3. DRAM 功耗 (不變)
        data_moved_bytes = (num_vertices**3 / B) * (op_size_bits / 8)
        self.energy.e_dram_act = data_moved_bytes * 8 * self.cfg_dram.energy_per_bit_int
        
        # 4. NoC 功耗 (不變)
        num_tiles_per_dim = num_vertices // B
        broadcast_data_bits = B * B * op_size_bits
        _, e_broadcast_per_iter = self._calculate_vertical_transfer_cost(broadcast_data_bits, 0, 512)
        self.energy.e_noc = e_broadcast_per_iter * num_tiles_per_dim * 2

        return total_time, self.energy
