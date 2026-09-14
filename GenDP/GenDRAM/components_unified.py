# from dataclasses import dataclass
# from config_unified import CfgPIM

# # --- PE 的核心：無乘法器的 ALU ---
# @dataclass
# class ALU_Unit:
#     # 估算一個 16-bit ALU (Add/Sub/Cmp/XOR) 的面積與能量
#     AREA = 400 * 1e-6 # 400 um^2
#     ENERGY_PER_OP = 0.2 * 1e-12 # 0.2 pJ

# # --- 播種區的 Search PE (簡化模型) ---
# @dataclass
# class Search_PE:
#     # Search PE 邏輯簡單，此處為估算值
#     AREA = 100 * 1e-6 # 100 um^2
#     ENERGY_PER_CYCLE = 0.1 * 1e-12 # 0.1 pJ

# # --- 比對區的 Compute PE ---
# @dataclass
# class Compute_PE:
#     # 主要由一個 ALU 和一些本地緩存組成
#     AREA = ALU_Unit.AREA + (50 * 1e-6) # 加上控制和緩存的開銷
#     ENERGY_PER_CYCLE = ALU_Unit.ENERGY_PER_OP

# # --- DRAM 層上的硬體資源 ---
# @dataclass
# class OnLayer_Hardware:
#     # 播種層
#     SEEDING_AREA = CfgPIM.SEARCH_PES_PER_LAYER * Search_PE.AREA
#     SEEDING_ENERGY_CYCLE = CfgPIM.SEARCH_PES_PER_LAYER * Search_PE.ENERGY_PER_CYCLE
    
#     # 比對層
#     ALIGNMENT_AREA = CfgPIM.ALIGNMENT_PES_PER_LAYER * Compute_PE.AREA
#     ALIGNMENT_ENERGY_CYCLE = CfgPIM.ALIGNMENT_PES_PER_LAYER * Compute_PE.ENERGY_PER_CYCLE

# 檔名: components_unified.py

from dataclasses import dataclass
from config_unified import CfgPIM

# --- PE 的核心：無乘法器的 ALU ---
@dataclass
class ALU_Unit:
    # 估算一個 16-bit ALU (Add/Sub/Cmp/XOR) 的面積與能量
    AREA = 400 * 1e-6 # 400 um^2
    ENERGY_PER_OP = 0.2 * 1e-12 # 0.2 pJ

# --- 搜尋區的 Search PE (簡化模型) ---
@dataclass
class Search_PE:
    # Search PE 邏輯簡單，此處為估算值
    AREA = 100 * 1e-6 # 100 um^2
    ENERGY_PER_CYCLE = 0.1 * 1e-12 # 0.1 pJ

# --- 計算區的 Compute PE ---
@dataclass
class Compute_PE:
    # 主要由一個 ALU 和一些本地緩存組成
    AREA = ALU_Unit.AREA + (50 * 1e-6) # 加上控制和緩存的開銷
    ENERGY_PER_CYCLE = ALU_Unit.ENERGY_PER_OP

# --- 邏輯晶片上的硬體資源總計 ---
# (原名 OnLayer_Hardware，已更名並修正)
@dataclass
class LogicDie_Hardware:
    # 搜尋區的總面積與週期能量
    TOTAL_SEEDING_AREA = CfgPIM.TOTAL_SEARCH_PES * Search_PE.AREA
    TOTAL_SEEDING_ENERGY_PER_CYCLE = CfgPIM.TOTAL_SEARCH_PES * Search_PE.ENERGY_PER_CYCLE
    
    # 計算區的總面積與週期能量
    TOTAL_ALIGNMENT_AREA = CfgPIM.TOTAL_COMPUTE_PES * Compute_PE.AREA
    TOTAL_ALIGNMENT_ENERGY_PER_CYCLE = CfgPIM.TOTAL_COMPUTE_PES * Compute_PE.ENERGY_PER_CYCLE