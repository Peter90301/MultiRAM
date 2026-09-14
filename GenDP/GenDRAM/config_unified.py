# 檔名: config_unified.py (最終修正版)

class CfgDRAM3D:
    # --- 32 GB monolithic 3D DRAM organization ---
    NUM_LAYERS = 1024
    NUM_TIERS = 8
    PHYSICAL_LAYERS_PER_TIER = NUM_LAYERS // NUM_TIERS

    TOTAL_CAPACITY_GB = 32
    CAPACITY_PER_TIER_GB = 4
    CAPACITY_PER_LAYER_MB = TOTAL_CAPACITY_GB * 1024 / NUM_LAYERS

    NUM_CHANNELS = 32
    PUS_PER_CHANNEL = 1
    TOTAL_PUS = NUM_CHANNELS * PUS_PER_CHANNEL
    BANK_GROUPS_PER_CHANNEL = 2
    BANKS_PER_BANK_GROUP = 4
    BANKS_PER_CHANNEL = BANK_GROUPS_PER_CHANNEL * BANKS_PER_BANK_GROUP
    TOTAL_BANKS = NUM_CHANNELS * BANKS_PER_CHANNEL
    BANK_CAPACITY_GBIT = 1
    TOTAL_CAPACITY_FROM_BANKS_GB = TOTAL_BANKS * BANK_CAPACITY_GBIT / 8

    # --- 時序參數 (來源: 3D_DRAM_MOE.pdf) ---
    BASE_TRCD = 2.29 * 1e-9
    TRCD_INCREMENT_PER_LAYER = 0.02 * 1e-9
    BASE_TRP = 4.77 * 1e-9
    TRCD_TIERS_NS = (2.29, 3.92, 5.99, 8.50, 11.44, 14.82, 18.63, 22.88)
    tRCD_tiers = [value * 1e-9 for value in TRCD_TIERS_NS]
    
    # --- 跨層通訊模型 ---
    INTER_LAYER_LATENCY_PER_HOP = 0.1 * 1e-9
    VERTICAL_BUS_WIDTH_BITS = 1024
    
    # --- 能量模型 ---
    INTER_LAYER_ENERGY_PER_BIT_PER_HOP = 0.05 * 1e-12
    energy_per_bit_int = 0.429 * 1e-12

    @classmethod
    def validate(cls):
        if cls.NUM_LAYERS % cls.NUM_TIERS != 0:
            raise ValueError("Physical M3D layers must divide evenly across tiers")
        if cls.NUM_TIERS * cls.CAPACITY_PER_TIER_GB != cls.TOTAL_CAPACITY_GB:
            raise ValueError("Tier capacities do not equal total M3D DRAM capacity")
        if cls.TOTAL_CAPACITY_FROM_BANKS_GB != cls.TOTAL_CAPACITY_GB:
            raise ValueError("Channel/bank organization does not equal total M3D DRAM capacity")
        if len(cls.tRCD_tiers) != cls.NUM_TIERS:
            raise ValueError("One tRCD value is required for each M3D DRAM tier")

class CfgPIM:
    CLK_PERIOD = 1. * 10**-9 # 1 GHz
    VERTICAL_BUS_BPS = CfgDRAM3D.VERTICAL_BUS_WIDTH_BITS / CLK_PERIOD
    
    # --- Functional partition: 8 channel-level search PUs and 24 compute PUs ---
    NUM_SEEDING_TIERS = 2
    NUM_ALIGNMENT_TIERS = CfgDRAM3D.NUM_TIERS - NUM_SEEDING_TIERS
    NUM_SEEDING_LAYERS = NUM_SEEDING_TIERS * CfgDRAM3D.PHYSICAL_LAYERS_PER_TIER
    NUM_ALIGNMENT_LAYERS = CfgDRAM3D.NUM_LAYERS - NUM_SEEDING_LAYERS
    
    # --- 硬體設定 (反映邏輯晶片上的真實架構) ---
    # 計算區
    NUM_COMPUTE_PUS = 24
    PES_PER_COMPUTE_PU = 16
    TOTAL_COMPUTE_PES = NUM_COMPUTE_PUS * PES_PER_COMPUTE_PU # 總共 384 個 PE

    # 搜尋區
    NUM_SEARCH_PUS = 8
    PES_PER_SEARCH_PU = 16
    TOTAL_SEARCH_PES = NUM_SEARCH_PUS * PES_PER_SEARCH_PU # 總共 128 個 PE

    @classmethod
    def validate(cls):
        if cls.NUM_COMPUTE_PUS + cls.NUM_SEARCH_PUS != CfgDRAM3D.TOTAL_PUS:
            raise ValueError("Compute and search PUs must equal one PU per M3D channel")


CfgDRAM3D.validate()
CfgPIM.validate()

# 應用設定
class CfgBioPIM:
    OP_SIZE_BITS = 5
    # 修正：為每個比對/播種任務添加操作數估算值
    ops_per_align_task = 12843 # 這是 main_analysis.py 中估算的值
    ops_per_seed_task = 500     # 這是新增的參數，代表每個 Seeding 任務所需的操作數估算值

class CfgAPSP:
    OP_SIZE_BITS = 16
