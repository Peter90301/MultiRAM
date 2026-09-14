import os
import time
import numpy as np
import heapq
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

class SpecHD_FeRAM_System:
    def __init__(self, num_tiles=32, d_dim=2048, tile_cols=512, q_levels=16):
        """
        初始化 Proteomics FeRAM 加速器系統
        :param num_tiles: FeRAM Tile (Macro) 的數量 (Spec: 32)
        :param d_dim: Hypervector 維度，對應 FeRAM Wordlines (Spec: 2048)
        :param tile_cols: 每個 Tile 的 Bitlines，對應可存的蛋白質數 (Spec: 512)
        :param q_levels: 強度量化等級 (Spec: 16)
        """
        # --- 硬體規格 ---
        self.num_tiles = num_tiles
        self.rows = d_dim      # Wordlines
        self.cols = tile_cols  # Bitlines
        
        # --- FeRAM 記憶體陣列 (The Analog Core) ---
        # 模擬 2T-2C 儲存: 使用 Bipolar (-1, +1) 代表物理狀態
        # 初始化為 0 (代表未寫入)
        self.feram_tiles = [np.zeros((self.rows, self.cols), dtype=np.int8) for _ in range(num_tiles)]
        
        # Metadata 表 (Logic Die 上的對照表，硬體中不需要，模擬用)
        self.metadata = [[None] * self.cols for _ in range(num_tiles)]

        # --- SpecHD 編碼器參數 (Logic Die: Item Memory) ---
        # 隨機生成基礎向量 (ID HVs 和 Level HVs)
        # 這裡模擬 Logic Die 上的 SRAM 查表
        print(f"[Init] Initializing SpecHD Item Memories (D={d_dim})...")
        self.id_hvs = np.random.choice([-1, 1], size=(2000, d_dim)).astype(np.int8) # 假設 m/z 範圍 0-2000
        self.level_hvs = np.random.choice([-1, 1], size=(q_levels, d_dim)).astype(np.int8)
        self.q_levels = q_levels

    def encode_spectrum(self, mz_list, intensity_list):
        """
        [Logic Die] SpecHD 編碼器: 將原始光譜轉為 Hypervector
        這裡模擬 Digital Logic 的運作
        """
        # 1. 預處理: 正規化與過濾 (Top-50 peaks logic would go here)
        if len(intensity_list) == 0:
            return np.zeros(self.rows)
            
        max_inten = np.max(intensity_list)
        
        # 2. 累加器 (Accumulator) 初始化
        sum_hv = np.zeros(self.rows, dtype=np.int16)

        # 3. 逐點編碼 (Binding & Bundling)
        for mz, inten in zip(mz_list, intensity_list):
            # 量化 m/z -> ID Index
            mz_idx = int(mz) 
            if mz_idx >= len(self.id_hvs): continue # 超出範圍忽略
            
            # 量化 Intensity -> Level Index
            lvl_idx = int((inten / max_inten) * (self.q_levels - 1))
            
            # 查表 (SRAM Lookup)
            id_vec = self.id_hvs[mz_idx]
            lvl_vec = self.level_hvs[lvl_idx]
            
            # Binding (XOR in Binary = Multiplication in Bipolar)
            # +1 * +1 = +1 (Same)
            # +1 * -1 = -1 (Diff) -> 這就是 XNOR 邏輯
            bound_vec = id_vec * lvl_vec
            
            # Bundling (Accumulate)
            sum_hv += bound_vec

        # 4. 二值化 (Majority Function) -> 產生 Query Voltage
        # 大於 0 設為 +1 (V_high), 小於 0 設為 -1 (V_low)
        query_hv = np.where(sum_hv >= 0, 1, -1).astype(np.int8)
        return query_hv

    def program_database(self, protein_database):
        """
        [Write Operation] 將參考蛋白質寫入 FeRAM
        實作: Transposed Mapping + Sharding
        """
        print(f"\n[FeRAM] Programming Database ({len(protein_database)} proteins)...")
        
        for idx, (prot_id, spectrum) in enumerate(tqdm(protein_database)):
            # 1. 先在 Logic Die 進行編碼
            encoded_hv = self.encode_spectrum(spectrum['mz'], spectrum['intensity'])
            
            # 2. Mapping: 決定去哪個 Tile, 哪條 Bitline
            tile_id = idx % self.num_tiles
            col_id = idx // self.num_tiles
            
            if col_id >= self.cols:
                print("Warning: FeRAM Capacity Full!")
                break
                
            # 3. 寫入 FeRAM (Transposed: 向量直立寫入)
            # self.feram_tiles[tile_id] 的形狀是 (Rows, Cols)
            # 我們將 encoded_hv (大小 2048) 寫入第 col_id 行
            self.feram_tiles[tile_id][:, col_id] = encoded_hv
            self.metadata[tile_id][col_id] = prot_id

    def analog_search(self, patient_spectrum, top_k=50):
        """
        [Read/Compute Operation] 模擬並行類比搜尋
        1. Query -> DAC -> Wordlines
        2. FeRAM Cell -> Current -> Bitlines
        3. ADC -> Logic Die -> Top-K
        """
        # 1. Query Encoding (Logic Die)
        query_vec = self.encode_spectrum(patient_spectrum['mz'], patient_spectrum['intensity'])
        
        all_candidates = []

        # 2. Parallel Search across all Tiles (Simultaneous Hardware Operation)
        for t_id in range(self.num_tiles):
            # --- 核心物理模擬: 類比電流總和 ---
            # Dot Product: I_out = V_in * G_cell
            # query_vec (1, 2048) dot Tile (2048, 512) -> Result (1, 512)
            # 結果代表 Hamming Similarity (分數越高越像)
            analog_currents = np.dot(query_vec, self.feram_tiles[t_id])
            
            # 3. ADC & Local Top-K (Logic Die)
            # 這裡模擬 ADC 讀取後，每個 Tile 內部的比較器選出 Local Top-K
            # 我們直接用 heapq 模擬這個數位電路行為
            local_top_indices = heapq.nlargest(top_k, range(len(analog_currents)), analog_currents.take)
            
            for col_idx in local_top_indices:
                score = analog_currents[col_idx]
                pid = self.metadata[t_id][col_idx]
                if pid is not None:
                    all_candidates.append((score, pid))

        # 4. Global Top-K Aggregation (Logic Die Center)
        final_results = heapq.nlargest(top_k, all_candidates, key=lambda x: x[0])
        return final_results

def _select_top_peaks(mz_list, intensity_list, max_peaks):
    if len(intensity_list) <= max_peaks:
        mz = np.array(mz_list, dtype=np.float32)
        inten = np.array(intensity_list, dtype=np.float32)
        order = np.argsort(mz)
        return mz[order], inten[order]
    mz = np.array(mz_list, dtype=np.float32)
    inten = np.array(intensity_list, dtype=np.float32)
    top_idx = np.argsort(inten)[-max_peaks:]
    mz = mz[top_idx]
    inten = inten[top_idx]
    order = np.argsort(mz)
    return mz[order], inten[order]


def parse_mgf_file(file_path, max_peaks=50, mz_max=2000.0):
    spectra = []
    base = os.path.basename(file_path)
    in_block = False
    current_title = None
    precursor_mz = None
    mz_list = []
    intensity_list = []
    spec_index = 0

    with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            upper = line.upper()
            if upper == "BEGIN IONS":
                in_block = True
                current_title = None
                precursor_mz = None
                mz_list = []
                intensity_list = []
                continue
            if upper == "END IONS":
                in_block = False
                if mz_list:
                    spec_index += 1
                    spec_id = current_title or f"{base}#S{spec_index:04d}"
                    mz, inten = _select_top_peaks(mz_list, intensity_list, max_peaks)
                    spectra.append(
                        (spec_id, {"mz": mz, "intensity": inten, "precursor_mz": precursor_mz})
                    )
                continue
            if not in_block:
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip().upper()
                if key == "TITLE":
                    current_title = value.strip()
                elif key == "PEPMASS":
                    try:
                        precursor_mz = float(value.strip().split()[0])
                    except ValueError:
                        precursor_mz = None
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    mz = float(parts[0])
                    inten = float(parts[1])
                except ValueError:
                    continue
                if 0 < mz <= mz_max and inten > 0:
                    mz_list.append(mz)
                    intensity_list.append(inten)
    return spectra


def load_mgf_folder(folder_path, max_peaks=50, mz_max=2000.0):
    mgf_files = sorted(
        f for f in os.listdir(folder_path) if f.lower().endswith(".mgf")
    )
    spectra = []
    for fname in mgf_files:
        file_path = os.path.join(folder_path, fname)
        spectra.extend(parse_mgf_file(file_path, max_peaks=max_peaks, mz_max=mz_max))
    return spectra


def spectrum_to_tokens(mz_list, intensity_list, max_peaks=50, mz_max=2000.0):
    tokens = np.zeros((max_peaks, 2), dtype=np.float32)
    if len(intensity_list) == 0:
        return tokens
    mz, inten = _select_top_peaks(mz_list, intensity_list, max_peaks)
    max_inten = np.max(inten) if np.max(inten) > 0 else 1.0
    mz_norm = mz / mz_max
    inten_norm = inten / max_inten
    n = min(len(mz_norm), max_peaks)
    tokens[:n, 0] = mz_norm[:n]
    tokens[:n, 1] = inten_norm[:n]
    return tokens


def build_transformer_dataset(accelerator, spectra, max_peaks=50, mz_max=2000.0):
    ids = []
    tokens = []
    hvs = []
    for spec_id, spectrum in spectra:
        mz = spectrum["mz"]
        inten = spectrum["intensity"]
        ids.append(spec_id)
        tokens.append(spectrum_to_tokens(mz, inten, max_peaks=max_peaks, mz_max=mz_max))
        hvs.append(accelerator.encode_spectrum(mz, inten))
    return (
        np.stack(tokens).astype(np.float32),
        np.stack(hvs).astype(np.int8),
        np.array(ids),
    )


def _get_precursor_mz(spectrum):
    precursor = spectrum.get("precursor_mz")
    if precursor is not None:
        return precursor
    if len(spectrum["mz"]) == 0:
        return 0.0
    return float(np.median(spectrum["mz"]))


def bucket_spectra_by_precursor(spectra, bucket_width=10.0):
    buckets = {}
    for idx, (_, spectrum) in enumerate(spectra):
        precursor = _get_precursor_mz(spectrum)
        bucket_id = int(precursor // bucket_width) if bucket_width > 0 else 0
        buckets.setdefault(bucket_id, []).append(idx)
    return buckets


def _hamming_distance_matrix(hvs):
    dim = hvs.shape[1]
    dot = hvs.astype(np.int32) @ hvs.astype(np.int32).T
    return (dim - dot) * 0.5


def nn_chain_hac(hvs, threshold_ratio=0.2, linkage="complete"):
    if linkage != "complete":
        raise ValueError("Only complete linkage is supported in this implementation.")

    dim = hvs.shape[1]
    threshold = threshold_ratio * dim
    clusters = [[idx] for idx in range(hvs.shape[0])]
    dist = _hamming_distance_matrix(hvs).astype(np.float32)

    chain = []
    while len(clusters) > 1:
        if not chain:
            chain.append(0)

        while True:
            last = chain[-1]
            row = dist[last].copy()
            row[last] = np.inf
            nn = int(np.argmin(row))

            if len(chain) >= 2 and nn == chain[-2]:
                if dist[last, nn] > threshold:
                    return clusters

                i, j = (last, nn) if last < nn else (nn, last)
                clusters[i].extend(clusters[j])

                dist[i, :] = np.maximum(dist[i, :], dist[j, :])
                dist[:, i] = dist[i, :]
                dist = np.delete(dist, j, axis=0)
                dist = np.delete(dist, j, axis=1)
                del clusters[j]

                chain = []
                break
            else:
                chain.append(nn)

    return clusters


if __name__ == "__main__":
    data_dir = "/mnt/hdd/tsunghan/raw-ms-dataset/cleaned_w_charge2"
    output_path = "/home/tsl012/multiomic/FeRAM_simulation/transformer_inputs.npz"
    max_peaks = 50
    mz_max = 2000.0
    energy_per_bit_fj = 100.0
    run_search_benchmark = True
    num_query_benchmark = None
    run_clustering = True
    cluster_threshold_ratio = 0.45  # Optimized via parameter sweep: best B³-F1=0.5589 (vs 0.20→0.5014, 0.35→0.5236)
    cluster_bucket_width = 10.0
    cluster_max_items_per_bucket = None

    total_start = time.perf_counter()
    read_start = time.perf_counter()
    spectra = load_mgf_folder(data_dir, max_peaks=max_peaks, mz_max=mz_max)
    if not spectra:
        raise SystemExit("No spectra found in cleaned_w_charge2.")
    read_end = time.perf_counter()

    program_start = time.perf_counter()
    accelerator = SpecHD_FeRAM_System(num_tiles=32, d_dim=2048, tile_cols=512)
    capacity = accelerator.num_tiles * accelerator.cols
    if len(spectra) > capacity:
        spectra = spectra[:capacity]

    accelerator.program_database(spectra)
    program_end = time.perf_counter()

    search_start = time.perf_counter()
    if run_search_benchmark:
        query_count = len(spectra) if num_query_benchmark is None else num_query_benchmark
        for idx in range(min(query_count, len(spectra))):
            _, spectrum = spectra[idx]
            accelerator.analog_search(spectrum, top_k=5)
    search_end = time.perf_counter()

    dataset_start = time.perf_counter()
    tokens, hvs, ids = build_transformer_dataset(
        accelerator, spectra, max_peaks=max_peaks, mz_max=mz_max
    )
    dataset_end = time.perf_counter()

    clustering_start = time.perf_counter()
    clusters = []
    bucket_sizes = []
    if run_clustering:
        buckets = bucket_spectra_by_precursor(spectra, bucket_width=cluster_bucket_width)
        for _, indices in buckets.items():
            bucket_sizes.append(len(indices))
            if cluster_max_items_per_bucket is not None:
                indices = indices[:cluster_max_items_per_bucket]
            if len(indices) == 0:
                continue
            hvs_bucket = hvs[indices]
            local_clusters = nn_chain_hac(
                hvs_bucket, threshold_ratio=cluster_threshold_ratio, linkage="complete"
            )
            for cluster in local_clusters:
                clusters.append([indices[i] for i in cluster])
    clustering_end = time.perf_counter()

    save_start = time.perf_counter()
    np.savez_compressed(output_path, tokens=tokens, hvs=hvs, ids=ids)
    save_end = time.perf_counter()
    total_end = time.perf_counter()

    bits_per_query = accelerator.rows * accelerator.num_tiles * accelerator.cols
    energy_per_query_j = energy_per_bit_fj * bits_per_query * 1e-15
    energy_per_query_mj = energy_per_query_j * 1e3
    query_count = len(spectra) if num_query_benchmark is None else num_query_benchmark
    energy_total_mj = energy_per_query_mj * min(query_count, len(spectra))

    print(f"[Output] Saved transformer inputs: {output_path}")
    print("\n[FeRAM Metrics]")
    print(f"Read MGF (s): {read_end - read_start:.4f}")
    print(f"Program DB (s): {program_end - program_start:.4f}")
    print(f"Search Benchmark (s): {search_end - search_start:.4f}")
    print(f"Build Dataset (s): {dataset_end - dataset_start:.4f}")
    if run_clustering:
        sizes = sorted((len(c) for c in clusters), reverse=True)
        if bucket_sizes:
            print(f"Buckets: {len(bucket_sizes)}")
            print(f"Max bucket size: {max(bucket_sizes)}")
        print(f"Clustering (s): {clustering_end - clustering_start:.4f}")
        print(f"Clusters: {len(clusters)}")
        print(f"Top-10 cluster sizes: {sizes[:10]}")
    print(f"Save NPZ (s): {save_end - save_start:.4f}")
    print(f"Total (s): {total_end - total_start:.4f}")
    print(f"Energy per query (mJ): {energy_per_query_mj:.6f}")
    print(f"Energy total for benchmark (mJ): {energy_total_mj:.6f}")