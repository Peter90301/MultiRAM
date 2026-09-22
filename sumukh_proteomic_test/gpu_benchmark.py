import argparse
import heapq
import math
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

try:
    import cupy as cp
    import cuml
except Exception:
    cp = None
    cuml = None


class SpecHDFeRAMSystem:
    """FeRAM simulation core aligned with SpecHD encoding."""

    def __init__(self, num_tiles=32, d_dim=2048, tile_cols=512, q_levels=16, seed=42):
        self.num_tiles = num_tiles
        self.rows = d_dim
        self.cols = tile_cols
        self.q_levels = q_levels

        rng = np.random.default_rng(seed)
        self.id_hvs = rng.choice([-1, 1], size=(2000, d_dim)).astype(np.int8)
        self.level_hvs = rng.choice([-1, 1], size=(q_levels, d_dim)).astype(np.int8)

        self.feram_tiles = [np.zeros((self.rows, self.cols), dtype=np.int8) for _ in range(num_tiles)]
        self.metadata = [[None] * self.cols for _ in range(num_tiles)]

    def encode_spectrum(self, mz_list, intensity_list):
        if len(intensity_list) == 0:
            return np.zeros(self.rows, dtype=np.int8)

        max_inten = np.max(intensity_list)
        sum_hv = np.zeros(self.rows, dtype=np.int16)

        for mz, inten in zip(mz_list, intensity_list):
            mz_idx = int(mz)
            if mz_idx < 0 or mz_idx >= len(self.id_hvs):
                continue

            lvl_idx = int((inten / max_inten) * (self.q_levels - 1)) if max_inten > 0 else 0
            lvl_idx = min(max(lvl_idx, 0), self.q_levels - 1)

            bound_vec = self.id_hvs[mz_idx] * self.level_hvs[lvl_idx]
            sum_hv += bound_vec

        return np.where(sum_hv >= 0, 1, -1).astype(np.int8)

    def program_database(self, protein_database):
        for idx, (prot_id, spectrum) in enumerate(protein_database):
            encoded_hv = self.encode_spectrum(spectrum["mz"], spectrum["intensity"])
            tile_id = idx % self.num_tiles
            col_id = idx // self.num_tiles

            if col_id >= self.cols:
                break

            self.feram_tiles[tile_id][:, col_id] = encoded_hv
            self.metadata[tile_id][col_id] = prot_id

    def analog_search(self, patient_spectrum, top_k=50):
        query_vec = self.encode_spectrum(patient_spectrum["mz"], patient_spectrum["intensity"])
        all_candidates = []

        for t_id in range(self.num_tiles):
            analog_currents = np.dot(query_vec, self.feram_tiles[t_id])
            local_top_indices = heapq.nlargest(top_k, range(len(analog_currents)), analog_currents.take)

            for col_idx in local_top_indices:
                pid = self.metadata[t_id][col_idx]
                if pid is not None:
                    all_candidates.append((analog_currents[col_idx], pid))

        return heapq.nlargest(top_k, all_candidates, key=lambda x: x[0])


def _select_top_peaks(mz_list, intensity_list, max_peaks):
    mz = np.array(mz_list, dtype=np.float32)
    inten = np.array(intensity_list, dtype=np.float32)

    if len(inten) > max_peaks:
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
                    spectra.append((spec_id, {"mz": mz, "intensity": inten, "precursor_mz": precursor_mz}))
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


def load_mgf_until_count(folder_path, target_count, max_peaks=50, mz_max=2000.0):
    mgf_files = sorted(f for f in os.listdir(folder_path) if f.lower().endswith(".mgf"))
    spectra = []

    for fname in mgf_files:
        file_path = os.path.join(folder_path, fname)
        file_spectra = parse_mgf_file(file_path, max_peaks=max_peaks, mz_max=mz_max)
        spectra.extend(file_spectra)
        if len(spectra) >= target_count:
            break

    return spectra


def write_mgf_file(spectra, output_path):
    with open(output_path, "w", encoding="utf-8") as handle:
        for spec_id, spectrum in spectra:
            handle.write("BEGIN IONS\n")
            handle.write(f"TITLE={spec_id}\n")
            precursor_mz = spectrum.get("precursor_mz")
            if precursor_mz is not None:
                handle.write(f"PEPMASS={precursor_mz}\n")

            mz_vals = spectrum["mz"]
            inten_vals = spectrum["intensity"]
            for mz, inten in zip(mz_vals, inten_vals):
                handle.write(f"{float(mz):.6f} {float(inten):.6f}\n")

            handle.write("END IONS\n\n")


def prepare_subset_files(
    dataset_dir,
    subset_dir,
    subset_sizes,
    max_peaks=50,
    mz_max=2000.0,
    seed=42,
    force=False,
):
    os.makedirs(subset_dir, exist_ok=True)
    expected_paths = {n: os.path.join(subset_dir, f"PXD000561_{n}.mgf") for n in subset_sizes}

    if not force and all(os.path.exists(path) for path in expected_paths.values()):
        return expected_paths

    max_subset = max(subset_sizes)
    raw_spectra = load_mgf_until_count(dataset_dir, max_subset, max_peaks=max_peaks, mz_max=mz_max)
    if len(raw_spectra) < max_subset:
        raise RuntimeError(f"可用光譜數不足。需要 {max_subset}，但只讀到 {len(raw_spectra)}。")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(raw_spectra))
    rng.shuffle(indices)
    shuffled = [raw_spectra[i] for i in indices]

    for n in subset_sizes:
        write_mgf_file(shuffled[:n], expected_paths[n])

    return expected_paths


def benchmark_gpu(query_hvs, database_hvs=None, top_k=50, device="cuda"):
    num_queries, feature_dim = query_hvs.shape
    if database_hvs is None:
        database_hvs = query_hvs
    database_size = database_hvs.shape[0]

    db_tensor = torch.tensor(database_hvs.T, dtype=torch.float16, device=device)
    query_tensor = torch.tensor(query_hvs, dtype=torch.float16, device=device)

    warmup_n = min(10, num_queries)
    _ = torch.matmul(query_tensor[:warmup_n], db_tensor)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    scores = torch.matmul(query_tensor, db_tensor)
    _ = torch.topk(scores, min(top_k, database_size), dim=1)
    end_event.record()
    torch.cuda.synchronize()

    elapsed_s = start_event.elapsed_time(end_event) / 1000.0
    return {
        "num_queries": num_queries,
        "database_size": database_size,
        "feature_dim": feature_dim,
        "gpu_search_seconds": elapsed_s,
        "gpu_latency_us_per_query": (elapsed_s / num_queries) * 1e6,
    }


def build_fixed_capacity_database(encoded_hvs, capacity):
    if encoded_hvs.shape[0] > capacity:
        raise ValueError(f"encoded_hvs count {encoded_hvs.shape[0]} exceeds capacity {capacity}")

    if encoded_hvs.shape[0] == capacity:
        return encoded_hvs

    padded = np.zeros((capacity, encoded_hvs.shape[1]), dtype=np.int8)
    padded[: encoded_hvs.shape[0]] = encoded_hvs
    return padded


def benchmark_feram_search(spectra, top_k=50, num_tiles=32, d_dim=2048, tile_cols=512, seed=42):
    accelerator = SpecHDFeRAMSystem(
        num_tiles=num_tiles,
        d_dim=d_dim,
        tile_cols=tile_cols,
        q_levels=16,
        seed=seed,
    )

    program_start = time.perf_counter()
    accelerator.program_database(spectra)
    program_end = time.perf_counter()

    search_start = time.perf_counter()
    for _, spectrum in spectra:
        accelerator.analog_search(spectrum, top_k=top_k)
    search_end = time.perf_counter()

    search_s = search_end - search_start
    return {
        "num_queries": len(spectra),
        "feram_program_seconds": program_end - program_start,
        "feram_search_seconds": search_s,
        "feram_latency_us_per_query": (search_s / len(spectra)) * 1e6,
    }


def estimate_feram_neurosim_latency(
    num_queries,
    d_dim,
    num_tiles,
    tile_cols,
    row_parallelism,
    dac_ns,
    wl_driver_ns,
    cell_read_ns,
    sense_amp_ns,
    adc_ns,
    adder_tree_ns,
    local_topk_ns,
    global_interconnect_ns,
    global_topk_ns,
    dac_energy_fj_per_row=20.0,
    cell_read_energy_fj=2.0,
    adc_energy_fj_per_col=200.0,
    digital_energy_fj_per_col=50.0,
):
    rows_per_cycle = max(1, int(row_parallelism))
    serial_cycles = max(1, int(math.ceil(d_dim / rows_per_cycle)))

    array_cycle_ns = dac_ns + wl_driver_ns + cell_read_ns + sense_amp_ns + adc_ns
    analog_read_ns = serial_cycles * array_cycle_ns

    local_digital_ns = adder_tree_ns + local_topk_ns
    global_digital_ns = global_interconnect_ns + global_topk_ns

    per_query_ns = analog_read_ns + local_digital_ns + global_digital_ns
    total_s = (per_query_ns * num_queries) * 1e-9

    total_cols = num_tiles * tile_cols
    per_query_energy_fj = (
        d_dim * dac_energy_fj_per_row
        + d_dim * total_cols * cell_read_energy_fj
        + total_cols * adc_energy_fj_per_col
        + total_cols * digital_energy_fj_per_col
    )
    total_energy_mj = per_query_energy_fj * num_queries * 1e-12

    return {
        "feram_hw_latency_us_per_query": per_query_ns * 1e-3,
        "feram_neurosim_time_s": total_s,
        "feram_neurosim_energy_mj": total_energy_mj,
        "feram_neurosim_energy_nj_per_query": per_query_energy_fj * 1e-6,
        "feram_hw_serial_cycles": serial_cycles,
        "feram_hw_array_cycle_ns": array_cycle_ns,
        "feram_hw_tile_capacity": num_tiles * tile_cols,
    }


def encode_spectra(accelerator, spectra):
    encoded = []
    for _, spectrum in spectra:
        encoded.append(accelerator.encode_spectrum(spectrum["mz"], spectrum["intensity"]))
    return np.stack(encoded).astype(np.int8)


def _get_precursor_mz(spectrum):
    precursor = spectrum.get("precursor_mz")
    if precursor is not None:
        return float(precursor)
    mz = spectrum.get("mz")
    if mz is None or len(mz) == 0:
        return 0.0
    return float(np.median(mz))


def bucket_spectra_by_precursor(spectra, bucket_width=10.0):
    buckets = {}
    for idx, (_, spectrum) in enumerate(spectra):
        precursor = _get_precursor_mz(spectrum)
        bucket_id = int(precursor // bucket_width) if bucket_width > 0 else 0
        buckets.setdefault(bucket_id, []).append(idx)
    return buckets


def estimate_fenand_coarse_filter(
    spectra,
    buckets,
    metadata_bytes_per_spectrum=16.0,
    output_bytes_per_spectrum=256.0,
    decompressed_stream_gbps=8.1,
    package_link_gbps=256.0,
    setup_overhead_us=50.0,
    filter_energy_pj_per_spectrum=20.0,
    package_link_energy_pj_per_bit=0.8,
):
    """Model metadata bucketing in FeNAND before FeRAM clustering.

    All spectra survive. The filter removes cross-bucket pair comparisons, so
    selectivity is derived from the observed bucket occupancy rather than an
    assumed spectrum drop rate. The emitted payload is one packed 2048-bit HV
    per spectrum by default.
    """
    num_spectra = len(spectra)
    unfiltered_pairs = num_spectra * (num_spectra - 1) // 2
    candidate_pairs = sum(
        len(indices) * (len(indices) - 1) // 2 for indices in buckets.values()
    )
    pair_survival_ratio = (
        candidate_pairs / unfiltered_pairs if unfiltered_pairs else 0.0
    )

    metadata_bytes = num_spectra * max(0.0, float(metadata_bytes_per_spectrum))
    output_bytes = num_spectra * max(0.0, float(output_bytes_per_spectrum))
    storage_gbps = max(float(decompressed_stream_gbps), 1e-12)
    link_gbps = max(float(package_link_gbps), 1e-12)
    effective_output_gbps = min(storage_gbps, link_gbps)

    setup_s = max(0.0, float(setup_overhead_us)) * 1e-6
    metadata_scan_s = metadata_bytes / (storage_gbps * 1e9)
    output_transfer_s = output_bytes / (effective_output_gbps * 1e9)
    total_s = setup_s + metadata_scan_s + output_transfer_s

    internal_energy_j = (
        num_spectra * max(0.0, float(filter_energy_pj_per_spectrum)) * 1e-12
    )
    link_energy_j = (
        output_bytes
        * 8.0
        * max(0.0, float(package_link_energy_pj_per_bit))
        * 1e-12
    )
    total_energy_mj = (internal_energy_j + link_energy_j) * 1e3

    return {
        "fenand_filter_enabled": True,
        "fenand_filter_setup_s": setup_s,
        "fenand_metadata_scan_s": metadata_scan_s,
        "fenand_to_feram_transfer_s": output_transfer_s,
        "fenand_filter_total_s": total_s,
        "fenand_filter_internal_energy_mj": internal_energy_j * 1e3,
        "fenand_to_feram_link_energy_mj": link_energy_j * 1e3,
        "fenand_filter_total_energy_mj": total_energy_mj,
        "fenand_metadata_bytes": int(metadata_bytes),
        "fenand_to_feram_bytes": int(output_bytes),
        "fenand_effective_output_GBps": effective_output_gbps,
        "unfiltered_candidate_pairs": int(unfiltered_pairs),
        "filtered_candidate_pairs": int(candidate_pairs),
        "candidate_pair_survival_ratio": pair_survival_ratio,
        "candidate_pair_reduction_percent": 100.0 * (1.0 - pair_survival_ratio),
        "spectrum_survival_ratio": 1.0 if num_spectra else 0.0,
        "model_provenance": (
            "8.1 GB/s FeNAND stream and 256 GB/s package link are architecture "
            "inputs; setup latency, bytes/record, internal filter energy, and "
            "package-link energy are explicit modeling assumptions."
        ),
    }


def _hamming_distance_matrix_numpy(hvs):
    dim = hvs.shape[1]
    dot = hvs.astype(np.int32) @ hvs.astype(np.int32).T
    return ((dim - dot) * 0.5).astype(np.float32)


def _hamming_distance_matrix_gpu(hvs):
    dim = hvs.shape[1]
    hv_tensor = torch.tensor(hvs, dtype=torch.float16, device="cuda")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    dot = hv_tensor @ hv_tensor.T
    end_event.record()
    torch.cuda.synchronize()
    matmul_s = start_event.elapsed_time(end_event) / 1000.0
    dist = (float(dim) - dot) * 0.5
    return dist.detach().cpu().numpy().astype(np.float32), matmul_s


def nn_chain_hac_from_distance(distance_matrix, threshold_ratio=0.45):
    if distance_matrix.shape[0] <= 1:
        return [[0]] if distance_matrix.shape[0] == 1 else []

    dim = int(np.max(distance_matrix)) * 2 if distance_matrix.size else 0
    threshold = threshold_ratio * dim if dim > 0 else np.inf
    clusters = [[idx] for idx in range(distance_matrix.shape[0])]
    dist = distance_matrix.copy()

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

            chain.append(nn)

    return clusters


def estimate_feram_bucket_clustering_neurosim(
    bucket_size,
    d_dim,
    num_tiles,
    tile_cols,
    row_parallelism,
    dac_ns,
    wl_driver_ns,
    cell_read_ns,
    sense_amp_ns,
    adc_ns,
    adder_tree_ns,
    local_topk_ns,
    global_interconnect_ns,
    global_topk_ns,
    dac_energy_fj_per_row=20.0,
    cell_read_energy_fj=2.0,
    adc_energy_fj_per_col=200.0,
    digital_energy_fj_per_col=50.0,
):
    if bucket_size <= 1:
        return {
            "feram_bucket_time_s": 0.0,
            "feram_bucket_energy_mj": 0.0,
            "feram_bucket_serial_cycles": 0,
            "feram_bucket_hw_latency_us_per_query": 0.0,
            "feram_bucket_hw_energy_nj_per_query": 0.0,
        }

    total_cols = max(1, num_tiles * tile_cols)
    rows_per_cycle = max(1, int(row_parallelism))
    row_cycles = max(1, int(math.ceil(d_dim / rows_per_cycle)))
    col_passes = max(1, int(math.ceil(bucket_size / total_cols)))

    array_cycle_ns = dac_ns + wl_driver_ns + cell_read_ns + sense_amp_ns + adc_ns
    analog_read_ns = row_cycles * col_passes * array_cycle_ns

    local_digital_ns = adder_tree_ns + local_topk_ns
    global_digital_ns = global_interconnect_ns + global_topk_ns
    per_query_ns = analog_read_ns + local_digital_ns + global_digital_ns

    # Distance matrix for one bucket compares B_i queries against B_i references.
    bucket_time_s = (per_query_ns * bucket_size) * 1e-9

    per_query_energy_fj = (
        d_dim * dac_energy_fj_per_row
        + d_dim * bucket_size * cell_read_energy_fj
        + bucket_size * adc_energy_fj_per_col
        + bucket_size * digital_energy_fj_per_col
    )
    bucket_energy_mj = per_query_energy_fj * bucket_size * 1e-12

    return {
        "feram_bucket_time_s": bucket_time_s,
        "feram_bucket_energy_mj": bucket_energy_mj,
        "feram_bucket_serial_cycles": row_cycles * col_passes,
        "feram_bucket_hw_latency_us_per_query": per_query_ns * 1e-3,
        "feram_bucket_hw_energy_nj_per_query": per_query_energy_fj * 1e-6,
    }


def estimate_feram_clustering_from_spectra(
    spectra,
    bucket_width=10.0,
    max_items_per_bucket=2000,
    d_dim=2048,
    num_tiles=32,
    tile_cols=512,
    row_parallelism=2048,
    neurosim_dac_ns=2.0,
    neurosim_wl_driver_ns=1.0,
    neurosim_cell_read_ns=4.0,
    neurosim_sense_amp_ns=1.0,
    neurosim_adc_ns=8.0,
    neurosim_adder_tree_ns=3.0,
    neurosim_local_topk_ns=6.0,
    neurosim_global_interconnect_ns=5.0,
    neurosim_global_topk_ns=7.0,
    neurosim_dac_energy_fj_per_row=20.0,
    neurosim_cell_read_energy_fj=2.0,
    neurosim_adc_energy_fj_per_col=200.0,
    neurosim_digital_energy_fj_per_col=50.0,
    fenand_coarse_filter=True,
    fenand_metadata_bytes_per_spectrum=16.0,
    fenand_output_bytes_per_spectrum=256.0,
    fenand_decompressed_stream_gbps=8.1,
    fenand_package_link_gbps=256.0,
    fenand_setup_overhead_us=50.0,
    fenand_filter_energy_pj_per_spectrum=20.0,
    package_link_energy_pj_per_bit=0.8,
):
    buckets = bucket_spectra_by_precursor(spectra, bucket_width=bucket_width)

    if fenand_coarse_filter:
        fenand = estimate_fenand_coarse_filter(
            spectra=spectra,
            buckets=buckets,
            metadata_bytes_per_spectrum=fenand_metadata_bytes_per_spectrum,
            output_bytes_per_spectrum=fenand_output_bytes_per_spectrum,
            decompressed_stream_gbps=fenand_decompressed_stream_gbps,
            package_link_gbps=fenand_package_link_gbps,
            setup_overhead_us=fenand_setup_overhead_us,
            filter_energy_pj_per_spectrum=fenand_filter_energy_pj_per_spectrum,
            package_link_energy_pj_per_bit=package_link_energy_pj_per_bit,
        )
    else:
        num_spectra = len(spectra)
        unfiltered_pairs = num_spectra * (num_spectra - 1) // 2
        filtered_pairs = sum(
            len(indices) * (len(indices) - 1) // 2
            for indices in buckets.values()
        )
        fenand = {
            "fenand_filter_enabled": False,
            "fenand_filter_total_s": 0.0,
            "fenand_filter_total_energy_mj": 0.0,
            "unfiltered_candidate_pairs": int(unfiltered_pairs),
            "filtered_candidate_pairs": int(filtered_pairs),
            "candidate_pair_survival_ratio": (
                filtered_pairs / unfiltered_pairs if unfiltered_pairs else 0.0
            ),
            "candidate_pair_reduction_percent": (
                100.0 * (1.0 - filtered_pairs / unfiltered_pairs)
                if unfiltered_pairs else 0.0
            ),
            "spectrum_survival_ratio": 1.0 if num_spectra else 0.0,
            "model_provenance": "Host-side bucketing; FeNAND cost disabled.",
        }

    feram_bucket_time_s = 0.0
    feram_bucket_energy_mj = 0.0
    feram_serial_cycles_acc = 0
    feram_used_buckets = 0
    bucket_sizes = [len(indices) for indices in buckets.values()]
    bucket_overflow_count = sum(
        size > max_items_per_bucket
        for size in bucket_sizes
        if max_items_per_bucket is not None and max_items_per_bucket > 0
    )

    unfiltered_feram = estimate_feram_bucket_clustering_neurosim(
        bucket_size=len(spectra),
        d_dim=d_dim,
        num_tiles=num_tiles,
        tile_cols=tile_cols,
        row_parallelism=row_parallelism,
        dac_ns=neurosim_dac_ns,
        wl_driver_ns=neurosim_wl_driver_ns,
        cell_read_ns=neurosim_cell_read_ns,
        sense_amp_ns=neurosim_sense_amp_ns,
        adc_ns=neurosim_adc_ns,
        adder_tree_ns=neurosim_adder_tree_ns,
        local_topk_ns=neurosim_local_topk_ns,
        global_interconnect_ns=neurosim_global_interconnect_ns,
        global_topk_ns=neurosim_global_topk_ns,
        dac_energy_fj_per_row=neurosim_dac_energy_fj_per_row,
        cell_read_energy_fj=neurosim_cell_read_energy_fj,
        adc_energy_fj_per_col=neurosim_adc_energy_fj_per_col,
        digital_energy_fj_per_col=neurosim_digital_energy_fj_per_col,
    )

    for _, indices in buckets.items():
        if len(indices) <= 1:
            continue

        b_i = len(indices)
        feram_bucket = estimate_feram_bucket_clustering_neurosim(
            bucket_size=b_i,
            d_dim=d_dim,
            num_tiles=num_tiles,
            tile_cols=tile_cols,
            row_parallelism=row_parallelism,
            dac_ns=neurosim_dac_ns,
            wl_driver_ns=neurosim_wl_driver_ns,
            cell_read_ns=neurosim_cell_read_ns,
            sense_amp_ns=neurosim_sense_amp_ns,
            adc_ns=neurosim_adc_ns,
            adder_tree_ns=neurosim_adder_tree_ns,
            local_topk_ns=neurosim_local_topk_ns,
            global_interconnect_ns=neurosim_global_interconnect_ns,
            global_topk_ns=neurosim_global_topk_ns,
            dac_energy_fj_per_row=neurosim_dac_energy_fj_per_row,
            cell_read_energy_fj=neurosim_cell_read_energy_fj,
            adc_energy_fj_per_col=neurosim_adc_energy_fj_per_col,
            digital_energy_fj_per_col=neurosim_digital_energy_fj_per_col,
        )
        feram_bucket_time_s += feram_bucket["feram_bucket_time_s"]
        feram_bucket_energy_mj += feram_bucket["feram_bucket_energy_mj"]
        feram_serial_cycles_acc += feram_bucket["feram_bucket_serial_cycles"]
        feram_used_buckets += 1

    multiram_total_s = float(fenand["fenand_filter_total_s"]) + feram_bucket_time_s
    multiram_total_energy_mj = (
        float(fenand["fenand_filter_total_energy_mj"])
        + feram_bucket_energy_mj
    )
    unfiltered_s = float(unfiltered_feram["feram_bucket_time_s"])
    unfiltered_energy_mj = float(unfiltered_feram["feram_bucket_energy_mj"])

    result = {
        "feram_cluster_core_s": feram_bucket_time_s,
        "feram_cluster_energy_mj": feram_bucket_energy_mj,
        "feram_unfiltered_baseline_s": unfiltered_s,
        "feram_unfiltered_baseline_energy_mj": unfiltered_energy_mj,
        "feram_avg_serial_cycles": (feram_serial_cycles_acc / feram_used_buckets) if feram_used_buckets else 0.0,
        "num_buckets": len(buckets),
        "max_bucket_size": max(bucket_sizes, default=0),
        "bucket_overflow_count": bucket_overflow_count,
        "bucket_width_da": float(bucket_width),
        "multiram_cluster_total_s": multiram_total_s,
        "multiram_cluster_total_energy_mj": multiram_total_energy_mj,
        "speedup_vs_unfiltered_feram": (
            unfiltered_s / multiram_total_s if multiram_total_s > 0 else None
        ),
        "latency_change_vs_unfiltered_percent": (
            100.0 * (multiram_total_s / unfiltered_s - 1.0)
            if unfiltered_s > 0 else None
        ),
        "energy_reduction_vs_unfiltered_percent": (
            100.0 * (1.0 - multiram_total_energy_mj / unfiltered_energy_mj)
            if unfiltered_energy_mj > 0 else None
        ),
    }
    result.update(fenand)
    return result


def benchmark_gpu_hyperspec(
    subset_file,
    hyperspec_root,
    hyperspec_main_relpath="src/main.py",
    use_gpu_cluster=False,
    cpu_core_preprocess=8,
    cpu_core_cluster=8,
    max_peaks=50,
    hd_dim=2048,
    hd_q=16,
    eps=0.6,
    cluster_alg="dbscan",
    cluster_charges=(2, 3),
):
    main_py = os.path.join(hyperspec_root, hyperspec_main_relpath)
    if not os.path.exists(main_py):
        raise FileNotFoundError(f"找不到 Hyper-Spec 入口: {main_py}")

    with tempfile.TemporaryDirectory(prefix="hyperspec_gpu_") as tmp_dir:
        input_dir = os.path.join(tmp_dir, "input")
        os.makedirs(input_dir, exist_ok=True)
        shutil.copy2(subset_file, os.path.join(input_dir, os.path.basename(subset_file)))

        output_stem = os.path.join(tmp_dir, "hyper_spec_output")
        cmd = [
            sys.executable,
            main_py,
            input_dir,
            output_stem,
            f"--cpu_core_preprocess={int(cpu_core_preprocess)}",
            f"--cpu_core_cluster={int(cpu_core_cluster)}",
            f"--max_peaks_used={int(max_peaks)}",
            f"--hd_dim={int(hd_dim)}",
            f"--hd_Q={int(hd_q)}",
            f"--eps={float(eps)}",
            f"--cluster_alg={cluster_alg}",
        ]
        if use_gpu_cluster:
            cmd.append("--use_gpu_cluster")
        if cluster_charges:
            cmd.append("--cluster_charges")
            cmd.extend([str(c) for c in cluster_charges])

        start = time.perf_counter()
        proc = subprocess.run(
            cmd,
            cwd=hyperspec_root,
            capture_output=True,
            text=True,
            check=False,
        )
        end = time.perf_counter()

        if proc.returncode != 0:
            err_tail = "\n".join(proc.stderr.strip().splitlines()[-30:])
            raise RuntimeError(
                "Hyper-Spec 執行失敗。\n"
                f"Command: {' '.join(cmd)}\n"
                f"Exit code: {proc.returncode}\n"
                f"stderr (tail):\n{err_tail}"
            )

        return {
            "gpu_search_seconds": end - start,
            "stdout_tail": "\n".join(proc.stdout.strip().splitlines()[-15:]),
            "stderr_tail": "\n".join(proc.stderr.strip().splitlines()[-15:]),
        }


def benchmark_gpu_hyperoms(
    subset_file,
    hyperoms_cmd_template,
    hyperoms_workdir=None,
):
    if not hyperoms_cmd_template:
        raise ValueError(
            "使用 hyperoms backend 時，必須提供 --hyperoms-cmd-template。"
        )

    with tempfile.TemporaryDirectory(prefix="hyperoms_gpu_") as tmp_dir:
        input_dir = os.path.join(tmp_dir, "input")
        os.makedirs(input_dir, exist_ok=True)
        input_file = os.path.join(input_dir, os.path.basename(subset_file))
        shutil.copy2(subset_file, input_file)

        output_stem = os.path.join(tmp_dir, "hyperoms_output")
        formatted_cmd = hyperoms_cmd_template.format(
            input_dir=input_dir,
            input_file=input_file,
            output_stem=output_stem,
            tmp_dir=tmp_dir,
        )
        cmd = shlex.split(formatted_cmd)

        start = time.perf_counter()
        proc = subprocess.run(
            cmd,
            cwd=hyperoms_workdir if hyperoms_workdir else None,
            capture_output=True,
            text=True,
            check=False,
        )
        end = time.perf_counter()

        if proc.returncode != 0:
            err_tail = "\n".join(proc.stderr.strip().splitlines()[-30:])
            raise RuntimeError(
                "HyperOMS 執行失敗。\n"
                f"Command: {' '.join(cmd)}\n"
                f"Exit code: {proc.returncode}\n"
                f"stderr (tail):\n{err_tail}"
            )

        return {
            "gpu_search_seconds": end - start,
            "stdout_tail": "\n".join(proc.stdout.strip().splitlines()[-15:]),
            "stderr_tail": "\n".join(proc.stderr.strip().splitlines()[-15:]),
        }


def benchmark_gpu_rapids_clustering(encoded_hvs, eps=0.45):
    if cuml is None:
        raise RuntimeError("找不到 cuML，請先在目前環境安裝 RAPIDS。")

    num_queries, dim = encoded_hvs.shape
    if num_queries <= 1:
        return {
            "gpu_cluster_seconds": 0.0,
            "gpu_cluster_num_clusters": num_queries,
        }

    hv = encoded_hvs.astype(np.float32, copy=False)

    # Use RAPIDS DBSCAN directly on HDC feature vectors.
    start = time.perf_counter()
    db = cuml.DBSCAN(
        eps=float(eps),
        min_samples=2,
        metric="euclidean",
        calc_core_sample_indices=False,
        output_type="numpy",
    )
    db.fit(hv)
    end = time.perf_counter()

    labels = np.asarray(db.labels_)
    valid = labels[labels >= 0]
    num_clusters = int(np.unique(valid).size) if valid.size else 0

    return {
        "gpu_cluster_seconds": end - start,
        "gpu_cluster_num_clusters": num_clusters,
    }


def estimate_encoding_hw(
    num_queries,
    cycles_per_query=100,
    clock_hz=500e6,
    energy_pj_per_query=50.0,
):
    cycles = max(1, int(cycles_per_query))
    freq = max(1.0, float(clock_hz))
    per_query_s = cycles / freq
    total_s = per_query_s * num_queries

    per_query_nj = max(0.0, float(energy_pj_per_query)) * 1e-3
    total_mj = per_query_nj * num_queries * 1e-6

    return {
        "encode_hw_s": total_s,
        "encode_hw_us_per_query": per_query_s * 1e6,
        "encode_hw_energy_mj": total_mj,
        "encode_hw_energy_nj_per_query": per_query_nj,
    }


def estimate_fenand_to_feram_transfer(
    num_queries,
    bytes_per_query=2048,
    bandwidth_mb_s=1600.0,
    setup_overhead_us=50.0,
):
    q = max(0, int(num_queries))
    b_per_q = max(0.0, float(bytes_per_query))
    bw_bps = max(1.0, float(bandwidth_mb_s)) * 1e6
    setup_s = max(0.0, float(setup_overhead_us)) * 1e-6

    total_bytes = q * b_per_q
    data_s = total_bytes / bw_bps
    total_s = setup_s + data_s

    return {
        "fenand_to_feram_setup_s": setup_s,
        "fenand_to_feram_data_s": data_s,
        "fenand_to_feram_total_s": total_s,
        "fenand_to_feram_us_per_query": (total_s / max(1, q)) * 1e6,
    }


def benchmark_nn_chain_clustering(
    encoded_hvs,
    spectra,
    threshold_ratio=0.45,
    bucket_width=10.0,
    max_items_per_bucket=2000,
    use_gpu_distance=False,
    d_dim=2048,
    num_tiles=32,
    tile_cols=512,
    row_parallelism=2048,
    neurosim_dac_ns=2.0,
    neurosim_wl_driver_ns=1.0,
    neurosim_cell_read_ns=4.0,
    neurosim_sense_amp_ns=1.0,
    neurosim_adc_ns=8.0,
    neurosim_adder_tree_ns=3.0,
    neurosim_local_topk_ns=6.0,
    neurosim_global_interconnect_ns=5.0,
    neurosim_global_topk_ns=7.0,
    neurosim_dac_energy_fj_per_row=20.0,
    neurosim_cell_read_energy_fj=2.0,
    neurosim_adc_energy_fj_per_col=200.0,
    neurosim_digital_energy_fj_per_col=50.0,
):
    start = time.perf_counter()
    buckets = bucket_spectra_by_precursor(spectra, bucket_width=bucket_width)
    total_clusters = 0
    bucket_count = 0
    gpu_matmul_s = 0.0
    cpu_hac_s = 0.0
    feram_bucket_time_s = 0.0
    feram_bucket_energy_mj = 0.0
    feram_serial_cycles_acc = 0
    feram_used_buckets = 0

    for _, indices in buckets.items():
        if max_items_per_bucket is not None and len(indices) > max_items_per_bucket:
            indices = indices[:max_items_per_bucket]
        if len(indices) <= 1:
            continue

        bucket_count += 1
        b_i = len(indices)

        feram_bucket = estimate_feram_bucket_clustering_neurosim(
            bucket_size=b_i,
            d_dim=d_dim,
            num_tiles=num_tiles,
            tile_cols=tile_cols,
            row_parallelism=row_parallelism,
            dac_ns=neurosim_dac_ns,
            wl_driver_ns=neurosim_wl_driver_ns,
            cell_read_ns=neurosim_cell_read_ns,
            sense_amp_ns=neurosim_sense_amp_ns,
            adc_ns=neurosim_adc_ns,
            adder_tree_ns=neurosim_adder_tree_ns,
            local_topk_ns=neurosim_local_topk_ns,
            global_interconnect_ns=neurosim_global_interconnect_ns,
            global_topk_ns=neurosim_global_topk_ns,
            dac_energy_fj_per_row=neurosim_dac_energy_fj_per_row,
            cell_read_energy_fj=neurosim_cell_read_energy_fj,
            adc_energy_fj_per_col=neurosim_adc_energy_fj_per_col,
            digital_energy_fj_per_col=neurosim_digital_energy_fj_per_col,
        )
        feram_bucket_time_s += feram_bucket["feram_bucket_time_s"]
        feram_bucket_energy_mj += feram_bucket["feram_bucket_energy_mj"]
        feram_serial_cycles_acc += feram_bucket["feram_bucket_serial_cycles"]
        feram_used_buckets += 1

        hvs_bucket = encoded_hvs[indices]
        if use_gpu_distance:
            dist, matmul_s = _hamming_distance_matrix_gpu(hvs_bucket)
            gpu_matmul_s += matmul_s
        else:
            dist = _hamming_distance_matrix_numpy(hvs_bucket)

        hac_start = time.perf_counter()
        local_clusters = nn_chain_hac_from_distance(dist, threshold_ratio=threshold_ratio)
        hac_end = time.perf_counter()
        cpu_hac_s += hac_end - hac_start
        total_clusters += len(local_clusters)

    end = time.perf_counter()
    return {
        "clustering_seconds": gpu_matmul_s if use_gpu_distance else end - start,
        "gpu_matmul_seconds": gpu_matmul_s,
        "cpu_hac_seconds": cpu_hac_s,
        "wall_seconds": end - start,
        "num_clusters": total_clusters,
        "num_buckets": bucket_count,
        "feram_cluster_core_s": feram_bucket_time_s,
        "feram_cluster_energy_mj": feram_bucket_energy_mj,
        "feram_avg_serial_cycles": (feram_serial_cycles_acc / feram_used_buckets) if feram_used_buckets else 0.0,
    }


def format_seconds(seconds):
    return f"{seconds:.6f}"


def median_value(values):
    return float(np.median(np.array(values, dtype=np.float64)))


def run_benchmark(
    subset_dir,
    subset_sizes,
    gpu_backend="hyperspec",
    hyperspec_root="/home/tsl012/multiomic/genomic_proteomic_CPU_test/Hyper-Spec-linux",
    hyperspec_main_relpath="src/main.py",
    hyperspec_use_gpu_cluster=False,
    hyperspec_cpu_core_preprocess=8,
    hyperspec_cpu_core_cluster=8,
    hyperspec_hd_q=16,
    hyperspec_eps=0.6,
    hyperspec_cluster_alg="dbscan",
    hyperspec_cluster_charges=(2, 3),
    hyperoms_cmd_template=None,
    hyperoms_workdir=None,
    rapids_cluster_eps=0.45,
    max_peaks=50,
    mz_max=2000.0,
    top_k=50,
    seed=42,
    num_tiles=32,
    tile_cols=512,
    row_parallelism=2048,
    neurosim_dac_ns=2.0,
    neurosim_wl_driver_ns=1.0,
    neurosim_cell_read_ns=4.0,
    neurosim_sense_amp_ns=1.0,
    neurosim_adc_ns=8.0,
    neurosim_adder_tree_ns=3.0,
    neurosim_local_topk_ns=6.0,
    neurosim_global_interconnect_ns=5.0,
    neurosim_global_topk_ns=7.0,
    neurosim_dac_energy_fj_per_row=20.0,
    neurosim_cell_read_energy_fj=2.0,
    neurosim_adc_energy_fj_per_col=200.0,
    neurosim_digital_energy_fj_per_col=50.0,
    feram_fixed_batch_overhead_us=120.0,
    feram_fixed_io_us_per_query=0.8,
    feram_fixed_post_us_per_query=0.2,
    fenand_to_feram_bytes_per_query=2048.0,
    fenand_to_feram_bandwidth_mb_s=1600.0,
    fenand_to_feram_setup_overhead_us=50.0,
    encode_hw_cycles_per_query=100,
    encode_hw_clock_hz=500e6,
    encode_hw_energy_pj_per_query=50.0,
    gpu_cluster_power_w=300.0,
    cluster_threshold_ratio=0.45,
    cluster_bucket_width=10.0,
    cluster_max_items_per_bucket=2000,
    benchmark_repeats=7,
):
    if not torch.cuda.is_available():
        raise RuntimeError("找不到 CUDA GPU。請在有 A100/CUDA 的環境執行。")

    print("=" * 84)
    print("PXD000561: FeRAM(NeuroSIM) vs GPU Traditional Clustering")
    print("=" * 84)
    print(f"Subset dir: {subset_dir}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Subset sizes: {subset_sizes}")
    print(f"FeRAM capacity: {num_tiles * tile_cols} vectors ({num_tiles} tiles x {tile_cols} cols)")
    print(f"Repeats per benchmark: {benchmark_repeats} (median)")
    print(
        "NeuroSim params(ns): "
        f"dac={neurosim_dac_ns}, wl={neurosim_wl_driver_ns}, cell={neurosim_cell_read_ns}, "
        f"sa={neurosim_sense_amp_ns}, adc={neurosim_adc_ns}, adder={neurosim_adder_tree_ns}, "
        f"local_topk={neurosim_local_topk_ns}, global_ic={neurosim_global_interconnect_ns}, "
        f"global_topk={neurosim_global_topk_ns}; row_parallelism={row_parallelism}"
    )
    print(
        "NeuroSim params(energy fJ): "
        f"dac/row={neurosim_dac_energy_fj_per_row}, cell={neurosim_cell_read_energy_fj}, "
        f"adc/col={neurosim_adc_energy_fj_per_col}, digital/col={neurosim_digital_energy_fj_per_col}"
    )
    print(
        "FeRAM fixed overhead model(us): "
        f"batch={feram_fixed_batch_overhead_us}, io/query={feram_fixed_io_us_per_query}, "
        f"post/query={feram_fixed_post_us_per_query}"
    )
    print(
        "FeNAND->FeRAM transfer model: "
        f"bytes/query={fenand_to_feram_bytes_per_query:.1f}, "
        f"bandwidth={fenand_to_feram_bandwidth_mb_s:.1f} MB/s, "
        f"setup={fenand_to_feram_setup_overhead_us:.1f} us"
    )
    print(
        "Encode HW model: "
        f"cycles/query={encode_hw_cycles_per_query}, clock_hz={encode_hw_clock_hz:.3e}, "
        f"energy/query={encode_hw_energy_pj_per_query} pJ"
    )
    print(f"GPU cluster power model: {gpu_cluster_power_w:.1f} W")
    print(
        "HyperSpec-like clustering: "
        f"NN-chain threshold_ratio={cluster_threshold_ratio}, "
        f"bucket_width={cluster_bucket_width}, max_items_per_bucket={cluster_max_items_per_bucket}"
    )
    print("-" * 84)

    if not hyperoms_cmd_template:
        raise ValueError("請提供 --hyperoms-cmd-template（homs-tc OMS 執行命令模板）。")

    print(
        "N | load_s | encode_s | pim_cluster_s | pim_oms_s | pim_e2e_s | gpu_rapids_cluster_s | gpu_homs_oms_s | gpu_e2e_s | ratio_gpu_over_pim_cluster | ratio_gpu_over_pim_oms | ratio_gpu_over_pim_e2e"
    )
    print("-" * 84)

    shared_encoder = SpecHDFeRAMSystem(seed=seed)

    for n in subset_sizes:
        subset_file = os.path.join(subset_dir, f"PXD000561_{n}.mgf")
        if not os.path.exists(subset_file):
            raise FileNotFoundError(f"找不到 subset 檔案: {subset_file}")

        load_times = []
        encode_times = []
        pim_cluster_times = []
        pim_oms_times = []
        pim_e2e_times = []
        gpu_cluster_times = []
        gpu_oms_times = []
        gpu_e2e_times = []

        for _ in range(benchmark_repeats):
            load_start = time.perf_counter()
            subset = parse_mgf_file(subset_file, max_peaks=max_peaks, mz_max=mz_max)
            load_end = time.perf_counter()

            if len(subset) < n:
                raise RuntimeError(f"subset 檔案 {subset_file} 光譜不足: 期待 {n}，實際 {len(subset)}")
            subset = subset[:n]

            encode_start = time.perf_counter()
            encoded_hvs = encode_spectra(shared_encoder, subset)
            encode_end = time.perf_counter()

            pim_cluster = estimate_feram_clustering_from_spectra(
                subset,
                bucket_width=cluster_bucket_width,
                max_items_per_bucket=cluster_max_items_per_bucket,
                d_dim=shared_encoder.rows,
                num_tiles=num_tiles,
                tile_cols=tile_cols,
                row_parallelism=row_parallelism,
                neurosim_dac_ns=neurosim_dac_ns,
                neurosim_wl_driver_ns=neurosim_wl_driver_ns,
                neurosim_cell_read_ns=neurosim_cell_read_ns,
                neurosim_sense_amp_ns=neurosim_sense_amp_ns,
                neurosim_adc_ns=neurosim_adc_ns,
                neurosim_adder_tree_ns=neurosim_adder_tree_ns,
                neurosim_local_topk_ns=neurosim_local_topk_ns,
                neurosim_global_interconnect_ns=neurosim_global_interconnect_ns,
                neurosim_global_topk_ns=neurosim_global_topk_ns,
                neurosim_dac_energy_fj_per_row=neurosim_dac_energy_fj_per_row,
                neurosim_cell_read_energy_fj=neurosim_cell_read_energy_fj,
                neurosim_adc_energy_fj_per_col=neurosim_adc_energy_fj_per_col,
                neurosim_digital_energy_fj_per_col=neurosim_digital_energy_fj_per_col,
            )

            pim_oms = estimate_feram_neurosim_latency(
                num_queries=n,
                d_dim=shared_encoder.rows,
                num_tiles=num_tiles,
                tile_cols=tile_cols,
                row_parallelism=row_parallelism,
                dac_ns=neurosim_dac_ns,
                wl_driver_ns=neurosim_wl_driver_ns,
                cell_read_ns=neurosim_cell_read_ns,
                sense_amp_ns=neurosim_sense_amp_ns,
                adc_ns=neurosim_adc_ns,
                adder_tree_ns=neurosim_adder_tree_ns,
                local_topk_ns=neurosim_local_topk_ns,
                global_interconnect_ns=neurosim_global_interconnect_ns,
                global_topk_ns=neurosim_global_topk_ns,
                dac_energy_fj_per_row=neurosim_dac_energy_fj_per_row,
                cell_read_energy_fj=neurosim_cell_read_energy_fj,
                adc_energy_fj_per_col=neurosim_adc_energy_fj_per_col,
                digital_energy_fj_per_col=neurosim_digital_energy_fj_per_col,
            )

            pim_fixed_oms_s = (
                max(0.0, float(feram_fixed_batch_overhead_us))
                + n
                * (
                    max(0.0, float(feram_fixed_io_us_per_query))
                    + max(0.0, float(feram_fixed_post_us_per_query))
                )
            ) * 1e-6
            fenand_transfer = estimate_fenand_to_feram_transfer(
                num_queries=n,
                bytes_per_query=fenand_to_feram_bytes_per_query,
                bandwidth_mb_s=fenand_to_feram_bandwidth_mb_s,
                setup_overhead_us=fenand_to_feram_setup_overhead_us,
            )

            gpu_cluster = benchmark_gpu_rapids_clustering(
                encoded_hvs=encoded_hvs,
                eps=rapids_cluster_eps,
            )

            gpu_oms = benchmark_gpu_hyperoms(
                subset_file=subset_file,
                hyperoms_cmd_template=hyperoms_cmd_template,
                hyperoms_workdir=hyperoms_workdir,
            )

            pim_cluster_s = float(pim_cluster["multiram_cluster_total_s"])
            pim_oms_core_s = float(pim_oms["feram_neurosim_time_s"])
            pim_oms_s = pim_oms_core_s + pim_fixed_oms_s + float(fenand_transfer["fenand_to_feram_total_s"])
            pim_e2e_s = pim_cluster_s + pim_oms_s

            gpu_cluster_s = float(gpu_cluster["gpu_cluster_seconds"])
            gpu_oms_s = float(gpu_oms["gpu_search_seconds"])
            gpu_e2e_s = gpu_cluster_s + gpu_oms_s

            load_times.append(load_end - load_start)
            encode_times.append(encode_end - encode_start)
            pim_cluster_times.append(pim_cluster_s)
            pim_oms_times.append(pim_oms_s)
            pim_e2e_times.append(pim_e2e_s)
            gpu_cluster_times.append(gpu_cluster_s)
            gpu_oms_times.append(gpu_oms_s)
            gpu_e2e_times.append(gpu_e2e_s)

        load_s = median_value(load_times)
        encode_s = median_value(encode_times)
        pim_cluster_s = median_value(pim_cluster_times)
        pim_oms_s = median_value(pim_oms_times)
        pim_e2e_s = median_value(pim_e2e_times)
        gpu_cluster_s = median_value(gpu_cluster_times)
        gpu_oms_s = median_value(gpu_oms_times)
        gpu_e2e_s = median_value(gpu_e2e_times)

        ratio_cluster = gpu_cluster_s / pim_cluster_s if pim_cluster_s > 0 else float("inf")
        ratio_oms = gpu_oms_s / pim_oms_s if pim_oms_s > 0 else float("inf")
        ratio_e2e = gpu_e2e_s / pim_e2e_s if pim_e2e_s > 0 else float("inf")

        print(
            f"{n:5d} | {format_seconds(load_s):>8} | {format_seconds(encode_s):>8} | "
            f"{format_seconds(pim_cluster_s):>13} | {format_seconds(pim_oms_s):>9} | {format_seconds(pim_e2e_s):>9} | "
            f"{format_seconds(gpu_cluster_s):>19} | {format_seconds(gpu_oms_s):>14} | {format_seconds(gpu_e2e_s):>9} | "
            f"{ratio_cluster:>25.3f}x | {ratio_oms:>22.3f}x | {ratio_e2e:>22.3f}x"
        )

        print(
            f"      pim_cluster_us_per_query={(pim_cluster_s / max(1, n)) * 1e6:.3f} "
            f"pim_oms_us_per_query={(pim_oms_s / max(1, n)) * 1e6:.3f} "
            f"gpu_cluster_us_per_query={(gpu_cluster_s / max(1, n)) * 1e6:.3f} "
            f"gpu_oms_us_per_query={(gpu_oms_s / max(1, n)) * 1e6:.3f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark FeRAM(NeuroSIM estimate) vs A100 traditional NN-chain clustering time/energy.")
    parser.add_argument("--dataset-dir", type=str, default="/mnt/hdd/tsunghan/raw-ms-dataset/PXD000561")
    parser.add_argument("--subset-dir", type=str, default="/home/tsl012/multiomic/sumukh_proteomic_test/pxd000561_subsets")
    parser.add_argument("--gpu-backend", type=str, default="hyperspec", choices=["hyperspec", "hyperoms"])
    parser.add_argument("--hyperspec-root", type=str, default="/home/tsl012/multiomic/genomic_proteomic_CPU_test/Hyper-Spec-linux")
    parser.add_argument("--hyperspec-main-relpath", type=str, default="src/main.py")
    parser.add_argument("--hyperspec-use-gpu-cluster", action="store_true")
    parser.add_argument("--hyperspec-cpu-core-preprocess", type=int, default=8)
    parser.add_argument("--hyperspec-cpu-core-cluster", type=int, default=8)
    parser.add_argument("--hyperspec-hd-q", type=int, default=16)
    parser.add_argument("--hyperspec-eps", type=float, default=0.6)
    parser.add_argument("--hyperspec-cluster-alg", type=str, default="dbscan", choices=["dbscan", "hc_single", "hc_complete", "hc_average"])
    parser.add_argument("--hyperspec-cluster-charges", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--hyperoms-cmd-template", type=str, default=None)
    parser.add_argument("--hyperoms-workdir", type=str, default=None)
    parser.add_argument("--rapids-cluster-eps", type=float, default=0.45)
    parser.add_argument("--subset-sizes", type=int, nargs="+", default=[100, 1000, 10000])
    parser.add_argument("--max-peaks", type=int, default=50)
    parser.add_argument("--mz-max", type=float, default=2000.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-tiles", type=int, default=32)
    parser.add_argument("--tile-cols", type=int, default=512)

    parser.add_argument("--row-parallelism", type=int, default=2048)
    parser.add_argument("--neurosim-dac-ns", type=float, default=2.0)
    parser.add_argument("--neurosim-wl-driver-ns", type=float, default=1.0)
    parser.add_argument("--neurosim-cell-read-ns", type=float, default=4.0)
    parser.add_argument("--neurosim-sense-amp-ns", type=float, default=1.0)
    parser.add_argument("--neurosim-adc-ns", type=float, default=8.0)
    parser.add_argument("--neurosim-adder-tree-ns", type=float, default=3.0)
    parser.add_argument("--neurosim-local-topk-ns", type=float, default=6.0)
    parser.add_argument("--neurosim-global-interconnect-ns", type=float, default=5.0)
    parser.add_argument("--neurosim-global-topk-ns", type=float, default=7.0)
    parser.add_argument("--neurosim-dac-energy-fj-per-row", type=float, default=20.0)
    parser.add_argument("--neurosim-cell-read-energy-fj", type=float, default=2.0)
    parser.add_argument("--neurosim-adc-energy-fj-per-col", type=float, default=200.0)
    parser.add_argument("--neurosim-digital-energy-fj-per-col", type=float, default=50.0)
    parser.add_argument("--feram-fixed-batch-overhead-us", type=float, default=120.0)
    parser.add_argument("--feram-fixed-io-us-per-query", type=float, default=0.8)
    parser.add_argument("--feram-fixed-post-us-per-query", type=float, default=0.2)
    parser.add_argument("--fenand-to-feram-bytes-per-query", type=float, default=2048.0)
    parser.add_argument("--fenand-to-feram-bandwidth-mb-s", type=float, default=1600.0)
    parser.add_argument("--fenand-to-feram-setup-overhead-us", type=float, default=50.0)
    parser.add_argument("--encode-hw-cycles-per-query", type=int, default=100)
    parser.add_argument("--encode-hw-clock-hz", type=float, default=500e6)
    parser.add_argument("--encode-hw-energy-pj-per-query", type=float, default=50.0)
    parser.add_argument("--gpu-cluster-power-w", type=float, default=300.0)

    parser.add_argument("--cluster-threshold-ratio", type=float, default=0.45)
    parser.add_argument("--cluster-bucket-width", type=float, default=10.0)
    parser.add_argument("--cluster-max-items-per-bucket", type=int, default=2000)

    parser.add_argument("--benchmark-repeats", type=int, default=7)
    parser.add_argument("--prepare-subsets", action="store_true")
    parser.add_argument("--force-regenerate-subsets", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    subset_sizes = sorted(args.subset_sizes)

    if not 5 <= args.benchmark_repeats <= 10:
        raise ValueError("--benchmark-repeats 必須介於 5 到 10")

    if args.prepare_subsets:
        created_paths = prepare_subset_files(
            dataset_dir=args.dataset_dir,
            subset_dir=args.subset_dir,
            subset_sizes=subset_sizes,
            max_peaks=args.max_peaks,
            mz_max=args.mz_max,
            seed=args.seed,
            force=args.force_regenerate_subsets,
        )
        print("Prepared subset files:")
        for n in subset_sizes:
            print(f"  {n}: {created_paths[n]}")

    if args.prepare_only:
        raise SystemExit(0)

    run_benchmark(
        subset_dir=args.subset_dir,
        subset_sizes=subset_sizes,
        gpu_backend=args.gpu_backend,
        hyperspec_root=args.hyperspec_root,
        hyperspec_main_relpath=args.hyperspec_main_relpath,
        hyperspec_use_gpu_cluster=args.hyperspec_use_gpu_cluster,
        hyperspec_cpu_core_preprocess=args.hyperspec_cpu_core_preprocess,
        hyperspec_cpu_core_cluster=args.hyperspec_cpu_core_cluster,
        hyperspec_hd_q=args.hyperspec_hd_q,
        hyperspec_eps=args.hyperspec_eps,
        hyperspec_cluster_alg=args.hyperspec_cluster_alg,
        hyperspec_cluster_charges=tuple(args.hyperspec_cluster_charges),
        hyperoms_cmd_template=args.hyperoms_cmd_template,
        hyperoms_workdir=args.hyperoms_workdir,
        rapids_cluster_eps=args.rapids_cluster_eps,
        max_peaks=args.max_peaks,
        mz_max=args.mz_max,
        top_k=args.top_k,
        seed=args.seed,
        num_tiles=args.num_tiles,
        tile_cols=args.tile_cols,
        row_parallelism=args.row_parallelism,
        neurosim_dac_ns=args.neurosim_dac_ns,
        neurosim_wl_driver_ns=args.neurosim_wl_driver_ns,
        neurosim_cell_read_ns=args.neurosim_cell_read_ns,
        neurosim_sense_amp_ns=args.neurosim_sense_amp_ns,
        neurosim_adc_ns=args.neurosim_adc_ns,
        neurosim_adder_tree_ns=args.neurosim_adder_tree_ns,
        neurosim_local_topk_ns=args.neurosim_local_topk_ns,
        neurosim_global_interconnect_ns=args.neurosim_global_interconnect_ns,
        neurosim_global_topk_ns=args.neurosim_global_topk_ns,
        neurosim_dac_energy_fj_per_row=args.neurosim_dac_energy_fj_per_row,
        neurosim_cell_read_energy_fj=args.neurosim_cell_read_energy_fj,
        neurosim_adc_energy_fj_per_col=args.neurosim_adc_energy_fj_per_col,
        neurosim_digital_energy_fj_per_col=args.neurosim_digital_energy_fj_per_col,
        feram_fixed_batch_overhead_us=args.feram_fixed_batch_overhead_us,
        feram_fixed_io_us_per_query=args.feram_fixed_io_us_per_query,
        feram_fixed_post_us_per_query=args.feram_fixed_post_us_per_query,
        fenand_to_feram_bytes_per_query=args.fenand_to_feram_bytes_per_query,
        fenand_to_feram_bandwidth_mb_s=args.fenand_to_feram_bandwidth_mb_s,
        fenand_to_feram_setup_overhead_us=args.fenand_to_feram_setup_overhead_us,
        encode_hw_cycles_per_query=args.encode_hw_cycles_per_query,
        encode_hw_clock_hz=args.encode_hw_clock_hz,
        encode_hw_energy_pj_per_query=args.encode_hw_energy_pj_per_query,
        gpu_cluster_power_w=args.gpu_cluster_power_w,
        cluster_threshold_ratio=args.cluster_threshold_ratio,
        cluster_bucket_width=args.cluster_bucket_width,
        cluster_max_items_per_bucket=args.cluster_max_items_per_bucket,
        benchmark_repeats=args.benchmark_repeats,
    )
