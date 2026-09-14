#!/usr/bin/env python3
"""
多参数批次处理 - 对比不同 threshold_ratio 的效果
同时测试 0.2, 0.3, 0.45 三个阈值并与 MS2cluster 基准比较
"""
import os
import time
import json
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict
from FeRAM_simulation import (
    SpecHD_FeRAM_System,
    parse_mgf_file,
    build_transformer_dataset,
    bucket_spectra_by_precursor,
    nn_chain_hac,
)


def normalize_spectrum_id(spec_id):
    """正规化光谱 ID 以便对齐"""
    import re
    spec_id_lower = spec_id.lower()
    
    # 提取文件名根部
    match = re.search(r'([^/\\]+?)\.mgf', spec_id_lower)
    if not match:
        match = re.search(r'file:"([^"]+?)"', spec_id_lower)
    
    if match:
        filename_root = match.group(1).replace('.mzml', '').replace('.raw', '')
        filename_root = re.sub(r'[._-]', '', filename_root)
    else:
        filename_root = 'unknown'
    
    # 提取扫描号
    scan_match = re.search(r'\.(\d+)\.(\d+)\.', spec_id_lower)
    if scan_match:
        scan_num = int(scan_match.group(1))
    else:
        scan_match = re.search(r'scan[=:]?\s*(\d+)', spec_id_lower)
        if scan_match:
            scan_num = int(scan_match.group(1))
        else:
            scan_num = 0
    
    return f"{filename_root}#S{scan_num:04d}"


def load_benchmark_clusters(csv_path):
    """载入 MS2cluster 基准聚类"""
    print(f"Loading benchmark from: {csv_path}")
    benchmark_map = {}
    
    with open(csv_path, 'r', encoding='latin-1', errors='ignore') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                filename = row['filename'].lower().replace('.mzml', '').replace('.raw', '')
                filename = filename.replace('_', '').replace('-', '').replace('.', '')
                scan_num = int(float(row['scanIndex_from1']))
                cluster_id = int(float(row['cluster_id']))
                
                key = f"{filename}#S{scan_num:04d}"
                benchmark_map[key] = cluster_id
            except:
                continue
    
    print(f"Loaded {len(benchmark_map):,} benchmark spectra")
    return benchmark_map


def compute_bcubed_metrics(feram_clusters, benchmark_clusters, common_ids):
    """计算 B³ Precision, Recall, F1"""
    # 创建 ID 到群的映射
    feram_map = {}
    for cluster_id, members in enumerate(feram_clusters):
        for spec_idx in members:
            feram_map[spec_idx] = cluster_id
    
    benchmark_map = {}
    for cluster_id, members in enumerate(benchmark_clusters):
        for spec_idx in members:
            benchmark_map[spec_idx] = cluster_id
    
    precision_sum = 0.0
    recall_sum = 0.0
    n = len(common_ids)
    
    for i, id_i in enumerate(common_ids):
        feram_cluster_i = feram_map.get(i)
        benchmark_cluster_i = benchmark_map.get(i)
        
        if feram_cluster_i is None or benchmark_cluster_i is None:
            continue
        
        # Precision: 在 FeRAM 同一群的，有多少比例在 Benchmark 也同群
        feram_same = [j for j in range(n) if feram_map.get(j) == feram_cluster_i]
        if len(feram_same) > 0:
            correct = sum(1 for j in feram_same if benchmark_map.get(j) == benchmark_cluster_i)
            precision_sum += correct / len(feram_same)
        
        # Recall: 在 Benchmark 同一群的，有多少比例在 FeRAM 也同群
        benchmark_same = [j for j in range(n) if benchmark_map.get(j) == benchmark_cluster_i]
        if len(benchmark_same) > 0:
            correct = sum(1 for j in benchmark_same if feram_map.get(j) == feram_cluster_i)
            recall_sum += correct / len(benchmark_same)
    
    precision = precision_sum / n if n > 0 else 0.0
    recall = recall_sum / n if n > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


class MultiThresholdBatchProcessor:
    def __init__(self, config, thresholds):
        self.config = config
        self.thresholds = thresholds
        self.data_dir = config['data_dir']
        self.output_dir = config['output_dir']
        self.benchmark_path = config['benchmark_path']
        
        # 创建输出目录
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # 加载基准
        self.benchmark_map = load_benchmark_clusters(self.benchmark_path)
        
        # 结果存储
        self.results = {thresh: {'batches': [], 'metrics': {}} for thresh in thresholds}
    
    def get_mgf_files(self):
        """获取所有 MGF 文件"""
        mgf_files = sorted([
            f for f in os.listdir(self.data_dir) 
            if f.lower().endswith('.mgf')
        ])
        return [os.path.join(self.data_dir, f) for f in mgf_files]
    
    def process_file_with_threshold(self, mgf_path, batch_idx, threshold):
        """使用指定 threshold 处理单个文件"""
        filename = os.path.basename(mgf_path)
        print(f"\n[Batch {batch_idx}] {filename} | Threshold={threshold}")
        
        start_time = time.perf_counter()
        
        # 1. 解析 MGF
        spectra = parse_mgf_file(
            mgf_path,
            max_peaks=self.config['max_peaks'],
            mz_max=self.config['mz_max']
        )
        
        if not spectra:
            return None
        
        # 2. 容量限制
        capacity = self.config['num_tiles'] * self.config['tile_cols']
        if len(spectra) > capacity:
            spectra = spectra[:capacity]
        
        # 3. 编码
        accelerator = SpecHD_FeRAM_System(
            num_tiles=self.config['num_tiles'],
            d_dim=self.config['d_dim'],
            tile_cols=self.config['tile_cols']
        )
        
        tokens, hvs, ids = build_transformer_dataset(
            accelerator, spectra,
            max_peaks=self.config['max_peaks'],
            mz_max=self.config['mz_max']
        )
        
        # 4. 聚类
        buckets = bucket_spectra_by_precursor(spectra, bucket_width=self.config['bucket_width'])
        clusters = []
        
        for _, indices in buckets.items():
            if len(indices) == 0:
                continue
            hvs_bucket = hvs[indices]
            local_clusters = nn_chain_hac(
                hvs_bucket,
                threshold_ratio=threshold,
                linkage="complete"
            )
            for cluster in local_clusters:
                clusters.append([indices[i] for i in cluster])
        
        # 5. 对齐并计算指标
        normalized_ids = [normalize_spectrum_id(spec_id) for spec_id in ids]
        common_indices = [i for i, nid in enumerate(normalized_ids) if nid in self.benchmark_map]
        
        if len(common_indices) > 0:
            # 构建对齐后的聚类
            feram_clusters_aligned = []
            benchmark_clusters_aligned = defaultdict(list)
            
            for cluster in clusters:
                aligned_cluster = [i for i, idx in enumerate(common_indices) if idx in cluster]
                if aligned_cluster:
                    feram_clusters_aligned.append(aligned_cluster)
            
            for i, idx in enumerate(common_indices):
                nid = normalized_ids[idx]
                benchmark_cluster_id = self.benchmark_map[nid]
                benchmark_clusters_aligned[benchmark_cluster_id].append(i)
            
            benchmark_clusters_list = list(benchmark_clusters_aligned.values())
            
            # 计算 B³ 指标
            precision, recall, f1 = compute_bcubed_metrics(
                feram_clusters_aligned,
                benchmark_clusters_list,
                common_indices
            )
        else:
            precision = recall = f1 = 0.0
        
        end_time = time.perf_counter()
        
        stats = {
            'batch_idx': batch_idx,
            'filename': filename,
            'threshold': threshold,
            'total_spectra': len(ids),
            'num_clusters': len(clusters),
            'common_spectra': len(common_indices),
            'feram_clusters_aligned': len(feram_clusters_aligned) if len(common_indices) > 0 else 0,
            'benchmark_clusters_aligned': len(benchmark_clusters_list) if len(common_indices) > 0 else 0,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'time': end_time - start_time,
        }
        
        print(f"  Spectra: {len(ids):,} | Clusters: {len(clusters):,} | "
              f"Common: {len(common_indices):,} | F1: {f1:.4f}")
        
        return stats
    
    def process_all(self):
        """处理所有文件，使用所有 threshold"""
        mgf_files = self.get_mgf_files()
        
        print(f"\n{'='*80}")
        print(f"MULTI-THRESHOLD BATCH CLUSTERING")
        print(f"{'='*80}")
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Total MGF files: {len(mgf_files)}")
        print(f"Thresholds to test: {self.thresholds}")
        print(f"{'='*80}\n")
        
        overall_start = time.perf_counter()
        
        # 对每个文件，测试所有 threshold
        for batch_idx, mgf_path in enumerate(mgf_files, start=1):
            print(f"\n{'='*80}")
            print(f"BATCH {batch_idx}/{len(mgf_files)}: {os.path.basename(mgf_path)}")
            print(f"{'='*80}")
            
            for threshold in self.thresholds:
                stats = self.process_file_with_threshold(mgf_path, batch_idx, threshold)
                if stats:
                    self.results[threshold]['batches'].append(stats)
        
        overall_end = time.perf_counter()
        
        # 计算总体指标
        self.compute_overall_metrics()
        
        # 保存结果
        self.save_results(overall_end - overall_start)
        
        # 打印比较
        self.print_comparison()
    
    def compute_overall_metrics(self):
        """计算每个 threshold 的总体指标"""
        for threshold in self.thresholds:
            batches = self.results[threshold]['batches']
            
            if not batches:
                continue
            
            total_spectra = sum(b['total_spectra'] for b in batches)
            total_clusters = sum(b['num_clusters'] for b in batches)
            total_common = sum(b['common_spectra'] for b in batches)
            
            # 加权平均 F1, Precision, Recall (按 common_spectra 加权)
            if total_common > 0:
                weighted_precision = sum(b['precision'] * b['common_spectra'] for b in batches) / total_common
                weighted_recall = sum(b['recall'] * b['common_spectra'] for b in batches) / total_common
                weighted_f1 = sum(b['f1'] * b['common_spectra'] for b in batches) / total_common
            else:
                weighted_precision = weighted_recall = weighted_f1 = 0.0
            
            self.results[threshold]['metrics'] = {
                'total_batches': len(batches),
                'total_spectra': total_spectra,
                'total_clusters': total_clusters,
                'total_common_spectra': total_common,
                'avg_precision': weighted_precision,
                'avg_recall': weighted_recall,
                'avg_f1': weighted_f1,
            }
    
    def save_results(self, total_time):
        """保存结果到 JSON"""
        output_path = os.path.join(self.output_dir, 'multi_threshold_results.json')
        
        output = {
            'config': self.config,
            'thresholds': self.thresholds,
            'total_time': total_time,
            'results': self.results,
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✅ Results saved to: {output_path}")
    
    def print_comparison(self):
        """打印三个 threshold 的对比表格"""
        print(f"\n{'='*80}")
        print(f"THRESHOLD COMPARISON SUMMARY")
        print(f"{'='*80}\n")
        
        # 表头
        print(f"{'Metric':<30} | ", end='')
        for thresh in self.thresholds:
            print(f"Thresh={thresh:<6} | ", end='')
        print()
        print(f"{'-'*80}")
        
        # 各项指标
        metrics_to_show = [
            ('total_batches', 'Total Batches', ''),
            ('total_spectra', 'Total Spectra', ','),
            ('total_clusters', 'Total Clusters', ','),
            ('total_common_spectra', 'Common Spectra', ','),
            ('avg_precision', 'Avg Precision', '.4f'),
            ('avg_recall', 'Avg Recall', '.4f'),
            ('avg_f1', 'Avg F1 Score', '.4f'),
        ]
        
        for key, label, fmt in metrics_to_show:
            print(f"{label:<30} | ", end='')
            for thresh in self.thresholds:
                value = self.results[thresh]['metrics'].get(key, 0)
                if fmt:
                    if fmt == ',':
                        print(f"{value:<13,} | ", end='')
                    else:
                        print(f"{value:<13{fmt}} | ", end='')
                else:
                    print(f"{value:<13} | ", end='')
            print()
        
        print(f"{'-'*80}\n")
        
        # 找出最佳 F1
        best_thresh = max(self.thresholds, key=lambda t: self.results[t]['metrics']['avg_f1'])
        best_f1 = self.results[best_thresh]['metrics']['avg_f1']
        
        print(f"🏆 BEST THRESHOLD: {best_thresh} (F1 = {best_f1:.4f})")
        print(f"{'='*80}\n")


def main():
    # 配置
    config = {
        'data_dir': '/mnt/hdd/tsunghan/raw-ms-dataset/cleaned_w_charge2',
        'output_dir': '/home/tsl012/multiomic/FeRAM_simulation/multi_threshold_results',
        'benchmark_path': '/home/tsl012/multiomic/FeRAM_simulation/MS2cluster_benchmark_seconds_count1 (3).csv',
        
        # FeRAM 硬件
        'num_tiles': 32,
        'd_dim': 2048,
        'tile_cols': 512,
        
        # 处理参数
        'max_peaks': 50,
        'mz_max': 2000.0,
        'bucket_width': 10.0,
    }
    
    # 要测试的阈值
    thresholds = [0.2, 0.3, 0.45]
    
    processor = MultiThresholdBatchProcessor(config, thresholds)
    processor.process_all()


if __name__ == "__main__":
    main()
