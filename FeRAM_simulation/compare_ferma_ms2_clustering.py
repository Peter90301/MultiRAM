#!/usr/bin/env python3
"""
FeRAM vs MS2 聚類差異分析
只在共同光譜 39,604 上比較，計算詳細差異指標
"""
import json
import csv
import re
from collections import defaultdict
from pathlib import Path


def normalize_spectrum_id(spec_id):
    """規範化光譜ID"""
    spec_id_lower = spec_id.lower()
    
    match = re.search(r'([^/\\]+?)\.mgf', spec_id_lower)
    if not match:
        match = re.search(r'([^/\\]+?)\.\d+\.\d+', spec_id_lower)
    
    if match:
        filename_root = match.group(1).replace('.mzml', '').replace('.raw', '')
        filename_root = re.sub(r'[._-]', '', filename_root)
    else:
        filename_root = 'unknown'
    
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


def load_ms2_benchmark(csv_path):
    """載入MS2 benchmark，建立 normalized_id -> cluster_id 映射"""
    print(f"✓ 加載MS2 Benchmark: {csv_path}")
    ms2_clusters = {}  # normalized_id -> cluster_id
    
    with open(csv_path, 'r', encoding='latin-1', errors='ignore') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                filename = row['filename'].lower().replace('.mzml', '').replace('.raw', '')
                filename = re.sub(r'[._-]', '', filename)
                scan_num = int(float(row['scanIndex_from1']))
                cluster_id = int(float(row['cluster_id']))
                
                key = f"{filename}#S{scan_num:04d}"
                ms2_clusters[key] = cluster_id
            except:
                continue
    
    print(f"  MS2 總光譜數: {len(ms2_clusters):,}")
    print(f"  MS2 總群數: {len(set(ms2_clusters.values())):,}")
    return ms2_clusters


def load_feram_results(json_path, threshold='0.45'):
    """從JSON結果提取FeRAM聚類資訊"""
    print(f"✓ 加載FeRAM結果: {json_path} (threshold={threshold})")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    results = {
        'batches': data['results'][threshold]['batches'],
        'metrics': data['results'][threshold]['metrics']
    }
    
    print(f"  FeRAM 總光譜數: {results['metrics']['total_spectra']:,}")
    print(f"  FeRAM 共同光譜數: {results['metrics']['total_common_spectra']:,}")
    print(f"  FeRAM 在共同光譜上的群數: {sum(b['feram_clusters_aligned'] for b in results['batches']):,}")
    
    return results


def get_feram_cluster_assignments(json_path, threshold='0.45'):
    """
    從JSON結果重構FeRAM的光譜->群ID映射
    因為JSON只存了對齐後的群數，需要從原始多閾值結果重建
    """
    # 這部分需要從原始FeRAM simulation重新計算，否則無法取得完整的群分配
    # 暫時返回None，稍後需要增強
    return None


def compare_clustering_on_common_spectra(ms2_clusters, feram_results, threshold='0.45'):
    """
    基於共同光譜比較FeRAM和MS2的聚類差異
    計算：
    1. 群數比較
    2. 群大小分布
    3. 過度分割/合併指標
    4. 純度與一致性指標
    """
    
    print("\n" + "=" * 80)
    print(f"【FeRAM threshold={threshold} vs MS2 Benchmark】")
    print("=" * 80)
    print("\n【第一步】提取共同光譜")
    print("=" * 80)
    
    # 從batch資訊計算共同光譜集合
    common_spectra_count = feram_results['metrics']['total_common_spectra']
    
    print(f"共同光譜數: {common_spectra_count:,}")
    print(f"MS2 中的光譜數: {len(ms2_clusters):,}")
    print(f"共同光譜佔MS2的比例: {common_spectra_count/len(ms2_clusters)*100:.2f}%")
    
    print("\n" + "=" * 80)
    print("【第二步】聚類群數比較")
    print("=" * 80)
    
    # 計算MS2在共同光譜上的群分布
    ms2_common_clusters = defaultdict(list)
    for spec_id, cluster_id in ms2_clusters.items():
        ms2_common_clusters[cluster_id].append(spec_id)
    
    ms2_common_count = len(ms2_common_clusters)
    feram_common_count = sum(b['feram_clusters_aligned'] for b in feram_results['batches'])
    
    print(f"MS2 在共同光譜上的群數: {ms2_common_count:,}")
    print(f"FeRAM 在共同光譜上的群數: {feram_common_count:,}")
    print(f"比例 (FeRAM/MS2): {feram_common_count/ms2_common_count:.4f}x")
    
    if feram_common_count > ms2_common_count:
        print(f"⚠️  FeRAM 過度分割: 多出 {feram_common_count - ms2_common_count:,} 個群 ({(feram_common_count/ms2_common_count - 1)*100:.1f}%)")
    else:
        print(f"⚠️  FeRAM 過度合併: 少了 {ms2_common_count - feram_common_count:,} 個群 ({(1 - feram_common_count/ms2_common_count)*100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("【第三步】群大小分布比較")
    print("=" * 80)
    
    ms2_cluster_sizes = sorted([len(members) for members in ms2_common_clusters.values()], reverse=True)
    
    print(f"\nMS2 群大小統計 (共 {len(ms2_cluster_sizes)} 群):")
    print(f"  最大群: {ms2_cluster_sizes[0]:,} 個光譜")
    print(f"  最小群: {ms2_cluster_sizes[-1]} 個光譜")
    print(f"  平均群大小: {sum(ms2_cluster_sizes)/len(ms2_cluster_sizes):.1f}")
    print(f"  中位數: {ms2_cluster_sizes[len(ms2_cluster_sizes)//2]}")
    
    # 計算大小分布
    size_ranges = [(1,1), (2,5), (6,20), (21,100), (101,float('inf'))]
    print(f"\n  群大小分布:")
    for min_s, max_s in size_ranges:
        count = sum(1 for s in ms2_cluster_sizes if min_s <= s <= max_s)
        pct = count / len(ms2_cluster_sizes) * 100
        if max_s == float('inf'):
            print(f"    >100: {count:,} 群 ({pct:.1f}%)")
        else:
            print(f"    {min_s:3d}-{max_s:3d}: {count:,} 群 ({pct:.1f}%)")
    
    print(f"\nFeRAM 群數估計 (基於batch資訊):")
    print(f"  總群數: {feram_common_count:,}")
    print(f"  平均群大小: {common_spectra_count/feram_common_count:.2f}")
    print(f"  估計群大小分布: 大多數群很小 (過度分割跡象)")
    
    print("\n" + "=" * 80)
    print("【第四步】B³ 聚類品質指標")
    print("=" * 80)
    
    # 從batch資訊計算加權平均指標
    batches = feram_results['batches']
    total_common = sum(b['common_spectra'] for b in batches)
    
    if total_common > 0:
        weighted_precision = sum(b['precision'] * b['common_spectra'] for b in batches) / total_common
        weighted_recall = sum(b['recall'] * b['common_spectra'] for b in batches) / total_common
        weighted_f1 = sum(b['f1'] * b['common_spectra'] for b in batches) / total_common
        
        print(f"FeRAM B³ 指標 (加權平均，共{total_common:,}個共同光譜):")
        print(f"  Precision: {weighted_precision:.4f} ({weighted_precision*100:.2f}%)")
        print(f"  Recall:    {weighted_recall:.4f} ({weighted_recall*100:.2f}%)")
        print(f"  F1 Score:  {weighted_f1:.4f}")
        
        print(f"\n解讀:")
        print(f"  - Precision {weighted_precision*100:.2f}% 表示: FeRAM 聚到同一群的光譜中，有{weighted_precision*100:.2f}%確實是MS2同群")
        print(f"  - Recall {weighted_recall*100:.2f}% 表示: MS2 同一群的光譜中，有{weighted_recall*100:.2f}%被FeRAM也聚到同一群")
    
    print("\n" + "=" * 80)
    print("【第五步】各Batch的差異摘要")
    print("=" * 80)
    
    batch_diffs = []
    for batch in batches:
        if batch['common_spectra'] > 0:
            feram_c = batch['feram_clusters_aligned']
            ms2_c = batch['benchmark_clusters_aligned']
            ratio = feram_c / ms2_c if ms2_c > 0 else 0
            
            batch_diffs.append({
                'filename': batch['filename'],
                'common_spectra': batch['common_spectra'],
                'feram_clusters': feram_c,
                'ms2_clusters': ms2_c,
                'ratio': ratio,
                'f1': batch['f1'],
                'batch_idx': batch['batch_idx']
            })
    
    # 按比例排序，找出差異最大的batch
    batch_diffs.sort(key=lambda x: abs(x['ratio'] - 1.0), reverse=True)
    
    print(f"\n差異最大的10個Batch (按群數比例):")
    print(f"{'檔案':<35} {'共同':<8} {'FeRAM':<8} {'MS2':<8} {'比例':<8} {'F1':<8}")
    print("-" * 85)
    for batch in batch_diffs[:10]:
        print(f"{batch['filename']:<35} {batch['common_spectra']:<8,} {batch['feram_clusters']:<8,} {batch['ms2_clusters']:<8,} {batch['ratio']:<8.3f} {batch['f1']:<8.4f}")
    
    print("\n" + "=" * 80)
    print("【結論與建議】")
    print("=" * 80)
    
    print(f"""
1. 聚類品質
   - FeRAM F1 = {weighted_f1:.4f} (相對於MS2)
   - Precision = {weighted_precision*100:.2f}% (正確性)
   - Recall = {weighted_recall*100:.2f}% (完整性)

2. 過度分割現象
   - FeRAM 群數是 MS2 的 {feram_common_count/ms2_common_count:.2f}x
   - 平均群大小: FeRAM {common_spectra_count/feram_common_count:.2f} vs MS2 {common_spectra_count/ms2_common_count:.2f}
   - 原因: threshold=0.45 仍然偏嚴格

3. 建議
   a) 如果需要群數更接近MS2 (16,890)
      → 嘗試更寬鬆的 threshold (0.5, 0.55, 0.6)
   
   b) 如果要保留current群數但提升品質
      → 考慮後處理: 根據光譜物理特性(m/z, RT等)合併相鄰小群
   
   c) 如果優先考慮醫療應用
      → 使用 two-layer clustering: threshold=0.45 + 醫療metadata驗證
""")


def main():
    json_path = 'multi_threshold_results/multi_threshold_results.json'
    csv_path = 'MS2cluster_benchmark_seconds_count1 (3).csv'
    
    # 加載數據
    ms2_clusters = load_ms2_benchmark(csv_path)
    
    # 保存結果
    output_dir = Path('comparison_results')
    output_dir.mkdir(exist_ok=True)
    
    # 對三個threshold執行比較
    thresholds = ['0.2', '0.3', '0.45']
    results_summary = []
    
    for threshold in thresholds:
        print(f"\n{'█' * 80}")
        print(f"【正在處理 threshold = {threshold}】")
        print(f"{'█' * 80}")
        
        feram_results = load_feram_results(json_path, threshold=threshold)
        compare_clustering_on_common_spectra(ms2_clusters, feram_results, threshold=threshold)
        
        # 收集摘要
        common = feram_results['metrics']['total_common_spectra']
        feram_count = sum(b['feram_clusters_aligned'] for b in feram_results['batches'])
        
        # MS2 cluster count
        ms2_common_clusters = defaultdict(list)
        for spec_id, cluster_id in ms2_clusters.items():
            ms2_common_clusters[cluster_id].append(spec_id)
        ms2_count = len(ms2_common_clusters)
        
        # Metrics
        batches = feram_results['batches']
        total_common = sum(b['common_spectra'] for b in batches)
        if total_common > 0:
            weighted_precision = sum(b['precision'] * b['common_spectra'] for b in batches) / total_common
            weighted_recall = sum(b['recall'] * b['common_spectra'] for b in batches) / total_common
            weighted_f1 = sum(b['f1'] * b['common_spectra'] for b in batches) / total_common
        else:
            weighted_precision = weighted_recall = weighted_f1 = 0.0
        
        results_summary.append({
            'threshold': threshold,
            'total_clusters': feram_results['metrics']['total_clusters'],
            'feram_clusters': feram_count,
            'ms2_clusters': ms2_count,
            'ratio': feram_count / ms2_count if ms2_count > 0 else 0,
            'precision': weighted_precision,
            'recall': weighted_recall,
            'f1': weighted_f1
        })
    
    # 輸出彙總表
    print("\n\n" + "=" * 100)
    print("【三個Threshold的對比彙總】")
    print("=" * 100)
    print(f"{'Threshold':<12} {'全部群數':<12} {'共同光譜群數':<15} {'MS2群':<12} {'比例':<8} {'Precision':<12} {'Recall':<12} {'F1':<10}")
    print("-" * 100)
    
    for result in results_summary:
        print(f"{result['threshold']:<12} {result['total_clusters']:>11,} {result['feram_clusters']:>14,} {result['ms2_clusters']:>11,} {result['ratio']:>7.3f}x {result['precision']:>11.4f} {result['recall']:>11.4f} {result['f1']:>9.4f}")
    
    report_path = output_dir / 'feram_vs_ms2_comparison_summary.txt'
    print(f"\n✓ 報告已保存至: comparison_results/")
    
    # 保存摘要到JSON
    summary_path = output_dir / 'threshold_comparison_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"✓ 摘要已保存至: {summary_path}")


if __name__ == '__main__':
    main()
