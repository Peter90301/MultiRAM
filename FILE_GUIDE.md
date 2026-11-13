# MultiRAM 檔案用途簡介

## 先看這幾個檔案

1. README.md：安裝與執行入口。
2. run_pim_full_pipeline_timing.py：完整流程。
3. GenDP/GenDRAM/：genomics PNM。
4. sumukh_proteomic_test/gpu_benchmark.py：FeRAM clustering。
5. proteomic_full_pipeline/pim_hyperoms_estimator.py：OMS。
6. estimate_pim_model_latency.py：解析式延遲/能源。
7. VALIDATION.md：已通過與未執行的檢查。

以下列出目前版本所有追蹤檔案，一行說明一個用途。既有 benchmark/analysis scripts 不表示本次全部重跑；project_* 是外推模型，不應當作完整資料集實測。
大資料/GPU/下載 scripts 執行前請先核對設定、路徑、磁碟空間與相依套件。

## 逐檔說明

### 根目錄

- `.gitignore`：排除資料集、編譯產物、快取及私人實驗輸出。
- `FENAND_COARSE_FILTER_MODEL.md`：FeNAND 粗篩的資料搬移、延遲與能源模型。
- `FENAND_HSU_MODEL.md`：FeNAND 參數的文獻來源與跨工作負載外推限制。
### FeRAM_simulation

- `FeRAM_simulation.py`：早期 FeRAM/HDC 模擬、光譜讀取及聚類實作。
- `QUICK_START.md`：FeRAM 模組的快速使用方式。
- `README.md`：早期 FeRAM 模組與實驗使用說明。
- `compare_ferma_ms2_clustering.py`：比較 FeRAM 與 MS2 基準的共同光譜聚類結果。
- `generate_comparison_plots.py`：繪製 FeRAM/GPU 效能與能效比較圖。
- `multi_threshold_batch.py`：批次比較不同聚類門檻及 B-cubed 品質。
- `proteomic_sumukh.py`：讀取聚類與標籤資料，進行 Falcon 類聚類品質評估。
### GenDP/GenDRAM

- `__init__.py`：Python 套件初始化。
- `compare_gpu_pim.py`：比較 GPU baseline 與 PIM 模型的 genomics 結果。
- `compare_gpu_pim_new.py`：另一版 GPU/PIM 比較流程，含 query 選取與 FASTA 輸出。
- `components_unified.py`：ALU、Search PE、Compute PE 與 logic-die 元件模型。
- `config_unified.py`：3D DRAM/PIM 硬體與工作負載設定類別。
- `graph_aligner_minigraph_like.py`：Minimizer、sequence graph 與 minigraph-like 比對功能。
- `main_analysis.py`：執行 genomic sequence-to-graph 分析的入口。
- `simulator_unified.py`：PIM 執行、資料庫管理與能源模擬。
### GenDP/mhc_test

- `mhc_ref.fa`：供 smoke test 使用的小型 MHC reference。
- `query_reads_100bp.fq`：供 smoke test 使用的 100-bp reads。
### 根目錄

- `M3D_DRAM_CONFIGURATION.md`：M3D DRAM 容量、tier、channel、bank 與 PU 設定說明。
- `MULTIRAM_CLUSTERING_QUALITY_OPTIMIZATION.md`：HDC 聚類品質調整與驗證說明。
- `README.md`：專案入口、安裝方式、執行範例及結果解讀限制。
### TRPCA-MOE-2/TRPCA_MoE/TRPCA

- `trpca.py`：Transformer、router、expert 與 MoE 模型定義。
### TRPCA-MOE-2/TRPCA_MoE/data

- `train_test_split.npz`：既有 MoE 實驗的 train/test 分割資料；非新上傳資料。
### TRPCA-MOE-2/TRPCA_MoE

- `moe_results.json`：既有 MoE 執行結果快取，可供主流程重用。
- `test_moe.py`：MoE 與 Transformer 訓練/評估比較，不是輕量單元測試。
### 根目錄

- `VALIDATION.md`：這次程式發布通過的檢查，以及未重跑的實驗。
- `estimate_pim_model_latency.py`：由 OMS 統計估計硬體延遲、能源及 read/restore 成本。
### multiram_data_movement/analysis

- `collect_results.py`：合併 profiling CSV 成統一指標表。
- `compute_data_movement.py`：計算資料搬移比例與 MultiRAM 減少比例。
- `evaluate_hga_e2e_mappers.py`：以 SAM truth 評估 minigraph/HGA 的 mapping 結果。
- `parse_nsys.py`：解析 Nsight 與 GPU 功率紀錄。
- `parse_perf.py`：解析 perf stat 與 time -v 的 CPU 紀錄。
- `platinum_genomics_data_movement.py`：彙整 Platinum NA12878 的資料搬移量。
- `plot_results.py`：將資料搬移與 runtime 結果畫成圖。
- `project_data_movement_time_gnr_a100.py`：將量測搬移時間外推到指定 Xeon/A100 設定。
- `project_full_na12878_cpu_hga_pim.py`：外推完整 NA12878 的 CPU/HGA/PIM 時間及搬移成本。
- `project_hga_e2e_subgraph_to_na12878.py`：由 matched-subgraph 實驗外推完整 NA12878。
- `project_matched_mhc_pim_to_na12878.py`：由 matched MHC PIM 模型外推 NA12878。
- `project_na12878_from_paper_throughput.py`：使用論文的 100-bp throughput 因子外推 NA12878。
- `project_pxd024364_pim_runtime.py`：外推 PXD024364 的 FeRAM clustering 與 OMS 時間。
- `project_table_vi_from_component_throughput.py`：由元件 throughput 產生表格與完整 PXD 外推結果。
- `summarize_data_movement_time.py`：彙整 cold/warm cache 與 CUDA copy 的時間占比。
- `summarize_hga_e2e_subgraph_experiment.py`：彙整 matched minigraph/HGA 端到端實驗。
- `summarize_matched_mhc_experiment.py`：彙整 matched MHC 的 CPU、GPU 與 PIM 結果。
### multiram_data_movement/configs

- `example_config.json`：資料搬移實驗的範例設定。
- `hmp2_mgx_mhc_10gb_config.json`：HMP2/MHC 10-GB 情境設定。
- `hmp2_mgx_mhc_config.json`：HMP2/MHC 實驗設定。
- `platinum_na12878_hg38_storage_config.json`：NA12878/hg38 儲存與輸入設定。
### multiram_data_movement/profiling

- `profile_cpu.sh`：執行 CPU profiling wrapper。
- `profile_gpu.sh`：執行 GPU profiling wrapper。
### multiram_data_movement/scripts

- `common.sh`：實驗 shell scripts 共用函式。
- `download_pxd024364_1tb.sh`：下載選定 PXD024364 子集；可能使用大量磁碟空間。
- `estimate_pxd024364_cpu_gpu_oms.py`：比較已完成 CPU/GPU OMS 子集並依 bytes 外推。
- `estimate_pxd024364_cpu_oms.py`：由已完成 SpectraST 結果估計完整 CPU OMS 時間。
- `evict_pxd024364_page_cache.py`：提示 Linux 移出指定檔案快取，供 cold-cache 實驗。
- `measure_platinum_alignment_quality.py`：量測 NA12878 比對品質代理指標。
- `prepare_data_movement_time_inputs.py`：準備固定 genomics 子集與 HGA 輸入。
- `prepare_hga_e2e_subgraphs.py`：準備帶 truth 的 MHC 子圖與 reads。
- `prepare_matched_mhc_inputs.py`：準備並核對 minigraph/HGA/PIM 的一致輸入。
- `prepare_pxd024364_1tb_manifest.py`：產生約 1-TB RAW 子集的下載清單。
- `run_data_movement_time_experiment.py`：執行可重現的 cold/warm CPU/GPU 搬移量測。
- `run_hga_candidate_mapper.py`：先以 minimap2 分派候選，再執行 HGA 子圖比對。
- `run_hga_e2e_subgraph_experiment.py`：執行 matched 子圖的 minigraph/HGA 比較。
- `run_matched_mhc_experiment.py`：執行 matched MHC CPU/GPU 量測。
- `run_matched_mhc_pim_model.py`：執行 matched MHC PIM 模型與搬移量計算。
- `run_matched_rapids_clustering_movement.py`：量測小型 PXD 光譜的 RAPIDS clustering 搬移。
- `run_matrix.py`：產生資料搬移實驗矩陣的命令範例。
- `run_platinum_na12878_hg38_full.sh`：啟動完整 NA12878/hg38 baseline 實驗。
- `run_pxd024364_1tb_experiment.sh`：啟動 PXD024364 大規模實驗流程。
- `run_pxd024364_baseline_stage.py`：對轉換後 MGF 執行可續跑的 CPU/GPU baseline。
- `run_pxd024364_gpu_oms.sh`：啟動 PXD024364 GPU OMS 流程。
- `run_pxd024364_raw_conversion.py`：將 RAW 轉成 MS2 MGF，保存可續跑的量測紀錄。
- `status_platinum_alignment_quality.py`：查看 NA12878 比對品質量測進度。
- `status_platinum_na12878_hg38_full.py`：查看完整 NA12878 minimap2 流程進度。
- `status_pxd024364_download.py`：依檔案大小核對 PXD 下載進度。
- `status_pxd024364_experiment.py`：查看 PXD 下載及各實驗階段進度。
- `summarize_pxd024364_experiment.py`：合併已完成 PXD 單檔指標成 JSON/Markdown。
- `verify_pxd024364_local_sha256.py`：產生可續跑的本地 SHA-256 清單。
### multiram_data_movement/tests

- `hga_tiny_linear.graph`：HGA 的微型線性圖測試輸入。
- `hga_tiny_reads.txt`：HGA 的微型 reads 測試輸入。
### multiram_data_movement/workloads/genomics

- `genomics_minimap2.sh`：單端 minimap2 workload wrapper。
- `genomics_minimap2_pairs.sh`：雙端 minimap2 workload wrapper。
### multiram_data_movement/workloads/oms

- `oms_cpu.sh`：CPU OMS workload wrapper。
- `oms_gpu.sh`：GPU OMS workload wrapper。
### multiram_data_movement/workloads/proteomics_clustering

- `proteomics_clustering_cpu.sh`：CPU clustering workload wrapper。
- `proteomics_clustering_gpu.sh`：GPU clustering workload wrapper。
### proteomic_full_pipeline

- `audit_oms_open_stage.py`：重播標準搜尋/FDR，記錄真正進入 open-search retry 的 query。
- `cpu_falcon_surrogate.py`：Falcon-style CPU clustering surrogate，非原版 Falcon。
- `evaluate_clustering_quality.py`：以 peptide truth 比較 Falcon、RAPIDS、MultiRAM 聚類。
- `fenand_coarse_filter.py`：FeNAND 上界粗篩、exact Top-K 與 recall/reduction 工具。
- `hyperoms_hek293_paper.ini`：HEK293/HyperOMS 實驗參數。
- `iprg2012_loose.ini`：iPRG2012 較寬鬆設定。
- `iprg2012_lowmem.ini`：iPRG2012 低記憶體設定。
- `iprg2012_relaxed.ini`：iPRG2012 放寬條件的設定。
- `msfragger_dummy_db.fasta`：MSFragger 流程用的 dummy database，不是生物分析資料庫。
- `pim_hyperoms_estimator.py`：HDC OMS 功能模擬、target-decoy 過濾與 FeRAM 成本估計。
- `run_b1926_b1930_oms_counts.py`：執行 HEK293 b1926–b1930 的 CPU/GPU/PIM OMS 計數。
- `run_homstc_chunked.py`：將 query 分塊執行 homs-tc，再合併 mzTab。
- `run_hyperspec_gpu_cluster.py`：以單一 MGF 執行 Hyper-Spec GPU clustering。
- `run_oms_filter_sweep.py`：執行標準搜尋的 FeNAND 粗篩敏感度分析。
- `run_oms_open_filter_sweep.py`：對 open-search retry 執行不設候選數上限的敏感度分析。
- `run_pim_chunked.py`：分塊執行 PIM OMS。
- `sweep_pim_hdc_calibration.py`：對 query 子集掃描 HDC OMS 校準設定。
- `test_fenand_coarse_filter.py`：粗篩、Top-K tie-breaking 與指標單元測試。
- `validate_multiram_clustering_series.py`：跨 HEK293 樣本驗證 MultiRAM 聚類品質。
### 根目錄

- `recalculate_ee_with_feram_restore.py`：加入 FeRAM restore、M3D 與 FeNAND 成本，重算能源效率。
- `recalculate_package_link_energy.py`：依傳輸量與每 bit 能源重算封裝連線能耗。
- `requirements-dev.txt`：測試用相依套件，包含 pytest。
- `requirements.txt`：Python 相依套件清單。
- `run_pim_full_pipeline_timing.py`：主入口：串接 genomics、clustering、OMS、MoE 與傳輸時間。
### sumukh_proteomic_test

- `FeRAM_aligned_encoding.py`：與 Hyper-Spec 風格對齊的 HDC 編碼及比較工具。
- `gpu_benchmark.py`：FeRAM/GPU 聚類 benchmark、成本模型與光譜處理。
### sumukh_proteomic_test/homs-tc

- `LICENSE`：保留 HOMSTC 第三方授權條款。
### sumukh_proteomic_test/homs-tc/configs

- `hek293.ini`：HOMSTC 的 HEK293 搜尋設定。
### sumukh_proteomic_test/homs-tc/homs_tc

- `__init__.py`：Python 套件初始化。
- `definition.py`：HOMSTC 設定與 spectrum-spectrum match 資料型別。
- `fdr_postprocessing.py`：將搜尋輸出轉成識別結果並執行 FDR 後處理。
- `filter.py`：target-decoy FDR 與分組 FDR 過濾。
- `parser.pyx`：SpectraST spectral library 的 Cython parser。
- `preprocess_utils.py`：光譜有效性、強度正規化與向量化工具。
- `preprocessor.py`：MGF 與 spectral library 預處理。
- `reader.py`：讀取 MGF 與 spectral library。
- `writer.py`：輸出 mzTab 識別結果。
### sumukh_proteomic_test/homs-tc

- `requirements.txt`：HOMSTC 原始套件版本清單；不是整個 repo 的統一 lockfile。
- `setup.py`：編譯 HOMSTC 的 Cython parser extension。
### sumukh_proteomic_test/pxd000561_subsets

- `PXD000561_100.mgf`：既有的 100-spectrum smoke-test 子集。
### tests

- `test_clustering_legacy.py`：保留原先 precursor、bucket、Hamming 門檻與成本回歸測試。
- `test_release.py`：編碼 batch 成本與主流程 import 回歸測試。

## 這次沒有上傳的內容

完整資料集、checkpoint、新產生結果、編譯 binary、私有 audit/審稿紀錄與第三方 thermal solver 未納入這次更新。原始工作目錄保留。
本文件依目前程式的模組、docstring 與入口整理；歷史腳本可能仍需依實際環境調整路徑。
