import collections

from joblib import Parallel, delayed

import tqdm as tqdm

import numpy as np
import pandas as pd

from sklearn import metrics

from os.path import exists
import hashlib


def calculate_md5(filename):
    with open(filename, "rb") as file_to_check:
        # read contents of the file
        data = file_to_check.read()    
        # pipe contents of the file through
        md5_returned = hashlib.md5(data).hexdigest()
    return md5_returned
        
        
def check_file_exist(file_list):
    non_exist = False
    for f_i in file_list:
        if exists(f_i):
            continue
        else:
            non_exist = True
            print("Non-existing file: ", f_i)
    
    if non_exist:
        raise Exception("Non exist files {}".format(f_i))
    
    # file_md5 = []
    # for f_i in file_list:
    #     md5 = calculate_md5(f_i)
    #     file_md5.append(md5)
        
    # return file_md5


def read_parquet(file_name):
    ids = pd.read_parquet(file_name)

    print ("length of parquet",len(ids))
    ids['sequence'] = ids['sequence'].str.replace('L', 'I')

    if 'scan_identifier' in ids.columns:
        ids.drop('identifier', inplace=True, axis=1)
        ids.rename(columns={'scan_identifier':'identifier'}, inplace=True)

    print("Read parquet file {} successfully!".format(file_name))
    return ids


def _count_majority_label_mismatch(labels):
    labels_assigned = labels.dropna()
    if len(labels_assigned) <= 1:
        return 0
    else:
        return len(labels_assigned) - labels_assigned.value_counts().iat[0]


def evaluate_clusters(clusters, min_cluster_size=None, max_cluster_size=None, charges=None):
    clusters = clusters.copy()

    # Step 1: Get cluster sizes
    cluster_sizes = clusters['cluster'].value_counts()

    # Step 2: Define bins and labels
    bins = [0, 2, 5, 20, 100, np.inf]
    labels = ['0-2', '2-5', '5-20', '20-100', '100+']

    # Step 3: Bin the sizes
    binned = pd.cut(cluster_sizes, bins=bins, labels=labels, right=False)

    # Step 4: Count how many clusters fall into each bin
    result = binned.value_counts().sort_index()

    print(" bin count",result)

    # Handle noise spectra
    idx_noise = clusters['cluster'] == -1
    num_noise = idx_noise.sum()
    if num_noise:
        new_cluster_start = clusters['cluster'].max()+1
        new_cluster_end = new_cluster_start+num_noise
        clusters.loc[idx_noise, ['cluster']] = list(np.arange(new_cluster_start, new_cluster_end))
            
    # Select spec. in charges and re-fine clustering labels
    if charges is not None:
        clusters = clusters[clusters['precursor_charge'].isin(charges)]

        base = 0
        for charge_i in charges:
            idx = clusters['precursor_charge']==charge_i
            clusters.loc[idx, 'cluster'] += base
            base = clusters['cluster'][idx].max()+1


    # Use consecutive cluster labels, skipping the noise points.    
    cluster_map = clusters['cluster'].value_counts(dropna=False)
    if -1 in cluster_map.index:
        cluster_map = cluster_map.drop(index=-1)

    cluster_map = (cluster_map.to_frame().reset_index().reset_index()
                   .rename(columns={'index': 'old', 'level_0': 'new'})
                   .set_index('old')['new'])
    cluster_map = cluster_map.to_dict(collections.defaultdict(lambda: -1))
    clusters['cluster'] = clusters['cluster'].map(cluster_map)
    

    # Only consider clusters with specific minimum (inclusive) and/or
    # maximum (exclusive) size.
    cluster_counts = clusters['cluster'].value_counts(dropna=False)
    if min_cluster_size is not None:
        clusters.loc[clusters['cluster'].isin(cluster_counts[
            cluster_counts < min_cluster_size].index), 'cluster'] = -1
    if max_cluster_size is not None:
        clusters.loc[clusters['cluster'].isin(cluster_counts[
            cluster_counts >= max_cluster_size].index), 'cluster'] = -1


    # Compute cluster evaluation measures.
    # Reassign noise points to singleton clusters.
    noise_mask = clusters['cluster'] == -1
    num_noise = noise_mask.sum()
    num_clustered = len(clusters) - num_noise
    prop_clustered = (len(clusters) - num_noise) / len(clusters)

    clusters_ident = clusters.dropna(subset=['sequence'])
    clusters_ident_non_noise = (clusters[~noise_mask]
                                .dropna(subset=['sequence']))


    # The number of incorrectly clustered spectra is the number of PSMs that
    # differ from the majority PSM. Unidentified spectra are not considered.
    prop_clustered_incorrect = sum(Parallel(n_jobs=-1)(
        delayed(_count_majority_label_mismatch)(clust['sequence'])
        for _, clust in clusters[~noise_mask].groupby('cluster')))
    print('number of incorrect', prop_clustered_incorrect)
    prop_clustered_incorrect /= (len(clusters_ident_non_noise)+1e-6)


    # Purity

    # Homogeneity measures whether clusters contain only identical PSMs.
    # This is only evaluated on non-noise points, because the noise cluster
    # is highly non-homogeneous by definition.
    homogeneity = metrics.homogeneity_score(clusters_ident_non_noise['sequence'], clusters_ident_non_noise['cluster'])
    
    # Completeness measures whether identical PSMs are assigned to the same
    # cluster.
    # This is evaluated on all PSMs, including those clustered as noise.
    completeness = metrics.completeness_score(clusters_ident['sequence'], clusters_ident['cluster'])

    return (num_clustered, num_noise,
            prop_clustered, prop_clustered_incorrect,
            homogeneity, completeness)


def get_clusters_falcon(filename, ids, PXD=None):
    print("Evaluating: ", filename)

    if filename[-3:]=='csv':
        cluster_labels = pd.read_csv(filename, comment='#')
    else:
        cluster_labels = pd.read_parquet(filename)


    if PXD is not None:
        cluster_labels = cluster_labels.assign(
            identifier="mzspec:"+PXD+":"+cluster_labels.identifier+":scan:"+cluster_labels.scan.map(str))

    
    if ids is None:
        return cluster_labels
    else:
        cluster_labels = pd.merge(cluster_labels, ids[['identifier', 'sequence']], how='left', on='identifier')  
        cluster_labels['sequence'] = (
            cluster_labels['sequence'] + '/' +
            cluster_labels['precursor_charge'].astype(str))


        return cluster_labels
    




def run_evaluation(filename_list, ids, charges, min_cluster_sizes):
    def eval_single_file(f_i, ids, min_cluster_sizes, PXD):
        # Evaluate clustering performance.
        cluster_labels = get_clusters_falcon(f_i, ids, PXD)

        min_cluster_size, max_cluster_size = min_cluster_sizes

        num_clustered, num_noise, \
            prop_clustered, prop_clustered_incorrect, \
            homogeneity, completeness = \
                evaluate_clusters(cluster_labels, min_cluster_size,
                                max_cluster_size, charges)
        
        return (f_i, charges,
                min_cluster_size, max_cluster_size,
                num_clustered, num_noise,
                prop_clustered, prop_clustered_incorrect,
                homogeneity, completeness)

    with Parallel(n_jobs=4) as parallel_pool:
        eval_table = parallel_pool(
            delayed(eval_single_file)(f_i, ids, min_cluster_sizes, PXD_i)\
                for f_i, PXD_i in tqdm.tqdm(filename_list))

    return eval_table



parquet_file = '/mnt/hdd/tsunghan/raw-ms-dataset/Hyper-Spec/1468dataset/out_eps0.3.parquet'
# parquet_file = '/keming-volume/data/PXD000561/kim2014_ids.parquet'

PXD = 'USI000000'
# PXD = 'PXD000561'
filename_list = [
    ('/mnt/hdd/tsunghan/raw-ms-dataset/Hyper-Spec/1468dataset/out_eps0.3.parquet', PXD),

    # ('./output/output_01468_hc_complete_D_2048_res_1_eps_0.45_noise_0.00_mlc_2b_sum_adc_7.parquet', PXD),
    # ('./output/output_01468_hc_complete_D_2048_res_1_eps_0.65_noise_0.00_mlc_2b_sum_adc_7.parquet', PXD),


    ]

# parquet_file = '/data/weihong/ms-dataset/PXD000561/kim2014_ids.parquet'
# PXD = 'PXD000561'
# filename_list = [
#     ("/home/lemur/FPGA_Clustering/MAY_7/outputs/output_bucket_PXD000561_hc_complete_eps0.28_width1.25.parquet", PXD)
#     # ("/home/lemur/FPGA_Clustering/MAY_7/outputs/output_bucket_PXD000561_hc_complete_eps0.28_width1.5.parquet", PXD),
#     # ("/home/lemur/FPGA_Clustering/MAY_7/outputs/output_bucket_PXD000561_hc_complete_eps0.28_width1.75.parquet", PXD),
#     # ("/home/lemur/FPGA_Clustering/MAY_7/outputs/output_bucket_PXD000561_hc_complete_eps0.28_width2.parquet", PXD),
#     # ("/home/lemur/FPGA_Clustering/MAY_7/outputs/output_kmeans_PXD000561_hc_complete_eps0.28_width0.1.parquet", PXD),
#     # ("/home/lemur/FPGA_Clustering/MAY_7/outputs/output_kmeans_PXD000561_hc_complete_eps0.28_width0.2.parquet", PXD),
#     # ("/home/lemur/FPGA_Clustering/MAY_7/outputs/output_kmeans_PXD000561_hc_complete_eps0.28_width0.5.parquet", PXD),
#     # ("/home/lemur/FPGA_Clustering/MAY_7/outputs/output_kmeans_PXD000561_hc_complete_eps0.28_width1.25.parquet", PXD),
#     # ("/home/lemur/FPGA_Clustering/MAY_7/outputs/output_kmeans_PXD000561_hc_complete_eps0.28_width1.5.parquet", PXD),
#     # ("/home/lemur/FPGA_Clustering/MAY_7/outputs/output_kmeans_PXD000561_hc_complete_eps0.28_width1.75.parquet", PXD),
#     # ("/home/lemur/FPGA_Clustering/MAY_7/outputs/output_kmeans_PXD000561_hc_complete_eps0.28_width1.parquet", PXD),
#     # ("/home/lemur/FPGA_Clustering/MAY_7/outputs/output_kmeans_PXD000561_hc_complete_eps0.28_width2.parquet", PXD),
# ]


output_csv_filename = 'empty_test.csv'
#output_csv_filename = 'cluster_01468_dbscan_noise_comparison_0.25_initial0.9_increeps_0.35.csv'
# output_csv_filename = 'cluster_00561_mlcsum_noise_comparison.csv'

ids = read_parquet(parquet_file)


min_cluster_sizes = (2, None)

charges = [2,3]
print("Charge: ", charges)


file_list = [f_i[0] for f_i in filename_list]
file_md5 = check_file_exist(file_list)


eval_results = run_evaluation(
    filename_list=filename_list, charges=charges,
    ids=ids, min_cluster_sizes=min_cluster_sizes)


performance = pd.DataFrame(eval_results, columns=[
    'path', 'eval_charges',
    'min_cluster_size', 'max_cluster_size',
    'num_clustered', 'num_noise',
    'prop_clustered', 'prop_clustered_incorrect',
    'homogeneity', 'completeness'])


if PXD is not None:
    output_csv_filename = PXD + '_' + output_csv_filename
performance.to_csv(output_csv_filename, index=False)