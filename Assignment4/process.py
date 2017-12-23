#%load_ext autoreload
#%autoreload 2
# Get the final results for the ICLR paper.
import pandas as pd
import glob
import os
import numpy as np
from results_utils import filter_result_files
from results_utils import latest_checkpoints
from results_utils import filename_to_method
from results_utils import extract_global_step
from results_utils import compute_mean_std
from IPython.display import display, HTML

import cPickle as pickle
from collections import namedtuple
from collections import defaultdict

Query = namedtuple('Query', ['label', 'mask'])

def build_latex_results(
    te_4_iid='',
    jm_4_iid='',
    bi_4_iid='',
    te_3_iid='',
    jm_3_iid='',
    bi_3_iid='',
    te_2_iid='',
    jm_2_iid='',
    bi_2_iid='',
    te_1_iid='',
    jm_1_iid='',
    bi_1_iid='',
    te_4_cmp='',
    jm_4_cmp='',
    bi_4_cmp='',
    te_4_cor_iid_pct=0.0, 
    te_4_cor_iid_err=0.0, 
    jm_4_cor_iid_pct=0.0, 
    jm_4_cor_iid_err=0.0, 
    bi_4_cor_iid_pct=0.0, 
    bi_4_cor_iid_err=0.0, 
    te_3_cov_iid_pct=0.0, 
    te_3_cov_iid_err=0.0, 
    jm_3_cov_iid_pct=0.0, 
    jm_3_cov_iid_err=0.0, 
    bi_3_cov_iid_pct=0.0, 
    bi_3_cov_iid_err=0.0, 
    te_3_cor_iid_pct=0.0, 
    te_3_cor_iid_err=0.0, 
    jm_3_cor_iid_pct=0.0, 
    jm_3_cor_iid_err=0.0, 
    bi_3_cor_iid_pct=0.0, 
    bi_3_cor_iid_err=0.0, 
    te_2_cov_iid_pct=0.0, 
    te_2_cov_iid_err=0.0, 
    jm_2_cov_iid_pct=0.0, 
    jm_2_cov_iid_err=0.0, 
    bi_2_cov_iid_pct=0.0, 
    bi_2_cov_iid_err=0.0, 
    te_2_cor_iid_pct=0.0, 
    te_2_cor_iid_err=0.0, 
    jm_2_cor_iid_pct=0.0, 
    jm_2_cor_iid_err=0.0, 
    bi_2_cor_iid_pct=0.0, 
    bi_2_cor_iid_err=0.0, 
    te_1_cov_iid_pct=0.0, 
    te_1_cov_iid_err=0.0, 
    jm_1_cov_iid_pct=0.0, 
    jm_1_cov_iid_err=0.0, 
    bi_1_cov_iid_pct=0.0, 
    bi_1_cov_iid_err=0.0, 
    te_1_cor_iid_pct=0.0, 
    te_1_cor_iid_err=0.0, 
    jm_1_cor_iid_pct=0.0, 
    jm_1_cor_iid_err=0.0, 
    bi_1_cor_iid_pct=0.0, 
    bi_1_cor_iid_err=0.0, 
    te_4_cor_cmp_pct=0.0, 
    te_4_cor_cmp_err=0.0, 
    jm_4_cor_cmp_pct=0.0, 
    jm_4_cor_cmp_err=0.0, 
    bi_4_cor_cmp_pct=0.0, 
    bi_4_cor_cmp_err=0.0,
    **_
  ):

  return r"""
\begin{{table}}
  \centering
  \begin{{tabular}}{{cccc}}
    \toprule

    \textbf{{Method}}  & \textbf{{\#Attributes}} & \textbf{{Coverage}} (\%)                        & \textbf{{\Correctness}} (\%)                                \\

    \toprule
    \rowcolor{{Gray}}\multicolumn{{4}}{{c}}{{\textbf{{\iid}}}}\\
    \toprule
    % {te_4_iid}
    \telbo  & \multirow{{3}}{{*}}{{4}} & -                                               & {te_4_cor_iid_pct:.2f} {{\tiny$\pm$}} {te_4_cor_iid_err:.2f}\\
    % {jm_4_iid}
    \jmvae  &  & -                                               & {jm_4_cor_iid_pct:.2f} {{\tiny$\pm$}} {jm_4_cor_iid_err:.2f}  \\
    % {bi_4_iid}
    \bivcca &  & -                                               & {bi_4_cor_iid_pct:.2f} {{\tiny$\pm$}} {bi_4_cor_iid_err:.2f} \\

    \midrule

    % {te_3_iid}
    \telbo  & \multirow{{3}}{{*}}{{3}} & {te_3_cov_iid_pct:.2f} {{\tiny$\pm$}} {te_3_cov_iid_err:.2f} & {te_3_cor_iid_pct:.2f} {{\tiny$\pm$}} {te_3_cor_iid_err:.2f} \\
    % {jm_3_iid}
    \jmvae  & & {jm_3_cov_iid_pct:.2f} {{\tiny$\pm$}} {jm_3_cov_iid_err:.2f} & {jm_3_cor_iid_pct:.2f} {{\tiny$\pm$}} {jm_3_cor_iid_err:.2f}  \\
    % {bi_3_iid}
    \bivcca & & {bi_3_cov_iid_pct:.2f} {{\tiny$\pm$}} {bi_3_cov_iid_err:.2f} & {bi_3_cor_iid_pct:.2f} {{\tiny$\pm$}} {bi_3_cor_iid_err:.2f}  \\

    \midrule

    % {te_2_iid}
    \telbo  & \multirow{{3}}{{*}}{{2}} & {te_2_cov_iid_pct:.2f} {{\tiny$\pm$}} {te_2_cov_iid_err:.2f} & {te_2_cor_iid_pct:.2f} {{\tiny$\pm$}} {te_2_cor_iid_err:.2f}  \\
    % {jm_2_iid}
    \jmvae  &  & {jm_2_cov_iid_pct:.2f} {{\tiny$\pm$}} {jm_2_cov_iid_err:.2f} & {jm_2_cor_iid_pct:.2f} {{\tiny$\pm$}} {jm_2_cor_iid_err:.2f} \\
    % {bi_2_iid}
    \bivcca &  & {bi_2_cov_iid_pct:.2f} {{\tiny$\pm$}} {bi_2_cov_iid_err:.2f} & {bi_2_cor_iid_pct:.2f} {{\tiny$\pm$}} {bi_2_cor_iid_err:.2f} \\

    \midrule

    % {te_1_iid}
    \telbo  & \multirow{{3}}{{*}}{{1}} & {te_1_cov_iid_pct:.2f} {{\tiny$\pm$}} {te_1_cov_iid_err:.2f} & {te_1_cor_iid_pct:.2f} {{\tiny$\pm$}} {te_1_cor_iid_err:.2f} \\
    % {jm_1_iid}
    \jmvae  & & {jm_1_cov_iid_pct:.2f} {{\tiny$\pm$}} {jm_1_cov_iid_err:.2f} & {jm_1_cor_iid_pct:.2f} {{\tiny$\pm$}} {jm_1_cor_iid_err:.2f} \\
    % {bi_1_iid}
    \bivcca & & {bi_1_cov_iid_pct:.2f} {{\tiny$\pm$}} {bi_1_cov_iid_err:.2f} & {bi_1_cor_iid_pct:.2f} {{\tiny$\pm$}} {bi_1_cor_iid_err:.2f} \\
    
    \toprule
    \rowcolor{{Gray}}\multicolumn{{4}}{{c}}{{\textbf{{\comp}}}}\\
    \toprule
    % {te_4_cmp}
    \telbo  & \multirow{{3}}{{*}}{{4}} & -                                               & {te_4_cor_cmp_pct:.2f} {{\tiny$\pm$}} {te_4_cor_cmp_err:.2f}  \\
    % {jm_4_cmp}
    \jmvae  & & -                                               & {jm_4_cor_cmp_pct:.2f} {{\tiny$\pm$}} {jm_4_cor_cmp_err:.2f}  \\
    % {bi_4_cmp}
    \bivcca & & -                                               & {bi_4_cor_cmp_pct:.2f} {{\tiny$\pm$}} {bi_4_cor_cmp_err:.2f}  \\

    \bottomrule
  \end{{tabular}}
\end{{table}}
  """.format(**locals()).replace('-                                               &', '-            &').replace(r'(\%)                        &', r'(\%) &').replace(r'(\%)                               &', r'(\%) &')

# Load the results files and put them in a nice pandas dataframe.
def plot_metrics(result_files, filt=('multimodal_elbo'), metrics=[(4.0, 'comprehensibility')], ban='_val_iclr_mnista_fresh'):
    results_latest = [x for x in result_files if filt in x]
    results_data = defaultdict(dict)
    results_data_mean_only = defaultdict(dict)
    for rfile in results_latest:
        pf = pickle.load(open(rfile, 'r'))

        # Iterate through the file and extract the mean and standard deviation in performance
        # across multiple splits.
        for metric in metrics:
            if metric[-1] == 'comprehensibility':
                metric_value, metric_std = compute_mean_std(pf[metric])
                metric_value = 1 - metric_value/metric[0]
                metric_std = metric_std/metric[0]
            else:
                metric_value, metric_std = compute_mean_std(pf[metric])
                
            metric_str = str(metric[0]) + '_' + str(metric[1])
                
            results_data[metric_str][filename_to_method(rfile, ban=ban)] = (metric_value, metric_std)
            results_data_mean_only[metric_str][filename_to_method(rfile, ban=ban)] = metric_value
        results_data['global_step'][filename_to_method(rfile, ban=ban)] = extract_global_step(rfile)  
    
    results_data = pd.DataFrame(results_data)
    results_data_mean_only = pd.DataFrame(results_data_mean_only)
    display(results_data)
    
    return results_data, results_data_mean_only

def pick_best_method(data_frame, sort_by='consolidated_jsd_sim'):
    """NOTE: data_frame should not have the std term, just mean."""
    cols = list(data_frame)
    col_subset = [x for x in cols if sort_by in x]
    # NOTE: Assumes that larger is better!!
    sorted_data_frame = pd.DataFrame(
        data_frame[col_subset].sum(axis=1)).sort_values(0, ascending=False)
    return sorted_data_frame.index[0]

SPLIT = 'val'
EXP_PREFIX = 'iclr_mnista_fresh_iid'
PATH_TO_RESULTS = './'#('./%s_%s' % (EXP_PREFIX, SPLIT))

result_files = glob.glob(PATH_TO_RESULTS + "/*.p")
results_to_show = latest_checkpoints(filter_result_files(result_files))

_, triple_elbo_results = plot_metrics(results_to_show, filt='multimodal_elbo', metrics=[(4.0, 'comprehensibility'),
                                                               (3.0, 'comprehensibility'),
                                                               (2.0, 'comprehensibility'),
                                                               (1.0, 'comprehensibility'),
                                                               (3.0, 'parametric_jsd_sim'),
                                                               (2.0, 'parametric_jsd_sim'),
                                                               (1.0, 'parametric_jsd_sim'),
                                                               (1.0, 'parametric_consolidated_jsd_sim'),
                                                               (2.0, 'parametric_consolidated_jsd_sim'),
                                                               (3.0, 'parametric_consolidated_jsd_sim'),
                                                               (4.0, 'parametric_consolidated_jsd_sim'),
                                                              ])

_, jmvae_results = plot_metrics(results_to_show, filt='jmvae', metrics=[(4.0, 'comprehensibility'),
                                                               (3.0, 'comprehensibility'),
                                                               (2.0, 'comprehensibility'),
                                                               (1.0, 'comprehensibility'),
                                                               (3.0, 'parametric_jsd_sim'),
                                                               (2.0, 'parametric_jsd_sim'),
                                                               (1.0, 'parametric_jsd_sim'),
                                                               (1.0, 'parametric_consolidated_jsd_sim'),
                                                               (2.0, 'parametric_consolidated_jsd_sim'),
                                                               (3.0, 'parametric_consolidated_jsd_sim'),
                                                               (4.0, 'parametric_consolidated_jsd_sim'),             
                                                              ])

_, bivcca_results = plot_metrics(results_to_show, filt='bivcca', metrics=[(4.0, 'comprehensibility'),
                                                               (3.0, 'comprehensibility'),
                                                               (2.0, 'comprehensibility'),
                                                               (1.0, 'comprehensibility'),
                                                               (3.0, 'parametric_jsd_sim'),
                                                               (2.0, 'parametric_jsd_sim'),
                                                               (1.0, 'parametric_jsd_sim'),
                                                               (1.0, 'parametric_consolidated_jsd_sim'),
                                                               (2.0, 'parametric_consolidated_jsd_sim'),
                                                               (3.0, 'parametric_consolidated_jsd_sim'),
                                                               (4.0, 'parametric_consolidated_jsd_sim'),
                                                              ])															  
# Load test numbers.
SPLIT = 'test'
EXP_PREFIX = 'iclr_mnista_fresh_iid'
PATH_TO_RESULTS = './'#('/coc/scratch/rvedantam3/runs/imagination/%s_%s' % (EXP_PREFIX, SPLIT))

result_files = glob.glob(PATH_TO_RESULTS + "/*.p")
results_to_show = latest_checkpoints(filter_result_files(result_files))

raw_triple_elbo_results, _ = plot_metrics(results_to_show, filt='multimodal_elbo', metrics=[(4.0, 'comprehensibility'),
                                                               (3.0, 'comprehensibility'),
                                                               (2.0, 'comprehensibility'),
                                                               (1.0, 'comprehensibility'),
                                                               (3.0, 'parametric_jsd_sim'),
                                                               (2.0, 'parametric_jsd_sim'),
                                                               (1.0, 'parametric_jsd_sim'),
                                                               (1.0, 'parametric_consolidated_jsd_sim'),
                                                               (2.0, 'parametric_consolidated_jsd_sim'),
                                                               (3.0, 'parametric_consolidated_jsd_sim'),
                                                               (4.0, 'parametric_consolidated_jsd_sim'),
                                                              ], ban='_test_iclr_mnista_fresh')
															  
raw_jmvae_results, _ = plot_metrics(results_to_show, filt='jmvae', metrics=[(4.0, 'comprehensibility'),
                                                               (3.0, 'comprehensibility'),
                                                               (2.0, 'comprehensibility'),
                                                               (1.0, 'comprehensibility'),
                                                               (3.0, 'parametric_jsd_sim'),
                                                               (2.0, 'parametric_jsd_sim'),
                                                               (1.0, 'parametric_jsd_sim'),
                                                               (1.0, 'parametric_consolidated_jsd_sim'),
                                                               (2.0, 'parametric_consolidated_jsd_sim'),
                                                               (3.0, 'parametric_consolidated_jsd_sim'),
                                                               (4.0, 'parametric_consolidated_jsd_sim'),             
                                                              ], ban='_test_iclr_mnista_fresh')
															  
raw_bivcca_results, _ = plot_metrics(results_to_show, filt='bivcca', metrics=[(4.0, 'comprehensibility'),
                                                               (3.0, 'comprehensibility'),
                                                               (2.0, 'comprehensibility'),
                                                               (1.0, 'comprehensibility'),
                                                               (3.0, 'parametric_jsd_sim'),
                                                               (2.0, 'parametric_jsd_sim'),
                                                               (1.0, 'parametric_jsd_sim'),
                                                               (1.0, 'parametric_consolidated_jsd_sim'),
                                                               (2.0, 'parametric_consolidated_jsd_sim'),
                                                               (3.0, 'parametric_consolidated_jsd_sim'),
                                                               (4.0, 'parametric_consolidated_jsd_sim'),
                                                              ], ban='_test_iclr_mnista_fresh')
															  
# best_triple_elbo = pick_best_method(triple_elbo_results)
# best_jmvae = pick_best_method(jmvae_results)
best_bivcca = pick_best_method(bivcca_results)


