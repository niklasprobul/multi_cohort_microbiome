#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import seaborn as sns

from pathlib import Path
from prep_data_modular import prep_data

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import make_scorer, get_scorer
from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm
from tqdm.contrib.telegram import tqdm as telegram_tqdm

import json
import numpy as np
import os
from os import path
import pandas as pd
import pickle
import time
from utility import *
import traceback
from dataclasses import dataclass
import logging
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from datetime import datetime, timedelta

from itertools import combinations, compress
from scipy.stats import ttest_ind 
from statsmodels.stats.multitest import multipletests


# ### meta

# In[2]:


REPETITION = 10 if is_server() else 1 # auto sets rep to 10 when executing on server
CROSS_VAL = 5
RANDOM_STATE = 42
EXEC_MODE = 'load'

# logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.ERROR)

# default 
N_ESTIMATOR = 1000
MAX_DEPTH = 5
CRITERION = 'entropy'
MIN_SAMPLES_LEAF = 5 
MAX_FEATURES = 'sqrt' # 0.3
MAX_SAMPLES = 0.75 # None # = all
CLASS_WEIGHT = 'balanced'

metrics = ["test_roc_auc", "test_f1", "test_mcc", "test_pr_auc"] #, "test_precision"]
# groups = ["study_accession", "country"]
groups = ["cohort_name"]

# # thomas et al
# N_ESTIMATOR = 1000
# MAX_DEPTH = None
# CRITERION = 'entropy'
# MIN_SAMPLES_LEAF = 5
# MAX_FEATURES = 0.3
# MAX_SAMPLES = None # = all
# CLASS_WEIGHT = None

N_JOBS=-1
MAX_CONCURRENT = 10
# basic preprocessing and concatination of raw input files
REDO_DATA_FORMATING = False


# In[3]:


if REDO_DATA_FORMATING:
    prep_data(verbose=True)


# In[4]:


secrets_file = "./secrets.json"

try:
    with open(secrets_file) as f:
        secrets = json.load(f)
        telegram_token = secrets['telegram_token']
        telegram_chatid = secrets['telegram_chatid']
except:
    print("Secrets file not found. No Telegram notifications will be sent.")
    telegram_token = ""
    telegram_chatid = ""


# In[5]:


#data_dir = '/media/niklas/T7/data/FeMAI/source_data/'
data_dir = './prep_data'
data_dir = Path(data_dir)

taxa = 'taxa.csv'
func = 'func.csv'
taxa_func = 'taxa_func.csv'

taxa_meta4 = 'taxa_meta4.csv'

taxa_counts_0 = 'taxa_counts_0.csv'
func_counts_0 = 'func_counts_0.csv'
taxa_func_counts_0 = 'taxa_func_counts_0.csv'

taxa_counts_10 = 'taxa_counts_10.csv'
func_counts_10 = 'func_counts_10.csv'
taxa_func_counts_10 = 'taxa_func_counts_10.csv'

#anno_raw = 'anno_full_raw.csv' # remove to_exclude
anno_all = 'anno_full_clean.csv' # + remove NAN in metadata

default_out = './data'
output_dir = default_out

fig_dir = os.path.join(output_dir, 'figures')
Path(fig_dir).mkdir(exist_ok=True, parents=True)

save_dir = os.path.join(output_dir, 'simulations')
Path(save_dir).mkdir(exist_ok=True, parents=True)

table_dir = os.path.join(save_dir, 'tables')
Path(table_dir).mkdir(exist_ok=True, parents=True)


# ### data prep

# #### data handling

# In[6]:


df_anno = pd.read_csv(path.join(data_dir, anno_all), index_col=0)


# In[7]:


df_anno.columns


# In[8]:


df_anno.groupby(by=['study_accession', 'country', 'cohort_name'])['MGS'].mean()


# In[9]:


grouped_df = df_anno.loc[~df_anno['cohort_name'].isin(['Canada1', 'India1', 'India2'])].groupby(['study_accession', 'country','cohort_name']).agg(
    num_samples=('sample_accession', 'count'),
    CRC=('health_status', lambda x: (x == 1).sum()),
    healthy=('health_status', lambda x: (x == 0).sum()),
    female=('gender', lambda x: (x == 0).sum()),
    male=('gender', lambda x: (x == 1).sum()),
    age_mean=('age', 'mean'),
    bmi_mean=('bmi', 'mean')
).reset_index()

# Create final table by adding other columns manually
final_table = pd.DataFrame({
    'Alias': grouped_df['cohort_name'],
    'Num. samples': grouped_df['num_samples'],
    'CRC': grouped_df['CRC'],
    '% CRC': np.round(grouped_df['CRC'] / grouped_df['num_samples'] * 100, 2),
    'Adenoma': 0,
    'Healthy': grouped_df['healthy'],
    'Female': grouped_df['female'],
    'Male': grouped_df['male'],
    'NA': 0,  # Placeholder for NA column
    'BMI Mean': np.round(grouped_df['bmi_mean'], 2),
    'Age Mean': np.round(grouped_df['age_mean'], 2)
})


# In[10]:


final_table.to_csv(os.path.join(table_dir, 'final_table.csv'), index=False)


# In[11]:


features = ['age', 'bmi', 'gender', 'health_status', 'country']
df_vars = df_anno[features]


# thomas et al 2019 paper :
# PRJNA447983	
# https://www.nature.com/articles/s41591-019-0405-7

# In[12]:


thomas_alias = {
    'PRJDB4176': 'V_Cohort2',
    'PRJEB10878': 'YuJ_2015',
    'PRJEB12449': 'VogtmannE_2016',
    'PRJEB27928': 'V_Cohort1',
    'PRJEB6070':  'ZellerG_2014',
    'PRJEB7774': 'FengQ_2015',
    'PRJNA389927': 'HanniganGD_2018',
}


# In[13]:


df_anno[['thomas_alias', 'study_accession', 'country']].value_counts(sort=False)


# ### central sim

# In[14]:


scoring = {
    'accuracy': get_scorer('accuracy'), 
    'balanced_accuracy': get_scorer('balanced_accuracy'), 
    'roc_auc': get_scorer('roc_auc'),
    'pr_auc': make_scorer(pr_auc_score),
    'f1': get_scorer('f1'),
    #'f2': make_scorer(fbeta_score, beta=2.0, zero_devision=np.nan),
    'precision': make_scorer(precision_score, zero_devision = 0),
    'recall': get_scorer('recall'),
    #'sensitivity': make_scorer(sensitivity_score),
    #'specificity': make_scorer(specificity_score),
    'mcc': get_scorer('matthews_corrcoef'),
}


# ## main analysis

# ### simulations

# In[15]:


df_anno_datasets_raw = {
    'core': df_anno.loc[df_anno.study_accession.isin(['PRJNA429097', 'PRJEB6070', 'PRJEB27928', 'PRJEB10878', 'PRJNA731589'])].copy(),
}



# In[16]:


@dataclass
class Experiment:
    name: str
    anno: str|Path
    count: str|Path
    datasets: dict[str, str|Path]
    clean_name: str = ''
    
    
    def __post_init__(self):
        if self.clean_name == '':
            self.clean_name = clean(self.name)


# In[17]:


# limit rerun to only core
experiments = [
    Experiment(name= 'count_taxa_0',                anno= anno_all,        count= taxa_counts_0,       datasets= df_anno_datasets_raw),
    Experiment(name= 'count_taxa_metaphlan4_0',     anno= anno_all,         count= taxa_meta4,          datasets= df_anno_datasets_raw),
    
    Experiment(name= 'count_taxa_func_0',           anno= anno_all,        count= taxa_func_counts_0,  datasets= df_anno_datasets_raw),
    Experiment(name= 'count_func_0',                anno= anno_all,        count= func_counts_0,           datasets= df_anno_datasets_raw),
    
    
    # Experiment(name= 'count_taxa_func_10',           anno= anno_all,        count= taxa_func_counts_10,  datasets= df_anno_datasets_raw),
    # Experiment(name= 'count_taxa_10',               anno= anno_all,        count= taxa_counts_10,       datasets= df_anno_datasets_raw),
    # Experiment(name= 'count_func_10',               anno= anno_all,        count= func_counts_10,           datasets= df_anno_datasets_raw),

    # Experiment(name= 'taxa_func',               anno= anno_all,        count= taxa_func,        datasets= df_anno_datasets_raw),
    # Experiment(name= 'taxa',                    anno= anno_all,        count= taxa,       datasets= df_anno_datasets_raw),
    # Experiment(name= 'func',                    anno= anno_all,        count= func,       datasets= df_anno_datasets_raw),

]


# In[18]:


if not is_server():
    experiments = experiments[:1]
    print("Only running minimal test ")


# In[19]:


def simulate(
    anno_df: pd.DataFrame,
    counts: pd.DataFrame,
    simulation_type: str,
    dataset_name: str,
    groups: list,
    clients: list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    label: str = "health_status",
    REPETITION: int = 10,
    N_ESTIMATOR: int = 100,
    N_JOBS: int = -1,
    CROSS_VAL: int = 5,
    feature_importance: str = 'gini',
    weighting: str = None,
    data_dir: str = './',
    scoring: dict = {'accuracy': get_scorer('accuracy')},
    exec_mode: str = 'add',
    filename: str = 'simulation_results.pkl',
    save_path: str = './',
    verbose: bool = False,
    log_level: int = logging.ERROR,
    log_file: str = "execution_time.json"
):
    """
    Unified function to run different types of simulations.
    """

    start_time = time.time()
    execution_log = {
        "simulation_type": simulation_type,
        "dataset_name": dataset_name,
        "start_time": start_time,
        "start_time_str" : datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S'),
        "steps": []
    }
    
    def log_step(step_name, start, level=logging.INFO, **kwargs):
        duration = time.time() - start
        step_log = {
            "step_name": step_name,
            "duration": duration,
            "start_time": start,
            "end_time": start + duration
        }
        step_log.update(kwargs)  # Add any additional key-value pairs
        execution_log["steps"].append(step_log)
        if log_level <= level:
            print(f"{step_name} took {str(timedelta(seconds=int(duration)))} seconds.")

    # Handle paths
    save_path = Path(save_path) / filename
    save_path_unfinished = save_path.with_suffix('.unfinished')
    log_file = save_path.parent / log_file

    # Check execution mode
    if exec_mode not in ['redo', 'load', 'add']:
        raise ValueError(f'Invalid exec_mode: {exec_mode}. Must be "redo" or "load".')

    # Initialize results
    res_out, f_imp_out = [], []

    # Load previous results if necessary
    if exec_mode == 'load':
        if save_path.exists():
            
            if log_level <= logging.INFO:
                print(f"Loading previous results for {dataset_name}.")
            return 
        else:
            print(f"No previous results found for {dataset_name}. Starting from scratch.")

    # Ensure directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Common preparations
    meta_features = ['health_status', 'gender', 'age', 'bmi'] + groups
    to_drop = [label] + groups

    if verbose:
        print("Merging annotation and count dataframes.")

    data = pd.merge(anno_df[meta_features], counts, left_index=True, right_index=True, how='inner')

    if data.empty:
        raise ValueError("Merged data is empty after dropping NA values. Please check your input data.")

    if verbose:
        print(f"Merged data contains {data.shape[0]} samples and {data.shape[1]} features.")

    # Mode-specific preparations
    present_groups = [grp[0] for grp, _df in anno_df.groupby(by=groups) if len(_df[label].unique()) >= 2 and len(_df) >= CROSS_VAL*2]
    datasets = {grp[0]: (grp_df.drop(columns=to_drop).values, grp_df[label].values) for grp, grp_df in data.groupby(groups) if grp[0] in present_groups}

    pbar_desc = f'{simulation_type} - {dataset_name}'

    if simulation_type == 'central':
        pbar_total = REPETITION * CROSS_VAL
    elif simulation_type == 'local':
        pbar_total = len(present_groups) * REPETITION
    else:  # 'combinations' or 'federated'
        if not datasets:
            raise ValueError("No valid groups found for the simulation. Please check your group criteria and input data.")
        pbar_total = sum(len(list(combinations(present_groups, n))) for n in clients) * REPETITION

    log_step(f"finish data prep", start_time, level=logging.INFO)

    # Track how many results have been written
    results_written = 0

    with tqdm(total=pbar_total, desc=pbar_desc) as pbar, open(save_path_unfinished, 'ab') as f:
        for rep in range(REPETITION):
            rep_start = time.time()
            if simulation_type == 'central':
                if verbose:
                    print(f'Starting repetition {rep + 1}/{REPETITION}')

                start = time.time()
                X = data.drop(columns=to_drop).values
                y = data[label].values

                if X.size == 0 or y.size == 0:
                    print(f"Skipping repetition {rep + 1}/{REPETITION}: No data available after preprocessing.")

                rf = RandomForestClassifier(n_estimators=N_ESTIMATOR, max_depth=MAX_DEPTH, criterion=CRITERION, min_samples_leaf=MIN_SAMPLES_LEAF, max_features=MAX_FEATURES,
                        class_weight=CLASS_WEIGHT,n_jobs=1, max_samples=MAX_SAMPLES)
                kf = StratifiedKFold(n_splits=CROSS_VAL, shuffle=True, random_state=rep)

                res, f_imp = custom_cross_validate(rf, X, y, features=data.drop(columns=to_drop).columns, cv=kf, scoring=scoring, feature_importance=feature_importance)

                meta_inf = {
                    "rep": [rep] * CROSS_VAL, "train_dataset": ['all'] * CROSS_VAL,
                    "val_dataset": ['all'] * CROSS_VAL, "analysis": [dataset_name] * CROSS_VAL,
                    "cv": list(range(CROSS_VAL)), 'n_clients': [1] * CROSS_VAL
                }

                res.update(meta_inf)
                f_imp.update(meta_inf)
                res_out.append(res)
                f_imp_out.append(f_imp)

                log_step("cross-validation", start, level=logging.DEBUG)

                pbar.update(CROSS_VAL)

                # Write only the new results one by one in append mode
                for new_res, new_f_imp in zip(res_out[results_written:], f_imp_out[results_written:]):
                    pickle.dump((new_res, new_f_imp), f)

                results_written += len(res_out[results_written:])

            elif simulation_type == "local":
                if verbose:
                    print(f'Starting local cross-validation {rep + 1}/{REPETITION}')
                for grp in present_groups:
                    start = time.time()
                    X, y = datasets[grp]
                    if X.size == 0 or y.size == 0:
                        print(f"Skipping repetition {rep + 1}/{REPETITION} for group {grp}: No data available after preprocessing.")
                        continue

                    rf = RandomForestClassifier(n_estimators=N_ESTIMATOR, max_depth=MAX_DEPTH, criterion=CRITERION, min_samples_leaf=MIN_SAMPLES_LEAF, max_features=MAX_FEATURES,
                        class_weight=CLASS_WEIGHT,n_jobs=1, max_samples=MAX_SAMPLES)
                    kf = StratifiedKFold(n_splits=CROSS_VAL, shuffle=True, random_state=rep)

                    res, f_imp = custom_cross_validate(rf, X, y, features=data.drop(columns=to_drop).columns, cv=kf, scoring=scoring, feature_importance=feature_importance)

                    meta_inf = {
                        "rep": [rep] * CROSS_VAL, "train_dataset": [grp] * CROSS_VAL,
                        "val_dataset": [grp] * CROSS_VAL, "analysis": [dataset_name] * CROSS_VAL,
                        "cv": list(range(CROSS_VAL)), 'n_clients': [1] * CROSS_VAL
                    }

                    res.update(meta_inf)
                    f_imp.update(meta_inf)
                    res_out.append(res)
                    f_imp_out.append(f_imp)

                    log_step("cross-validation", start, dataset=grp, level=logging.DEBUG)
                    pbar.update(1)

                    # Write only the new results one by one in append mode
                    for new_res, new_f_imp in zip(res_out[results_written:], f_imp_out[results_written:]):
                        pickle.dump((new_res, new_f_imp), f)

                    results_written += len(res_out[results_written:])
            else:
                for n_clients in clients:
                    client_start = time.time()
                    # if simulation_type == 'federated':
                    #     start = time.time()
                    #     estimators = {grp: RandomForestClassifier(n_estimators=N_ESTIMATOR//n_clients, max_depth=MAX_DEPTH, criterion=CRITERION, min_samples_leaf=MIN_SAMPLES_LEAF, max_features=MAX_FEATURES,
                    #                     class_weight=CLASS_WEIGHT,n_jobs=1, max_samples=MAX_SAMPLES).fit(datasets[grp][0], datasets[grp][1]) for grp in datasets}
                    #     log_step("training", start, level=logging.DEBUG)
                    for comb in combinations(present_groups, n_clients):
                        test_data = [_ for _ in present_groups if _ not in comb]
                            
                        if simulation_type == 'combinations':
                            train_data = [datasets[grp] for grp in comb if grp in datasets]
                            if len(train_data) != len(comb):
                                continue
                            X, y = np.concatenate([dat[0] for dat in train_data]), np.concatenate([dat[1] for dat in train_data])

                            if X.size == 0 or y.size == 0:
                                print(f"Skipping combination {comb}: No data available after preprocessing.")
                                continue

                            start = time.time()
                            rf = RandomForestClassifier(n_estimators=N_ESTIMATOR, max_depth=MAX_DEPTH, criterion=CRITERION, min_samples_leaf=MIN_SAMPLES_LEAF, max_features=MAX_FEATURES,
                                        class_weight=CLASS_WEIGHT,n_jobs=1, max_samples=MAX_SAMPLES).fit(X, y)
                            log_step(f"training", start, dataset=comb, level=logging.DEBUG)


                            start = time.time()
                            for grp in test_data:
                                X_test, y_test = datasets[grp]
                                res, f_imp = custom_comb_validate(rf, X, y, X_test, y_test, features=data.drop(columns=to_drop).columns.tolist(), scoring=scoring, feature_importance=feature_importance)
                                
                                meta_inf = {
                                    "rep": [rep], "train_dataset": [', '.join(map(str, comb))] ,
                                    "val_dataset": [grp], "analysis": [dataset_name],
                                    "cv": [0], 'n_clients': [n_clients]
                                }
                                
                                res.update(meta_inf)
                                f_imp.update(meta_inf)
                                res_out.append(res)
                                f_imp_out.append(f_imp)
                            log_step(f"validation", start, dataset=comb, level=logging.DEBUG)
                            
                        # elif simulation_type == 'federated':
                            
                        #     start = time.time()
                        #     if weighting == 'mcc':
                        #         # using method based on https://ieeexplore.ieee.org/document/9867984 
                        #         # combinded MCC from each tree above threshold 0.2 as weight
                        #         # removes about 80% of the trees -> worth a 2nd look
                        #         weights = {}
                        #         for grp in comb:
                        #             weights[grp] = {i: [] for i in range(len(estimators[grp].estimators_))}
                        #             for val_grp in comb:
                        #                 if grp == val_grp:
                        #                     continue
                        #                 for i, tree in enumerate(estimators[grp].estimators_):
                        #                     y_pred = tree.predict(datasets[val_grp][0])
                        #                     conf = confusion_matrix(datasets[val_grp][1], y_pred)
                        #                     weights[grp][i].append(conf)
                        #             for i in weights[grp]:
                        #                 summed_conf_matrix = np.sum(np.array(weights[grp][i]), axis=0)
                        #                 if summed_conf_matrix.ndim == 1:
                        #                     summed_conf_matrix = np.expand_dims(summed_conf_matrix, axis=0)
                        #                 mcc = mcc_from_cm(summed_conf_matrix)
                        #                 weights[grp][i] = mcc if mcc >= 0.2 else 0

                        #         combined_estimators, combined_weights = [], []
                        #         for grp in comb:
                        #             combined_estimators.extend(estimators[grp].estimators_)
                        #             combined_weights.extend(weights[grp].values())
                        #         combined_rf = VotingClassifier(estimators=combined_estimators, weights=combined_weights, voting='soft')
                        #     else:
                        #         pass
                            
                        #     log_step("weighting", start, dataset=comb, weighting=weighting, level=logging.DEBUG)
                            
                        #     start = time.time()
                        #     for grp in test_data:
                        #         X_test, y_test = datasets[grp]
                        #         res, f_imp = custom_comb_validate(combined_rf, X, y, X_test, y_test, features=data.drop(columns=to_drop).columns.tolist(), scoring=scoring, feature_importance=feature_importance)
                                
                        #         meta_inf = {
                        #             "rep": [rep] * CROSS_VAL, "train_dataset": [', '.join(map(str, comb))] * CROSS_VAL,
                        #             "val_dataset": [grp] * CROSS_VAL, "analysis": [dataset_name] * CROSS_VAL,
                        #             "cv": list(range(CROSS_VAL)), 'n_clients': [n_clients] * CROSS_VAL
                        #         }
                                
                        #         res.update(meta_inf)
                        #         f_imp.update(meta_inf)
                        #         res_out.append(res)
                        #         f_imp_out.append(f_imp)
                                
                        #     log_step(f"validation", start, dataset=comb, level=logging.DEBUG)

                        
                        #display(res_out[0])
                        #display(res)

                        # Write only the new results one by one in append mode
                        for new_res, new_f_imp in zip(res_out[results_written:], f_imp_out[results_written:]):
                            pickle.dump((new_res, new_f_imp), f)

                        results_written += len(res_out[results_written:])
                                    
                        pbar.update(1)

                    log_step(f"{n_clients} client(s)", client_start, level=logging.DEBUG)

            log_step(f"repetition", rep_start, repetition=rep + 1, level=logging.WARNING)

        # Final save
        os.rename(save_path_unfinished, save_path)
        log_step("writing final results", start, level=logging.WARNING)

    execution_log["end_time"] = time.time()
    execution_log["end_time_str"] = datetime.fromtimestamp(execution_log["end_time"]).strftime('%Y-%m-%d %H:%M:%S')
    execution_log["total_duration"] = str(timedelta(seconds=int(execution_log["end_time"] - start_time)))

    log_step("finish simulation", start_time, level=logging.INFO)
    append_to_json(log_file, execution_log)


# In[20]:


def format_pickled_results(experiment, simulation_type, save_dir=save_dir, table_dir=table_dir, prefix='', **kwargs):
    save_path = path_join(save_dir, f'{prefix}{experiment.name}_{simulation_type}_results.pkl')
    results, f_imp = [], []
    with open(save_path, 'rb') as f:
        try:
            while True:
                res_out_tmp, f_imp_out_tmp = pickle.load(f)
                results.append(res_out_tmp)
                f_imp.append(f_imp_out_tmp)
        except EOFError as e:   
            pass

    results_df = pd.DataFrame(results)
    
    if simulation_type in ['local', 'central']:
        to_explode = results_df.columns.values.tolist()
    else:
        to_explode = list(compress(results_df.columns.to_list(),
                    [type(_) == list for _ in results_df.iloc[0]]))
    results_df = results_df.explode(column=to_explode).apply(pd.to_numeric, errors='ignore')
    results_df = results_df.reset_index(drop=True)
    
    # write results to human readible files under /data/simulations/tables 
    out_dir = path.join(table_dir, f'{prefix}{simulation_type}_experiments', f'{experiment.name}.csv')
    Path(out_dir).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_dir, sep=',', index=False)
    
def load_results(experiment, simulation_type, save_dir=table_dir, prefix='', **kwargs):
    path = Path(path_join(save_dir, f'{prefix}{simulation_type}_experiments', f'{experiment.name}.csv'))
    if not path.exists():
        print(f"Could not find results for {simulation_type} - {experiment.name}. Trying to recreate them ... ")
        
        try:
            format_pickled_results(experiment, simulation_type, prefix=prefix)    
        except Exception as e:
            traceback.print_exc()
            raise FileNotFoundError(f"Recreation failed, Make sure the simulation was run successfull. {simulation_type} - {experiment.name}", e)
        
    results_df = pd.read_csv(path)
    return results_df
    


# In[21]:


def summarize_results(df, summary_df, simulation_type):
#     df = df.groupby(by=['analysis', 'train_dataset', 'val_dataset', 'n_clients']).median().reset_index()
    df['simulation_type'] = simulation_type
    summary_df = pd.concat([summary_df, df]).reset_index(drop=True)
    
    return summary_df


# In[22]:


def simulate_threaded(experiment, simulation_type, save_dir=save_dir, EXEC_MODE=EXEC_MODE, REPETITION=REPETITION, N_JOBS=N_JOBS, prefix='', **kwargs):
    anno_df = pd.read_csv(Path(data_dir) / experiment.anno, index_col=0) if isinstance(experiment.anno, (str, Path)) else experiment.anno
    counts_df = pd.read_csv(Path(data_dir) / experiment.count, index_col=0) if isinstance(experiment.count, (str, Path)) else experiment.counts

    try:
        simulate(
            anno_df=anno_df, counts=counts_df, simulation_type=simulation_type, dataset_name=experiment.name, 
            groups=groups, save_path=save_dir, filename=f'{prefix}{experiment.name}_{simulation_type}_results.pkl',
            exec_mode=EXEC_MODE, REPETITION=REPETITION, N_JOBS=N_JOBS, **kwargs
        )
    except ValueError as e:
        print(f"Skipping {experiment.name} due to error: {e}")
        traceback.print_exc()

    format_pickled_results(experiment, simulation_type, save_dir=save_dir, prefix=prefix, **kwargs)

def run_simulation_with_threading(experiments, simulation_type, max_threads=N_JOBS, threads_per_job = 6, groups=groups, **kwargs):
    num_cores = os.cpu_count() or 1
    disable_telegram = False if is_server() else True

    if max_threads is None or max_threads == -1:
        max_threads = num_cores -2  # Use all available threads
    elif max_threads == -2:
        max_threads = num_cores // 2  # Use half of the available threads
    else:
        max_threads = min(max_threads, num_cores)  # Limit to the available threads
        
    max_concurrent_jobs = max_threads // threads_per_job
    
    with ProcessPoolExecutor(max_workers=max_concurrent_jobs) as executor:
        future_to_exp = {
            executor.submit(
                partial(simulate_threaded, exp, simulation_type, N_JOBS=threads_per_job, **kwargs)
            ): exp for exp in experiments
        }
        with telegram_tqdm(
            total=len(future_to_exp), desc=f'running {simulation_type}', 
            token=telegram_token, chat_id=telegram_chatid, disable=disable_telegram
        ) as pbar:
            for future in concurrent.futures.as_completed(future_to_exp):
                exp = future_to_exp[future]
                try:
                    pbar.update(1)
                except Exception as exc:
                    print(f"Experiment {exp.name} generated an exception: {exc}")
                    traceback.print_exc()


# In[23]:


EXEC_MODE = 'load' # 'redo'
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = f"execution_time_{timestamp}.json"
verbose = False

# Run central simulations
print("Running central simulations...")
run_simulation_with_threading(experiments, scoring=scoring, 
    simulation_type='central', feature_importance='', verbose=verbose, EXEC_MODE=EXEC_MODE, log_file=log_file)


print("\n-----------------------------\n")
print("Running local simulations...")
# Run local simulations
run_simulation_with_threading(experiments, scoring=scoring, 
    simulation_type='local', feature_importance='', verbose=verbose, EXEC_MODE=EXEC_MODE, log_file=log_file)

print("\n-----------------------------\n")
print("Running combinations simulations...")
# Run combinations simulations
run_simulation_with_threading(experiments, scoring=scoring, 
    simulation_type='combinations', feature_importance='', verbose=verbose, EXEC_MODE=EXEC_MODE, log_file=log_file)

# # no fed simulation for now
# print("\n-----------------------------\n")
# print("Running federated simulations...")
# # Run federated simulations
# results_federated, features_federated = run_simulation_with_threading(experiments, simulation_type='federated', feature_importance='', verbose=True, EXEC_MODE=EXEC_MODE)

print("\n-----------------------------\n")


# ### plot

# In[24]:


def plot_performance_box(_df: pd.DataFrame, _central_df:pd.DataFrame|None = None, _local_df:pd.DataFrame|None = None,  metric="test_roc_auc", 
                         title:str = "federated analysis", save_path:str="./data/simulations/figures/", no_output:bool=False, palette='tab10', **kwargs):
    target_path = Path.joinpath(Path(save_dir), 'figures', 'box', metric)
    target_path.mkdir(exist_ok=True, parents=True)  
    
    vmin, vmax = (0.0, 1.0) if metric != 'test_mcc' else (-0.5, 1.0)
        
    _df = _df.loc[_df['n_clients'] < 11]
    
    for analysis, a_df in _df.groupby(by="analysis"):
        # a_df = a_df.apply(pd.to_numeric, errors='ignore')
        #display(a_df.groupby(by=["val_dataset", 'n_clients']).median(numeric_only=True).T)
        #if not no_output: display(a_df.groupby(by=['n_clients']).median(numeric_only=True).T)

        plt.figure(figsize=(16,8))
        p = sns.boxplot(data=a_df, x='n_clients', y=metric, hue='val_dataset', fliersize=1, palette=palette, **kwargs)
        
        if _central_df is not None:
            plt.axhline(_central_df[metric].median(), color='r', linestyle='--')
        
        plt.ylim((vmin, vmax))
        plt.legend(title='Validation Cohorts', bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=6)
        plt.suptitle(f'{clean(analysis)} - {title}')
        plt.xlabel("Number of Cohorts")
        plt.ylabel(clean(metric))
        
        plt.savefig(Path(target_path, f'{analysis}_{title.replace(" ", "_")}.png'), bbox_inches='tight', dpi=300)
        if no_output:
            plt.close()
        else:
            plt.show()


# In[25]:


def plot_performance_line_additive_cohort(_df: pd.DataFrame, _central_df:pd.DataFrame|None = None, _local_df:pd.DataFrame|None = None,  metric="test_roc_auc",
                                         title:str = "federated analysis", save_path:str="./data/simulations/figures/", no_output:bool=False, palette='tab10', **kwargs):
    target_path = Path.joinpath(Path(save_dir), 'figures', 'line_additive', metric)
    target_path.mkdir(exist_ok=True, parents=True)  
    vmin, vmax = (0.0, 1.0) if metric != 'test_mcc' else (-0.5, 1.0)
    for analysis, base_df in _df.groupby(by="analysis"):
        unique_cohorts = base_df.loc[base_df['n_clients'] == 1, 'train_dataset'].unique()   
        for c in unique_cohorts:
            a_df = base_df.apply(pd.to_numeric, errors='ignore')
            a_df = a_df.loc[(a_df['train_dataset'].str.contains(c, regex=False)) & (a_df['n_clients'] <=2)]
            
            # Ensure train_dataset is treated as a categorical variable
            a_df['train_dataset'] = a_df['train_dataset'].astype('category')
            
            median_mcc = a_df.groupby('train_dataset')[metric].median().sort_values()
            ordered_train_dataset = [c] + [x for x in median_mcc.index if x != c]
            
            plt.figure(figsize=(12,8))
            p = sns.pointplot(data=a_df, x='train_dataset', y=metric, hue='val_dataset', palette=palette, linestyles='', 
                              order=ordered_train_dataset, errorbar=None, dodge=False, markers='o', **kwargs)
            
            plt.ylim((vmin, vmax))
            plt.legend(bbox_to_anchor=(1.02, 1), loc=2,)
            plt.title(f'{clean(analysis)} {c} - {title}')
            plt.xlabel(f"cohort combinations")
            plt.ylabel(clean(metric))
            
            # Format x-tick labels
            labels = [_.get_text() for _ in p.get_xticklabels()]
            
            new_labels = [c]
            for _ in labels:
                if _ == c:
                    continue
                
                _ = _.replace(c, '').replace(', ', '')    
                _ = f'{c}\n+ {_}'
                new_labels.append(_)
            p.set_xticklabels(new_labels)
            
            plt.savefig(Path(target_path, f'{analysis}_{c}_{title.replace(" ", "_")}.png'), bbox_inches='tight')
            if no_output:
                plt.close()
            else:
                plt.show()
            


# In[26]:


def plot_performance_box_additive_cohort(_df: pd.DataFrame, _central_df:pd.DataFrame|None = None, _local_df:pd.DataFrame|None = None,  metric="test_roc_auc", 
                                         title:str = "federated analysis", save_path:str="./data/simulations/figures/", no_output:bool=False, palette='tab10', **kwargs):
    target_path = Path.joinpath(Path(save_dir), 'figures', 'box_additive', metric)
    target_path.mkdir(exist_ok=True, parents=True)  
    vmin, vmax = (0.0, 1.0) if metric != 'test_mcc' else (-0.5, 1.0)
    for analysis, base_df in _df.groupby(by="analysis"):
        unique_cohorts = base_df.loc[base_df['n_clients'] == 1, 'train_dataset'].unique()   
        for c in unique_cohorts:
            a_df = base_df.apply(pd.to_numeric, errors='ignore')
            a_df = a_df.loc[(a_df['train_dataset'].str.contains(c, regex=False)) & (a_df['n_clients'] <=2)]
            
            # Ensure train_dataset is treated as a categorical variable
            a_df['train_dataset'] = a_df['train_dataset'].astype('category')
            
            median_mcc = a_df.groupby('train_dataset')[metric].median().sort_values()
            ordered_train_dataset = [c] + [x for x in median_mcc.index if x != c]
            
            plt.figure(figsize=(12,8))
            sns.boxplot(data=a_df, x='train_dataset', y=metric, color='lightgrey', fill=False, 
                              order=ordered_train_dataset, **kwargs)
            p = sns.stripplot(data=a_df, x='train_dataset', y=metric, hue="val_dataset", palette=palette, 
                              order=ordered_train_dataset, **kwargs)
            plt.ylim((vmin, vmax))
            plt.legend(bbox_to_anchor=(1.02, 1), loc=2)
            plt.title(f'{clean(analysis)} {c} - {title}')
            plt.xlabel(f"cohort combinations")
            plt.ylabel(clean(metric))
            
            # Format x-tick labels
            labels = [_.get_text() for _ in p.get_xticklabels()]
            
            new_labels = [c]
            for _ in labels:
                if _ == c:
                    continue
                
                _ = _.replace(c, '').replace(', ', '')    
                _ = f'{c}\n+ {_}'
                new_labels.append(_)
            p.set_xticklabels(new_labels)
            
            plt.savefig(Path(target_path, f'{analysis}_{c}_{title.replace(" ", "_")}.png'), bbox_inches='tight')
            if no_output:
                plt.close()
            else:
                plt.show()
            


# In[27]:


def plot_performance_multi_box_additive_cohort(_df: pd.DataFrame, _central_df:pd.DataFrame|None = None, _local_df:pd.DataFrame|None = None, cohorts=["Brazil1", "China1"],  metric="test_roc_auc", 
                                         title:str = "federated analysis", save_path:str="./data/simulations/figures/", no_output:bool=False, palette='tab10', **kwargs):
    target_path = Path.joinpath(Path(save_dir), 'figures', 'box_additive_multi', metric)
    target_path.mkdir(exist_ok=True, parents=True)  
    vmin, vmax = (0.0, 1.0) if metric != 'test_mcc' else (-0.5, 1.0)
    _df = _df.apply(pd.to_numeric, errors='ignore')
    _df = _df.loc[_df['n_clients'] <=2]
    unique_cohorts = _df.loc[_df['n_clients'] == 1, 'train_dataset'].unique()   
    analysis = _df.iloc[0]["analysis"]
    
    row, col = 1, len(cohorts)
    gs = GridSpec(row, col)
    
    fig = plt.figure(figsize=(12, 6))
    axes = [fig.add_subplot(gs[i, j]) for i in range(row) for j in range(col)]
    
    for i, c in enumerate(cohorts):
        a_df = _df.loc[_df['train_dataset'].str.contains(c, regex=False)]
        
        # Ensure train_dataset is treated as a categorical variable
        a_df['train_dataset'] = a_df['train_dataset'].astype('category')
        
        median_mcc = a_df.groupby('train_dataset')[metric].median().sort_values()
        ordered_train_dataset = [c] + [x for x in median_mcc.index if x != c]
        
        sns.boxplot(data=a_df, x='train_dataset', y=metric, color='lightgrey', fill=False, 
                        order=ordered_train_dataset, ax=axes[i], **kwargs)
        sns.stripplot(data=a_df, x='train_dataset', y=metric, hue="val_dataset", palette=palette, 
                        order=ordered_train_dataset, ax=axes[i], **kwargs)
        
        axes[i].set_ylim((vmin, vmax))
        axes[i].set_title(f'{c} vs. {c} + 1')
        #axes[i].set_xlabel(f"cohort combinations")
        axes[i].set_ylabel(clean(metric))
        axes[i].legend().remove()
        
        # Format x-tick labels
        labels = [_.get_text() for _ in axes[i].get_xticklabels()]
        
        new_labels = [c]
        for _ in labels:
            if _ == c:
                continue
            
            _ = _.replace(c, '').replace(', ', '')    
            _ = f'+ {_}'
            new_labels.append(_)
        axes[i].set_xticklabels(new_labels, rotation=-90)
        
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Validation Cohorts', bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=6)
    
    fig.suptitle(f'{title}')
    
    for ax, letter in zip([*axes], ['A', 'B', 'C', 'D']):
        ax.text(-0.13, 1.05, letter, transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='center')
    
    
    plt.savefig(Path(target_path, f'{analysis}_{c}_{title.replace(" ", "_")}.png'), bbox_inches='tight')
    if no_output:
        plt.close()
    else:
        plt.show()
    


# In[28]:


def plot_performance_violin_additive_cohort(_df: pd.DataFrame, _central_df:pd.DataFrame|None = None, _local_df:pd.DataFrame|None = None,  metric="test_roc_auc", 
                                         title:str = "federated analysis", save_path:str="./data/simulations/figures/", no_output:bool=False, palette='tab10', **kwargs):
    target_path = Path.joinpath(Path(save_dir), 'figures', 'violin_additive', metric)
    target_path.mkdir(exist_ok=True, parents=True)  
    vmin, vmax = (0.0, 1.0) if metric != 'test_mcc' else (-0.5, 1.0)
    for analysis, base_df in _df.groupby(by="analysis"):
        unique_cohorts = base_df.loc[base_df['n_clients'] == 1, 'train_dataset'].unique()   
        for c in unique_cohorts:
            a_df = base_df.apply(pd.to_numeric, errors='ignore')
            a_df = a_df.loc[(a_df['train_dataset'].str.contains(c, regex=False)) & (a_df['n_clients'] <=2)]
            
            # Ensure train_dataset is treated as a categorical variable
            a_df['train_dataset'] = a_df['train_dataset'].astype('category')
            
            median_mcc = a_df.groupby('train_dataset')[metric].median().sort_values()
            ordered_train_dataset = [c] + [x for x in median_mcc.index if x != c]
            
            plt.figure(figsize=(12,8))
            
            sns.violinplot(data=a_df, x='train_dataset', y=metric, color="lightgrey", fill=False, inner=None)
            p = sns.stripplot(data=a_df, x='train_dataset', y=metric, hue="val_dataset", palette=palette, 
                              order=ordered_train_dataset, **kwargs)
            
            plt.ylim((vmin, vmax))
            plt.legend(bbox_to_anchor=(1.02, 1), loc=2,)
            plt.title(f'{clean(analysis)} {c} - {title}')
            plt.xlabel(f"cohort combinations")
            plt.ylabel(clean(metric))
            
            # Format x-tick labels
            labels = [_.get_text() for _ in p.get_xticklabels()]
            
            new_labels = [c]
            for _ in labels:
                if _ == c:
                    continue
                
                _ = _.replace(c, '').replace(', ', '')    
                _ = f'{c}\n+ {_}'
                new_labels.append(_)
            p.set_xticklabels(new_labels)
            
            plt.savefig(Path(target_path, f'{analysis}_{c}_{title.replace(" ", "_")}.png'), bbox_inches='tight')
            if no_output:
                plt.close()
            else:
                plt.show()
            


# In[29]:


def plot_double_linebox(_df: pd.DataFrame, _central_df: pd.DataFrame, _local_df: pd.DataFrame, _anno_df: None|pd.DataFrame = None, metric1: str = "test_mcc", metric2: str = "test_f1", 
                        title: str = "double trouble", save_path: str = "./data/simulations/", no_output: bool = False, as_line = False, palette = 'Accent', **kwargs):
    target_path = Path(save_path) / 'figures' / 'double_linebox' / "_".join([metric1, metric2])
    target_path.mkdir(exist_ok=True, parents=True)  
    
    vmin1, vmax1 = (-0.5, 1.0) 
    vmin2, vmax2 = (0.0, 1.0) 
    errorbar = kwargs.pop('errorbar', 'sd')    

    # remove highest number of clients
    _df =  _df.loc[_df['n_clients'] < _df['n_clients'].max()]
    
    fig = plt.figure(figsize=(12, 8))
    
    row, col = 2, 4
    gs = GridSpec(row, col, width_ratios=[1, 3, 3, 4])
    
    axes = [fig.add_subplot(gs[i, j]) for i in range(row) for j in range(col)]
    
    # Ensure we have the correct number of axes
    assert len(axes) == row * col, f"Expected {row * col} axes, but got {len(axes)}"
    
    # Plot for metric1
    sns.boxplot(data=_central_df, y=metric1, ax=axes[0], color='r', **kwargs)
    axes[0].set_title('Central\nModel')
    axes[0].set_ylabel(clean(metric1), size='large')
    axes[0].set_xlabel("All Cohorts")
    axes[0].set_xticks([])
    axes[0].set_ylim((vmin1, vmax1))
    
    sns.boxplot(data=_local_df.loc[_local_df['n_clients'] == 1], x='n_clients', y=metric1, hue='train_dataset', palette=palette, ax=axes[1], **kwargs)
    
    axes[1].axhline(_central_df[metric1].median(), color='r', linestyle='--')
    axes[1].set_xticks([])
    axes[1].set_ylim((vmin1, vmax1))
    axes[1].set_xlabel("Individual Cohorts\n(intra-cohort)")
    axes[1].set_ylabel("")
    axes[1].set_title('Local Models\n')
    axes[1].legend().remove()
    
    sns.boxplot(data=_df.loc[_df['n_clients'] == 1], x='n_clients', y=metric1, hue='train_dataset', palette=palette, ax=axes[2], **kwargs)
    
    axes[2].axhline(_central_df[metric1].median(), color='r', linestyle='--')
    axes[2].set_xticks([])
    axes[2].set_ylim((vmin1, vmax1))
    axes[2].set_xlabel("Individual Cohorts\n(inter-cohort)")
    axes[2].set_ylabel("")
    axes[2].set_title('Combined Models n=1\n(by training cohort)')
    axes[2].legend().remove()
    
    
    sns.pointplot(data=_df.loc[_df['n_clients'] > 0], x='n_clients', y=metric1, hue='val_dataset', 
                  palette=palette, dodge=True, ax=axes[3], marker=".", markersize=10, markeredgewidth=3,
                  err_kws={'linewidth': 0.5}, errorbar=errorbar, **kwargs)

    axes[3].axhline(_central_df[metric1].median(), color='r', linestyle='--')
    axes[3].set_ylim((vmin1, vmax1))
    axes[3].set_ylabel("")
    axes[3].set_xlabel("Number of Cohorts")
    axes[3].set_title("Combined Models\n(by validation cohort)")
    axes[3].legend().remove()
    
    handles, labels = axes[2].get_legend_handles_labels()
    
    # Plot for metric2
    sns.boxplot(data=_central_df, y=metric2, ax=axes[4], color='r', **kwargs)
    axes[4].set_ylabel(clean(metric2), size='large')
    axes[4].set_xlabel("All Cohorts")
    axes[4].set_ylim((vmin2, vmax2))
    axes[4].set_xticks([])
    
    sns.boxplot(data=_local_df.loc[_local_df['n_clients'] == 1], x='n_clients', y=metric2, hue='train_dataset', palette=palette, ax=axes[5], **kwargs)
    
    axes[5].axhline(_central_df[metric2].median(), color='r', linestyle='--')
    axes[5].set_xticks([])
    axes[5].set_ylim((vmin2, vmax2))
    axes[5].set_xlabel("Individual Cohorts\n(intra-cohort)")
    axes[5].set_ylabel("")
    axes[5].legend().remove()
    
    sns.boxplot(data=_df.loc[_df['n_clients'] == 1], x='n_clients', y=metric2, hue='train_dataset', palette=palette, ax=axes[6], **kwargs)
    
    axes[6].axhline(_central_df[metric2].median(), color='r', linestyle='--')
    axes[6].set_xticks([])
    axes[6].set_ylim((vmin2, vmax2))
    axes[6].set_xlabel("Individual Cohorts\n(inter-cohort)")
    axes[6].set_ylabel("")
    axes[6].legend().remove()
    
    sns.pointplot(data=_df.loc[_df['n_clients'] > 0], x='n_clients', y=metric2, hue='val_dataset', 
                  palette=palette, dodge=True, ax=axes[7], marker=".", markersize=10, markeredgewidth=3,
                  err_kws={'linewidth': 0.5}, errorbar=errorbar, **kwargs)
    
    axes[7].axhline(_central_df[metric2].median(), color='r', linestyle='--')
    axes[7].set_ylim((vmin2, vmax2))
    axes[7].set_ylabel("")
    axes[7].set_xlabel("Number of Cohorts")
    axes[7].legend().remove()
    
    fig.legend(handles, labels, title='Validation Cohorts', bbox_to_anchor=(0.5, 0.02), loc='upper center', ncol=6)
    
    fig.suptitle(f'{title}')
    
    for ax, letter in zip([axes[0], axes[1], axes[2], axes[3]], ['A', 'B', 'C', 'D']):
        ax.text(-0.1, 1.15, letter, transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='center')
    
    fig.tight_layout()
    
    fig.savefig(Path(target_path, f'{title.replace(" ", "_")}.png'), bbox_inches='tight')
    
    if _anno_df is not None:
        fig.legends.clear()
        
        new_labels = []
        for label in labels:
            n_samples = _anno_df.loc[_anno_df['cohort_name'] == label].__len__()
            new_labels.append(f"{label} ({n_samples})")
        fig.legend(handles, new_labels, title='Validation Cohorts', bbox_to_anchor=(0.5, 0.02), loc='upper center', ncol=6)
        
        fig.tight_layout()
        fig.savefig(Path(target_path, f'{title.replace(" ", "_")}_nsamples.png'), bbox_inches='tight')
    
    if no_output:
        plt.close(fig)
    else:
        plt.show()


# In[30]:


def plot_performance_linebox(_df: pd.DataFrame, _central_df:pd.DataFrame, _local_df:pd.DataFrame, metric:str="test_roc_auc", 
                             title:str = "federated analysis", save_path:str="./data/simulations/figures/", no_output:bool=False, as_line=False, palette='Accent', **kwargs):
    target_path = Path.joinpath(Path(save_dir), 'figures', 'linebox', metric)
    target_path.mkdir(exist_ok=True, parents=True)  
    
    vmin, vmax = (0.0, 1.0) if metric != 'test_mcc' else (-0.5, 1.0)
    errorbar = kwargs.pop('errorbar', 'sd')    
    for analysis, a_df in _df.groupby(by="analysis"):
        # a_df = a_df.apply(pd.to_numeric, errors='ignore')
        #display(a_df.groupby(by=["val_dataset", 'n_clients']).median(numeric_only=True).T)
        #if not no_output: display(a_df.groupby(by=['n_clients']).median(numeric_only=True).T)

        # remove highest number of clients
        a_df =  a_df.loc[a_df['n_clients'] < a_df['n_clients'].max()]
        
        #plt.figure(figsize=[12,8])
        fig = plt.figure(figsize=(12,8))
        gs = GridSpec(1, 4, width_ratios=[1, 3, 3, 4])
        axes = [fig.add_subplot(gs[i]) for i in range(4)]
        
        if _central_df is not None:
            sns.boxplot(data=_central_df, y=metric, ax=axes[0], color='r', **kwargs)
            axes[0].set_title('Central\nModel')
            axes[0].set_ylabel(clean(metric))
            axes[0].set_xlabel("All Cohorts")
            axes[0].set_xticks([])
            axes[0].set_ylim((vmin, vmax))
        
        sns.boxplot(data=_local_df.loc[_local_df['n_clients'] == 1], x='n_clients', y=metric, hue='train_dataset', palette=palette, ax=axes[1], **kwargs)
        if _central_df is not None:
            axes[1].axhline(_central_df[metric].median(), color='r', linestyle='--')
        axes[1].set_xticks([])
        axes[1].set_ylim((vmin, vmax))
        axes[1].set_xlabel("Individual Cohorts")
        axes[1].set_ylabel("")
        axes[1].set_title('Local Models\n')
        axes[1].legend().remove()
        
        sns.boxplot(data=_df.loc[_df['n_clients'] == 1], x='n_clients', y=metric, hue='train_dataset', palette=palette, ax=axes[2], **kwargs)
        if _central_df is not None:
            axes[2].axhline(_central_df[metric].median(), color='r', linestyle='--')
        axes[2].set_xticks([])
        axes[2].set_ylim((vmin, vmax))
        axes[2].set_xlabel("Individual Cohorts")
        axes[2].set_ylabel("")
        axes[2].set_title('Combined Models n=1\n(by training cohort)')
        axes[2].legend().remove()
        
        
        sns.pointplot(data=_df.loc[_df['n_clients'] > 0], x='n_clients', y=metric, hue='val_dataset', 
                    palette=palette, dodge=True, ax=axes[3], marker=".", markersize=10, markeredgewidth=3,
                    err_kws={'linewidth': 0.5}, errorbar=errorbar, **kwargs)
        if _central_df is not None:
            axes[3].axhline(_central_df[metric].median(), color='r', linestyle='--')
        axes[3].set_ylim((vmin, vmax))
        axes[3].set_ylabel("")
        axes[3].set_xlabel("Number of Cohorts")
        axes[3].set_title("Combined Models\n(by validation cohort)")
        axes[3].legend().remove()
        
        handles, labels = axes[3].get_legend_handles_labels()
    
        plt.legend().remove()
        #ax2.set_xticks([int(_) for _ in range(1, len(a_df['n_clients'].unique()))])
        
        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2,)
        handles, labels = plt.gca().get_legend_handles_labels()
        fig.legend(handles, labels, title='Validation Cohorts', bbox_to_anchor=(0.5, 0.02), loc='upper center', ncol=6)

            
        plt.title("Performance of all Evaluation Steps\nby validation cohort")
        fig.suptitle(f'{clean(analysis)} - {title}')
        
        plt.savefig(Path(target_path, f'{analysis}_{title.replace(" ", "_")}.png'), bbox_inches='tight')
        if no_output:
            plt.close()
        else:
            plt.show()


# In[31]:


def plot_performance_by_train_box(_df: pd.DataFrame, _central_df:pd.DataFrame|None = None, _local_df:pd.DataFrame|None = None,  metric="test_roc_auc", 
                                  title:str = "federated analysis", save_path:str="./data/simulations/figures/", no_output:bool=False, palette='Accent', **kwargs):
    
    target_path = Path.joinpath(Path(save_dir), 'figures', 'box_train', metric)
    target_path.mkdir(exist_ok=True, parents=True)    
    vmin, vmax = (0.0, 1.0) if metric != 'test_mcc' else (-0.5, 1.0)
    if _local_df is not None:
        _local_df.loc[:, 'n_clients'] = 0
        _df = pd.concat([_df, _local_df]).reset_index(drop=True)
        #display(_df.n_clients.value_counts())
    for analysis, a_df in _df.groupby(by="analysis"):

        #plt.figure(figsize=[12,8])
        p = sns.boxplot(data=a_df, x='n_clients', y=metric, hue='train_dataset', palette=palette, **kwargs)
        
        if _central_df is not None:
            plt.axhline(_central_df[metric].median(), color='r', linestyle='--')
                    
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=UserWarning)
            ticklabels = []
            if _local_df is not None:
                ticklabels.append('local')
                
            ticklabels.extend([str(_) for _ in range(1, len(a_df['n_clients'].unique()))])
            p.set_xticklabels(ticklabels)
        
        plt.ylim((vmin, vmax))
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2,)
        plt.title(f'{clean(analysis)} - {title}')
        plt.xlabel("number of training clients")
        plt.ylabel(clean(metric))
        
        plt.savefig(Path(target_path, f'{analysis}_{title.replace(" ", "_")}.png'), bbox_inches='tight')
        if no_output:
            plt.close()
        else:
            plt.show()


# In[32]:


def plot_box_local(local_df, metric="test_roc_auc", title="baseline", no_output=False, save_dir=save_dir, palette='Accent', **kwargs):
    target_path = Path.joinpath(Path(save_dir), 'figures', 'local', metric)
    target_path.mkdir(exist_ok=True, parents=True)    

    vmin, vmax = (0.0, 1.0) if metric != 'test_mcc' else (-0.5, 1.0)
    
    for analysis, a_df in local_df.groupby(by="analysis"):
        # a_df = a_df.apply(pd.to_numeric, errors='ignore')
        #display(a_df.groupby(by=["val_dataset", 'n_clients']).median(numeric_only=True).T)
        #if not no_output: display(a_df.groupby(by=['n_clients']).median(numeric_only=True).T)

        #plt.figure(figsize=[12,8])
        p = sns.boxplot(data=a_df, x='val_dataset', y=metric, hue='val_dataset', palette=palette, **kwargs)
        
        plt.ylim((vmin, vmax))
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2,)
        plt.title(f'{clean(analysis)} - {title} ')
        plt.xlabel("cohort")
        plt.ylabel(clean(metric))
        plt.xticks(rotation=45, ha='right') 
        plt.savefig(Path(target_path, f'{analysis}_{metric}_baseline.png'), bbox_inches='tight')
        
        if no_output:
            plt.close()
        else:
            plt.show()


# In[33]:


def plot_performance_heatmap(comb_df, local_df, show_core=False, metric='test_roc_auc', title_prefix:str = '', no_output=False, 
                             save_dir = save_dir, max_clients=1, palette='vlag', **kwargs):

    target_path = Path.joinpath(Path(save_dir), 'figures', 'heatmaps', metric)
    target_path.mkdir(exist_ok=True, parents=True)    
    
    if local_df is not None:
        _df = pd.concat([comb_df, local_df]).reset_index(drop=True)
    else:
        _df = comb_df
    
    for analysis, _df in _df.groupby(by='analysis'):
        
        save_title = f'median_{title_prefix}{"_" if title_prefix else ""}{metric}_{analysis}_{max_clients}clients'
        
        if max_clients > 1:
            _df = _df.loc[_df['n_clients'] == max_clients]
        else:
            _df = _df.loc[_df['n_clients'] <=max_clients]
            
        if not show_core:
            _df = _df.loc[(_df['val_dataset'] != 'core') & (_df['train_dataset'] != 'core')]
        
        _df = generate_heatmap_tables(_df, metric=metric, title=save_title, save_dir=save_dir)
        if metric in ['test_mcc']:
            vmin, vmax = -1.0, 1.0
        else:
            vmin, vmax = 0.0, 1.0
            
        square = True if max_clients == 1 else False
        
        match max_clients:
            case 1: figsize, fraction = (8, 8),  0.15
            case 2: figsize, fraction = (8, 10), 0.1
            case 3: figsize, fraction = (8, 36), 0.03
            case 4: figsize, fraction = (8, 56), 0.03
            case 5: figsize, fraction = (8, 112), 0.03
            case 6: figsize, fraction = (10, 180), 0.03
            case 7: figsize, fraction = (10, 112), 0.03
            case 8: figsize, fraction = (10, 56), 0.03
            case 9: figsize, fraction = (12, 36), 0.03
            case 10: figsize, fraction = (12, 10), 0.1
            case 11: figsize, fraction = (14,8), 0.15
            case _: figsize, fraction = (8, 40), 0.01
            
        plt.figure(figsize=figsize)     
        ax = sns.heatmap(data=_df, annot=True, fmt='.2f', annot_kws={'size':8}, cbar_kws={'shrink': 0.5, 'fraction':fraction}, vmin=vmin, vmax=vmax, cmap=palette, square=square, **kwargs)
        title = f'{title_prefix}{clean(analysis)}: mean {metric}{f" {max_clients} train cohorts" if max_clients > 1 else ""}'
        if max_clients > 1:
            ax.set_yticks([x + 0.5 for x in range(_df.shape[0])])
            ax.set_yticklabels(_df.index, ha='right', size=6)
            #ax.tick_params(axis='y', which='both', pad=max([len(_) for _ in _df.index.tolist()])*3.5)
        else:
            for i in range(min(_df.shape)):
                ax.add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor='yellow', lw=3))

        plt.title(title)
        plt.ylabel("training dataset")
        plt.xlabel("validation dataset")
        #plt.xticks(rotation=30, ha='right')
        plt.xticks()
        plt.yticks()
        
        plt.tight_layout() 
        
        # Dynamically calculate the required space for y-ticklabels
        renderer = plt.gcf().canvas.get_renderer()
        ytick_labels = ax.get_yticklabels()
        max_label_width = max([label.get_window_extent(renderer=renderer).width for label in ytick_labels])

        # Convert label width from display units to figure units
        label_width_inch = max_label_width / plt.gcf().dpi
        current_fig_width = plt.gcf().get_size_inches()[0]
        left_margin = label_width_inch / current_fig_width + 0.05  # Adding a small padding

        # Ensure the left margin has enough space for y-tick labels
        plt.subplots_adjust(left=left_margin)

        
        plt.savefig(Path.joinpath(target_path, f'{save_title}.png'), bbox_inches='tight')

        if no_output:
            plt.close()
        else:    
            plt.show()
            


# In[34]:


def plot_performance_heatmap_additive_cohort(comb_df, local_df, show_core=False, metric='test_roc_auc', title: str|None = None, no_output=False, 
                                             save_dir = save_dir, max_clients=1, palette='vlag', **kwargs):
    gen_title = True if title is None else False
    target_path = Path.joinpath(Path(save_dir), 'figures', 'heatmaps_additive', metric)
    target_path.mkdir(exist_ok=True, parents=True)    
    df = comb_df
    df = df.loc[df['n_clients'] <= max_clients]
    
    for analysis, _df in df.groupby(by='analysis'):
            
        unique_cohorts = _df.loc[_df['n_clients'] == 1, 'train_dataset'].unique()   
        for c in unique_cohorts:
            c_df = _df.loc[_df['train_dataset'].str.contains(c, regex=False)]
            
            c_df = c_df.loc[c_df['n_clients'] <=max_clients]
            if not show_core:
                c_df = c_df.loc[(c_df['val_dataset'] != 'core') & (c_df['train_dataset'] != 'core')]
            c_df = c_df.loc[:, ['train_dataset', 'val_dataset', metric]]
            c_df = c_df.groupby(by=['train_dataset', 'val_dataset']).median().reset_index()
            c_df = c_df.pivot(index='train_dataset', columns='val_dataset', values=metric)
            c_df = c_df.astype(np.float32)
            
            # Sort rows so that the row with train_dataset=c is the first row
            c_df = c_df.reindex([c] + [x for x in c_df.index if x != c])
            if c_df.empty:
                print(f"No data available for analysis {analysis} and cohort {c}. Skipping plot.")
                continue   
            
            if metric in ['test_mcc']:
                vmin, vmax = -1.0, 1.0
            else:
                vmin, vmax = 0.0, 1.0
            
            try:
                plt.figure(figsize=(8,8))
                
                sns.heatmap(data=c_df, annot=True, fmt='.2f', annot_kws={'size':8}, square=True, vmin=vmin, vmax=vmax, cmap=palette)
                if gen_title:
                    title = f'{clean(analysis)} {c}: mean {metric}{f" {max_clients} train cohorts" if max_clients > 1 else ""}'


                plt.title(title)
                plt.ylabel("training dataset")
                plt.xlabel("validation dataset")
                
                plt.tight_layout()  
            
                plt.savefig(Path.joinpath(target_path, f'mean_{metric}_{analysis}_{c}.png'))
                if no_output:
                    plt.close('all')
                else:    
                    plt.show()
                
            except Exception as e:
                print(f"Failed to plot for analysis {analysis} and cohort {c}: {e}")
                traceback.print_exc()
            finally:
                plt.close('all')


# In[35]:


def plot_significance_facetgrid(_df, group='val_dataset', metric='test_mcc', title=None, cohort_order=None, no_output=False, save_dir: str|Path = save_dir, palette='Accent'):
    if title is None:
        title = f"Significance of {clean(metric)} change with increased number of cohorts - Welch's t-test"
    target_path = Path(save_dir) / 'figures' / 'significance_grid' / metric
    target_path.mkdir(exist_ok=True, parents=True)
    csv_output_path = Path(table_dir) / 'significance_analysis' 
    csv_output_path.mkdir(exist_ok=True, parents=True)

    gen_order = True if cohort_order is None else False
    if isinstance(palette, list):
        palette = palette[1:] + palette[:1]

    results = []  # To store t-test results

    for analysis, a_df in _df.groupby(by="analysis"):
        if gen_order:
            cohort_order = sorted(a_df[group].unique())
        plt.figure(figsize=(12, 8))

        g = sns.FacetGrid(a_df, col=group, col_wrap=4, height=2, aspect=1.5, col_order=cohort_order, hue_order=cohort_order,
                          sharex=False, sharey=False, hue=group, palette=palette)
        g.map_dataframe(sns.boxplot, x='n_clients', y=metric, fliersize=1)
        g.set(ylim=(-0.5, 1.0))

        def annotate_significance_direct(data, ax, x='n_clients', y=metric):
            clients = sorted(data[x].unique())

            for i, (c1, c2) in enumerate(zip(clients[:-1], clients[1:])):
                data1 = data[data[x] == c1][y]
                data2 = data[data[x] == c2][y]

                t_stat, p_val = ttest_ind(data1, data2, nan_policy='omit', equal_var=False)
                dof = len(data1) + len(data2) - 2

                conf_interval = (
                    (data1.mean() - data2.mean()) - 1.96 * (data1.std() / len(data1)**0.5 + data2.std() / len(data2)**0.5),
                    (data1.mean() - data2.mean()) + 1.96 * (data1.std() / len(data1)**0.5 + data2.std() / len(data2)**0.5)
                )

                corrected_results = multipletests([p_val], alpha=0.05, method='bonferroni')
                adjusted_pvals = corrected_results[1][0]

                annotation = '***' if adjusted_pvals < 0.001 else '**' if adjusted_pvals < 0.01 else '*' if adjusted_pvals < 0.05 else 'ns'
                y_max = max(data1.max(), data2.max()) + 0.08

                ax.plot([i+0.1, i + 1 - 0.1], [y_max, y_max], color='black')
                ax.plot([i+0.1, i+0.1], [y_max - 0.02, y_max], color='black')
                ax.plot([i + 1 - 0.1, i + 1 - 0.1], [y_max - 0.02, y_max], color='black')
                ax.text((i + 0.5), y_max + 0.03, annotation, ha='center', color='black', size='small')

                results.append({
                    'Analysis': analysis,
                    'Group_1': c1,
                    'Group_2': c2,
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'Adjusted_p_value': adjusted_pvals,
                    'Degrees_of_Freedom': dof,
                    '95%_Confidence_Interval': conf_interval
                })

        for ax, val_dataset in zip(g.axes.flat, cohort_order):
            subset = a_df[a_df[group] == val_dataset]
            if not subset.empty:
                annotate_significance_direct(subset, ax)

        g.set_titles(col_template="{col_name}")

        legend_elements = [
            Line2D([0], [0], color='black', lw=0, label='Significance:'),
            Line2D([0], [0], color='black', lw=0, label='*** < 0.001'),
            Line2D([0], [0], color='black', lw=0, label='** < 0.01'),
            Line2D([0], [0], color='black', lw=0, label='* < 0.05'),
            Line2D([0], [0], color='black', lw=0, label='ns > 0.05')
        ]

        g.fig.legend(handles=legend_elements, loc='upper left', framealpha=0, bbox_to_anchor=(1, 0.92))
        g.set_ylabels(clean(metric))
        g.set_xlabels("Number of Clients")
        if title:
            plt.subplots_adjust(top=0.9)
            g.fig.suptitle(title, fontsize=16)

        g.tight_layout()

        plt.savefig(target_path / f'significance_comb_{metric}_{analysis}.png', bbox_inches='tight', dpi=300)
        if no_output:
            plt.close(g.fig)
        else:
            plt.show()

    # Write results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(csv_output_path / f'significance_analysis_{metric}.csv', index=False)


# In[36]:


def plot_all_summary_performance(df:pd.DataFrame, metric:str="test_roc_auc", errorbar:str|None=None, save_dir:str|Path=save_dir, title:str="summary comparison", prefix='', cohort_order=None, palette="Accent", no_output=False, **kwargs):
    target_path = Path.joinpath(Path(save_dir), 'figures', 'summary', 'comparison', metric)
    target_path.mkdir(exist_ok=True, parents=True) 
    #df = df.groupby(by=['simulation_type', 'analysis', 'n_clients']).median(numeric_only=True).reset_index()
    if prefix:
        prefix += " " if not prefix.endswith(" ") else ""
    title = f"{prefix}{title}"
    
    vmin, vmax = (0.0, 1.0) if metric != 'test_mcc' else (-0.5, 1.0)
    dodge = True if len(df['analysis'].unique().tolist()) > 1 else False

    df = df.loc[df["simulation_type"] == "combinations"]
    df = df.loc[df['n_clients'] < df['n_clients'].max()]
    
    df_0 = df.loc[df['analysis'].str.contains("_0")]
    df_0_0 = df_0.loc[~df_0['analysis'].str.contains("_batch_")]
    df_0_1 = df_0.loc[df_0['analysis'].str.contains("_batch_")] 
    df_10 = df.loc[df['analysis'].str.contains("_10")]
    df_1_0 = df_10.loc[~df_10['analysis'].str.contains("_batch_")]
    df_1_1 = df_10.loc[df_10['analysis'].str.contains("_batch_")] 
    
    
    palette = sns.color_palette(palette, 12)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=False, sharex=False) 
    
    sns.pointplot(data=df_0_0, x='n_clients', y=metric, hue='analysis', ax=axes[0][0],
                palette=palette, dodge=dodge, marker=".", markersize=10, markeredgewidth=3,
                err_kws={'linewidth':0.5}, errorbar=errorbar, hue_order=cohort_order, **kwargs)  

    sns.pointplot(data=df_0_1, x='n_clients', y=metric, hue='analysis', ax=axes[0][1],
                palette=palette, dodge=dodge, marker=".", markersize=10, markeredgewidth=3,
                err_kws={'linewidth':0.5}, errorbar=errorbar, hue_order=cohort_order, **kwargs)  
    
    sns.pointplot(data=df_1_0, x='n_clients', y=metric, hue='analysis', ax=axes[1][0],
                palette=palette, dodge=dodge, marker=".", markersize=10, markeredgewidth=3, linestyles='--',
                err_kws={'linewidth':0.5}, errorbar=errorbar, hue_order=cohort_order, **kwargs)  
    
    sns.pointplot(data=df_1_1, x='n_clients', y=metric, hue='analysis', ax=axes[1][1],
                palette=palette, dodge=dodge, marker=".", markersize=10, markeredgewidth=3, linestyles='--',
                err_kws={'linewidth':0.5}, errorbar=errorbar, hue_order=cohort_order, **kwargs)  
    
    
    for ax in axes.flat:
        ax.set_ylim((vmin, vmax))
        ax.set_xlabel("number of cohorts")
        ax.set_ylabel(clean(metric))
        ax.legend().remove()
        
    axes[0][0].set_title(f"No Batch Correction - No Filter")
    axes[0][1].set_title(f"Batch Correction - No Filter")
    axes[1][0].set_title(f"No Batch Correction - 10% Filter")
    axes[1][1].set_title(f"Batch Correction - 10% Filter")
        
    fig.suptitle(title)
    
    # Collect handles and labels from all subplots
    handles, labels = [], []
    for ax in axes.flat:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    
    # Create custom legend handles
    custom_handles = []
    for handle, label in zip(handles, labels):
        linestyle = '--' if '10' in label else '-'
        color = handle.get_color()
        custom_handles.append(Line2D([0], [0], color=color, linestyle=linestyle, marker='.', markersize=0, markeredgewidth=3))
    
    # Create a shared legend
    legend = fig.legend(custom_handles, labels, title='Dataset Configurations', bbox_to_anchor=(0.5, 0.02), loc='upper center', ncol=4)    
    
    # param_box = f"""
    # Model Training Parameters
    # -------------------------
    # N_ESTIMATOR:       {N_ESTIMATOR}
    # MAX_DEPTH:         {MAX_DEPTH}
    # CRITERION:         {CRITERION}
    # MIN_SAMPLES_LEAF:  {MIN_SAMPLES_LEAF}
    # MAX_FEATURES:      {MAX_FEATURES}
    # MAX_SAMPLES:       {MAX_SAMPLES}
    # CROSS_VAL:         {CROSS_VAL}
    # CLASS_WEIGHT:      {CLASS_WEIGHT}
    # """
    # # Transform legend coordinates to figure coordinates
    # legend_bbox = legend.get_window_extent().transformed(fig.transFigure.inverted()).ymax
    
    # # Add the text box to the plot, aligned with the top of the legend
    # plt.text(0.99, legend_bbox, param_box, transform=fig.transFigure, fontsize=10, family="monospace",
    #          va='top', ha='left', bbox=dict(boxstyle="round,pad=0.5", edgecolor="darkgrey", facecolor="white"))
    
    plt.savefig(Path(target_path, f'summary_{title.replace(" ", "_")}.png'), bbox_inches='tight')
    if no_output:
        plt.close()
    else:
        plt.show()
    


# In[37]:


def plot_selected_summary_performance(df:pd.DataFrame, metric:str="test_roc_auc", errorbar:str|None=None, save_dir:str|Path=save_dir, title:str="select summary comparison", prefix='', cohort_order=None, palette="Accent", no_output=False, **kwargs):
    target_path = Path.joinpath(Path(save_dir), 'figures', 'summary', 'select_comparison', metric)
    target_path.mkdir(exist_ok=True, parents=True) 
    #df = df.groupby(by=['simulation_type', 'analysis', 'n_clients']).median(numeric_only=True).reset_index()
    if prefix:
        prefix += " " if not prefix.endswith(" ") else ""
    title = f"{prefix}{title}"

    df = df.loc[df["simulation_type"] == "combinations"]
    df = df.loc[df['n_clients'] < df['n_clients'].max()]
    df = df.loc[~df["analysis"].str.contains("func_")]
    
    vmin, vmax = (0.0, 1.0) if metric != 'test_mcc' else (-0.5, 1.0)
    dodge = True if len(df['analysis'].unique().tolist()) > 1 else False
    
    fig = plt.figure(figsize=(12, 8))
    
    sns.pointplot(data=df, x='n_clients', y=metric, hue='analysis',
                palette=palette, dodge=dodge, marker=".", markersize=10, markeredgewidth=3,
                err_kws={'linewidth':0.5}, errorbar=errorbar, hue_order=cohort_order, **kwargs)  
    
    plt.ylim((vmin, vmax))
    plt.xlabel("number of cohorts")
    plt.ylabel(clean(metric))
    plt.legend()        
    plt.title(title)

    
    # Create custom legend handles
    # custom_handles = []
    # for handle, label in zip(handles, labels):
    #     linestyle = '--' if '10' in label else '-'
    #     color = handle.get_color()
    #     custom_handles.append(Line2D([0], [0], color=color, linestyle=linestyle, marker='.', markersize=0, markeredgewidth=3))
    
    handles, labels = plt.gca().get_legend_handles_labels()
    labels = [clean(_) for _ in labels]
    # Create a shared legend
    plt.legend(handles, labels, title='Dataset Configurations', bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=4)   

    plt.savefig(Path(target_path, f'summary_{title.replace(" ", "_")}.png'), bbox_inches='tight')
    if no_output:
        plt.close()
    else:
        plt.show()
    


# In[38]:


def filter_data(_df, filter=None):
    # remove samples from filter list
    if filter is not None:
        for f in filter:
            _df = _df.loc[~_df['train_dataset'].str.contains(f, regex=False)]
            _df = _df.loc[_df['val_dataset'] != f]
            
    return _df

def split_data_by_train(_df, filter = None):
    _df = filter_data(_df, filter)
    _out_df = _df.copy()
    
    _df = _df['train_dataset'].str.split(', ').apply(pd.Series, 1).stack()
    _df.index = _df.index.droplevel(-1) # to line up with df's index
    _df.name = 'train_dataset' # needs a name to join
    
    _out_df = _out_df.drop(columns='train_dataset').join(_df)
        
    return _out_df
    


# In[39]:


summary_df = pd.DataFrame()

palette = sns.color_palette(
    ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']
)

cohort_order = ['Austria1', 'Brazil1', 'China1', 'China3', 'China5', 'France1', 'Germany1', 'Germany2', 'Italy1', 'Japan1', 'USA1', 'USA2']
disable_telegram = False if is_server() else True

def plot_results_for_experiment(experiment, metrics=['test_mcc'], palette='Accent', cohort_order=None):
    
    results_central_df = load_results(experiment, "central")
    results_local_df = load_results(experiment, "local")
    results_comb_df = load_results(experiment, "combinations")
    
    # split samples by train and build fed-like datasets
    train_split_results_comb_df = split_data_by_train(results_comb_df)
    fedlike_results_comb_df = split_data_by_train(results_comb_df, filter=['Brazil1', 'France1', 'USA1', 'USA2'])
    fedlike_results_central_df = filter_data(results_central_df, filter=['Brazil1', 'France1', 'USA1', 'USA2'])
    fedlike_results_local_df = filter_data(results_local_df, filter=['Brazil1', 'France1', 'USA1', 'USA2'])

    warnings.simplefilter("ignore")
    # Suppress warnings and logging INFO messages
    with warnings.catch_warnings():
        # Suppress INFO level logging for matplotlib
        logging.getLogger('matplotlib').setLevel(logging.ERROR)
        
        for m1,m2 in [('test_roc_auc', 'test_f1'), ('test_mcc', 'test_accuracy')]:
            plot_double_linebox(results_comb_df, results_central_df, results_local_df, metric1=m1, metric2=m2, 
                                title=f"{clean(experiment.name)} {m1} {m2} comb analysis - line by eval",  
                                errorbar='sd', no_output=True, palette=palette, hue_order=cohort_order, _anno_df = df_anno)
            plot_double_linebox(results_comb_df, results_central_df, results_local_df, metric1=m1, metric2=m2, 
                                title=f"{clean(experiment.name)} {m1} {m2} comb analysis - line by eval - no errorbar", 
                                errorbar=None, no_output=True, palette=palette, hue_order=cohort_order, _anno_df = df_anno)
        
        for metric in metrics:
            plot_significance_facetgrid(results_comb_df, metric=metric, cohort_order=cohort_order, no_output=True, palette=palette)
            plot_box_local(results_local_df, title=f"{metric} local baseline", metric=metric, no_output=True, palette=palette, hue_order=cohort_order)
            plot_performance_multi_box_additive_cohort(results_comb_df, results_central_df, results_local_df, title=f"{metric} comb analysis - additive cohorts", metric=metric, no_output=True, palette=palette, hue_order=cohort_order)
            plot_performance_line_additive_cohort(results_comb_df, results_central_df, results_local_df, title=f"{metric} comb analysis - additive cohorts", metric=metric, no_output=True, palette=palette, hue_order=cohort_order)
            plot_performance_violin_additive_cohort(results_comb_df, results_central_df, results_local_df, title=f"{metric} comb analysis - additive cohorts", metric=metric, no_output=True, palette=palette, hue_order=cohort_order)
            plot_performance_box_additive_cohort(results_comb_df, results_central_df, results_local_df, title=f"{metric} comb analysis - additive cohorts", metric=metric, no_output=True, palette=palette, hue_order=cohort_order)
            plot_performance_box(results_comb_df, results_central_df, results_local_df, title=f"{metric} comb analysis - by eval", metric=metric, no_output=True, palette=palette, hue_order=cohort_order)
            
            plot_performance_linebox(results_comb_df, results_central_df, results_local_df, errorbar=None, title=f"{metric} comb analysis - line by eval - no errorbar", metric=metric, no_output=True, palette=palette, hue_order=cohort_order)
            plot_performance_linebox(results_comb_df, results_central_df, results_local_df, errorbar='sd', title=f"{metric} comb analysis - line by eval", metric=metric, no_output=True, palette=palette, hue_order=cohort_order)
            plot_performance_linebox(fedlike_results_comb_df, fedlike_results_central_df, fedlike_results_local_df, errorbar=None, title=f"{metric} comb analysis - line by eval - fed datasets - no errorbar", metric=metric, no_output=True, palette=palette, hue_order=cohort_order)
            plot_performance_linebox(fedlike_results_comb_df, fedlike_results_central_df, fedlike_results_local_df, errorbar='sd', title=f"{metric} comb analysis - line by eval - fed datasets", metric=metric, no_output=True, palette=palette, hue_order=cohort_order)
            
            plot_performance_heatmap_additive_cohort(results_comb_df, results_local_df, show_core=False, metric=metric, title=None, no_output=True, max_clients=2, hue_order=cohort_order)
            
            for max_clients in range(1, min(results_comb_df['n_clients'].max() + 1, 12)):
                plot_performance_heatmap(results_comb_df, results_local_df, show_core=False, metric=metric, no_output=True, max_clients=max_clients)
            pass
 

# Use ProcessPoolExecutor to run the plotting function concurrently for each experiment
with concurrent.futures.ProcessPoolExecutor() as executor:
    futures = {executor.submit(plot_results_for_experiment, experiment, metrics, palette, cohort_order): experiment for experiment in experiments}
    disable_telegram = False if is_server() else True
    with telegram_tqdm(
        total=len(futures), desc=f'plotting results', 
        token=telegram_token, chat_id=telegram_chatid, disable=disable_telegram
    ) as pbar:
        for future in concurrent.futures.as_completed(futures):
            experiment = futures[future]
            try:
                future.result()
                pbar.update(1)
            except Exception as exc:
                print(f'{experiment.name} generated an exception: {exc}')
                traceback.print_exc()


# In[40]:


def load_summary(experiments):
    summary_df = pd.DataFrame()
    for experiment in experiments:
        results_comb_df = load_results(experiment, "combinations")
        summary_df = summarize_results(results_comb_df, summary_df, simulation_type='combinations')
    summary_df = summary_df.reset_index(drop=True)
    return summary_df   


# In[41]:


with telegram_tqdm(metrics, total=len(metrics)+1, desc="Summary Metrics", 
                   token=telegram_token, chat_id=telegram_chatid, disable=disable_telegram) as pbar:
    pbar.set_description(f"loading summary")
    # plot summary figures 
    summary_df = load_summary(experiments)
    summary_core_df = split_data_by_train(summary_df, filter=['Brazil1', 'France1', 'USA1', 'USA2'])
    summary_cohort_order = [experiment.name for experiment in experiments]
    selected_cohort_order = [_ for _ in summary_cohort_order if "func_" not in _]
    disable_telegram = False if is_server() else True
    pbar.update(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for metric in metrics:
            pbar.set_description(f"plotting summary - {metric}")
            plot_selected_summary_performance(summary_df, metric=metric, title=f"Dataset Configuration Performance Comparison - {metric}", errorbar='sd', palette=palette, cohort_order=selected_cohort_order, save_dir=save_dir, no_output=True)
            plot_selected_summary_performance(summary_df, metric=metric, title=f"Dataset Configuration Performance Comparison - {metric} - no errorbars", palette=palette, cohort_order=selected_cohort_order, save_dir=save_dir, no_output=True)
            
            plot_all_summary_performance(summary_df, metric=metric, title=f"Dataset Configuration Performance Comparison - {metric} - no errorbars", palette=palette, cohort_order=summary_cohort_order, save_dir=save_dir, no_output=True)
            plot_all_summary_performance(summary_df, metric=metric, title=f"Dataset Configuration Performance Comparison - {metric}", errorbar="sd", palette=palette, cohort_order=summary_cohort_order, save_dir=save_dir, no_output=True)
            plot_all_summary_performance(summary_core_df, metric=metric, title=f"Core Dataset Configuration Performance Comparison - {metric} - no errorbars", palette=palette, cohort_order=summary_cohort_order, save_dir=save_dir, no_output=True)
            plot_all_summary_performance(summary_core_df, metric=metric, title=f"Core Dataset Configuration Performance Comparison - {metric}", errorbar="sd", palette=palette, cohort_order=summary_cohort_order, save_dir=save_dir, no_output=True)
            pbar.update(1)
            


# ### misc

# In[42]:


df_anno['cohort_name'].value_counts(sort=False).sort_index()


# ### stop 
