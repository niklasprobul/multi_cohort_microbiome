import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
from sklearn.metrics import get_scorer, precision_recall_curve, auc
from types import SimpleNamespace
from pathlib import Path
from os.path import join as path_join
from os import cpu_count
from itertools import chain, combinations
import json

clean_names = {
    'Taxa + Function':'Taxa + Function ',
    'taxa_func':'Taxa + Function ',
    'taxa' : 'Taxa',
    'func' : 'Function',
    'brackets_taxa' : 'Taxa binned',
    'genus_taxa': 'IGC2 - genus level taxa',
    'count_taxa_metaphlan4_0': 'MetaPhlAn4',
    'taxa_reduced85': "Taxa + Function (corr < 0.85)",
    'taxa_reduced90': "Taxa + Function (corr < 0.90)",
    'taxa_reduced95': "Taxa + Function  (corr < 0.95)",
    'taxa_func_batch_bat_0': "Taxa + Function combat",
    'taxa_func_batch_pls_0': "Taxa + Function pls",
    'taxa_func_batch_svd_0': "Taxa + Function svd",
    'taxa_func_batch_bmc_0': "Taxa + Function bmc",
    'taxa_func_batch_rbe_0': "Taxa + Function rbe",
    'taxa_french_batch_rbe_0' : 'Taxa rbe',
    'taxa_french_batch_bat_0' : 'Taxa combat',
    'taxa_french_batch_pls_0' : 'Taxa pls',
    'taxa_french_batch_svd_0' : 'Taxa svd',
    'taxa_french_batch_bmc_0' : 'Taxa bmc',
    'count_taxa_0' : "Taxa raw 0%",
    'count_func_0' : "Function raw 0%",
    'count_taxa_func_0' : "Taxa + Function 0%",
    'count_taxa_10' : "Taxa raw 10%",
    'count_func_10' : "Function raw 10%",
    'count_taxa_func_10' : "Taxa + Function raw 10%",
    'taxa_french_batch_rbe_0' : 'Taxa rbe',
    'func_french_batch_rbe_0' : 'Function rbe',
    'taxa_func_french_batch_rbe_0' : 'Taxa + Function rbe',
    'taxa_french_batch_bat_0' : 'Taxa bat',
    'func_french_batch_bat_0' : 'Function bat',
    'taxa_func_french_batch_bat_0' : 'Taxa + Function bat',
    'taxa_french_batch_pls_0' : 'Taxa pls',
    'func_french_batch_pls_0' : 'Function pls',
    'taxa_func_french_batch_pls_0' : 'Taxa + Function pls',
    'taxa_french_batch_svd_10' : 'Taxa svd',
    'func_french_batch_svd_10' : 'Function svd',
    'taxa_func_french_batch_svd_0' : 'Taxa + Function svd',
    'taxa_french_batch_bmc_0' : 'Taxa bmc',
    'func_french_batch_bmc_0' : 'Function bmc',
    'taxa_func_french_batch_bmc_0' : 'Taxa + Function bmc',
    'taxa_french_batch_rbe_10' : 'Taxa rbe 10% filter',
    'func_french_batch_rbe_10' : 'Function rbe 10% filter',
    'taxa_func_french_batch_rbe_10' : 'Taxa + Function rbe 10% filter',
    'taxa_french_batch_bat_10' : 'Taxa bat 10% filter',
    'func_french_batch_bat_10' : 'Function bat 10% filter',
    'taxa_func_french_batch_bat_10' : 'Taxa + Function bat 10% filter',
    'taxa_french_batch_pls_10' : 'Taxa pls 10% filter',
    'func_french_batch_pls_10' : 'Function pls 10% filter',
    'taxa_func_french_batch_pls_10' : 'Taxa + Function pls 10% filter',
    'taxa_french_batch_svd_10' : 'Taxa svd 10% filter',
    'func_french_batch_svd_10' : 'Function svd 10% filter',
    'taxa_func_french_batch_svd_10' : 'Taxa + Function svd 10% filter',
    'taxa_french_batch_bmc_10' : 'Taxa bmc 10% filter',
    'func_french_batch_bmc_10' : 'Function bmc 10% filter',
    'taxa_func_french_batch_bmc_10' : 'Taxa + Function bmc 20% filter',
    'taxa_french_batch_rbe_20' : 'Taxa rbe 20% filter',
    'func_french_batch_rbe_20' : 'Function rbe 20% filter',
    'taxa_func_french_batch_rbe_20' : 'Taxa + Function rbe 20% filter',
    'taxa_french_batch_bat_20' : 'Taxa bat 20% filter',
    'func_french_batch_bat_20' : 'Function bat 20% filter',
    'taxa_func_french_batch_bat_20' : 'Taxa + Function bat 20% filter',
    'taxa_french_batch_pls_20' : 'Taxa pls 20% filter',
    'func_french_batch_pls_20' : 'Function pls 20% filter',
    'taxa_func_french_batch_pls_20' : 'Taxa + Function pls 20% filter',
    'taxa_french_batch_svd_20' : 'Taxa svd 20% filter',
    'func_french_batch_svd_20' : 'Function svd 20% filter',
    'taxa_func_french_batch_svd_20' : 'Taxa + Function svd 20% filter',
    'taxa_french_batch_bmc_20' : 'Taxa bmc 20% filter',
    'func_french_batch_bmc_20' : 'Function bmc 20% filter',
    'taxa_func_french_batch_bmc_20' : 'Taxa + Function bmc 20% filter',
    "test_mcc" : "MCC",
    "test_roc_auc" : "AUROC",
    "test_f1" : "F1",
    "test_pr_auc" : "AUPRC",
    "test_precision" : "Precision",
    "test_balanced_accuracy" : "Balanced Accuracy",
    "test_accuracy" : "Accuracy",
    "test_recall" : "Recall",
}

def clean(name:str, dict=clean_names):
    return dict.get(name, name)
    
def generate_heatmap_tables(_df: pd.DataFrame, metric:str="test_roc_auc", title:str = "heatmap", save_dir:str="./data/simulations/"):
    if save_dir:
        target_path = Path.joinpath(Path(save_dir), 'tables', 'heatmap', )
        target_path.mkdir(exist_ok=True, parents=True)    
        
    _df = _df.loc[:, ['train_dataset', 'val_dataset', metric]]
    _df = _df.groupby(by=['train_dataset', 'val_dataset']).median().reset_index()
    _df = _df.pivot(index='train_dataset', columns='val_dataset', values=metric)
    _df = _df.astype(np.float32)
    
    if save_dir:
        _df.to_csv(Path(target_path, f'{title}.tsv'), sep='\t')
    return _df
            
            
def is_server():
    n_cpus = cpu_count()
    if not n_cpus:
        return False
    return False if n_cpus < 33 else True

def nested_dict():
    return defaultdict(dict)


def append_to_json(file_path, new_data):
    file_path = Path(file_path)
    if file_path.exists():
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    if isinstance(data, list):
        data.append(new_data)
    else:
        data = [data, new_data]

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
        
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)  # allows duplicate elements
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def load_datasets(d, data_dir: str|Path= '.'):
    """Recursively convert dictionaries to SimpleNamespace objects."""
    if isinstance(d, dict):
        for key, value in d.items():
            d[key] = load_datasets(value, data_dir)
        return SimpleNamespace(**d)
    elif isinstance(d, list):
        return [load_datasets(item, data_dir) for item in d]
    else:
        return load_file(d, data_dir)
    
def load_file(file_name, file_dir):
    if file_name.endswith('.csv'):
        return pd.read_csv(path_join(file_dir, file_name), index_col=0)
    elif file_name.endswith('.tsv'):
        return pd.read_csv(path_join(file_dir, file_name), sep='\t', index_col=0)
    elif file_name.endswith('.xlsx'):
        return pd.read_excel(path_join(file_dir, file_name), index_col=0)
    else:
        raise ValueError(f'File type not supported: {path_join(file_dir, file_name)}')
    
def get_file_name():
    return __file__

def parse_df(df):
    if "gender" in df.columns:
        df["gender"] = df["gender"].replace({"female": 0, "male": 1})
    if "health_status" in df.columns:
        df["health_status"] = df["health_status"].replace({"H": 0, "P": 1})
    if 'study_accession' in df.columns:
        thomas_alias = {
            'PRJDB4176': 'V_Cohort2',
            'PRJEB10878': 'YuJ_2015',
            'PRJEB12449': 'VogtmannE_2016',
            'PRJEB27928': 'V_Cohort1',
            'PRJEB6070':  'ZellerG_2014',
            'PRJEB7774': 'FengQ_2015',
            'PRJNA389927': 'HanniganGD_2018',
        }
        
        df.loc[:, 'thomas_alias'] = df['study_accession'].map(thomas_alias)
        if "mgp_sample_alias" in df.columns:
            df.loc[df['mgp_sample_alias']=='Vercelli', 'thomas_alias'] = 'Cohort1'
            df.loc[df['mgp_sample_alias']=='Milan', 'thomas_alias'] = 'Cohort2'

    return df

def extract_labels(df, label_col="health_status"):
    if label_col in df.columns: 
        y = df.loc[:, label_col]
        X = df.drop(columns=label_col)
    else: 
        X, y = None, None
        print(f'{label_col} no in df columns!')
    return X,y

def add_score(data, method, score, df):
    df = pd.concat([df, pd.DataFrame({ "data" : [data]*len(score),"method": [method]*len(score),"score": score,})])
    return df

def custom_cross_validate(estimator, X, y, features, cv=5, scoring=None, feature_importance = 'gini'):
    #feature importance can be 'shap', 'permutation', 'both' or None

    # Initialize the cross-validation splitter if not provided
    if not isinstance(cv, StratifiedKFold):
        cv = StratifiedKFold(n_splits=cv)        
    
    # Initialize the scoring metrics
    if scoring is None:
        scoring = {'accuracy': get_scorer('accuracy')}
    
    scorers = scoring
    
    # Perform cross-validation
    scores = defaultdict(list)
    importances = defaultdict(nested_dict)
    
    for cv_n, (train_index, test_index) in enumerate(cv.split(X, y)):
        # Split the data into training and testing sets
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        
        # Fit the estimator on the training data
        estimator.fit(X_train, y_train)
        
        # Evaluate the estimator on the testing data
        
        # for dat in ('train', 'test'):
        for dat in ('test', ):
            if dat == 'train':
                _X = X_train
                _y = y_train
            else:
                _X = X_test
                _y = y_test
                
            for key, scorer in scorers.items():
                try:
                    score = scorer(estimator, _X, _y)
                except:
                    score = np.NaN
                
                scores[f'{dat}_{key}'].append(score)
            
            if dat == 'test':
                continue
            # Collect feature importances
            if isinstance(estimator, RandomForestClassifier) and False:
                if feature_importance:
                    feature_importance = feature_importance.lower()
                    if feature_importance not in ('shap', 'both', 'gini'):
                        raise ValueError("feature_importance must be one of 'gini', 'permutation', 'both' or None")
                    
                    if feature_importance in ('shap', 'both'):
                        
                        # explainer = shap.TreeExplainer(estimator)
                        # explanation = explainer(X_test)

                        # shap_values = explanation.values
                        # display(shap_values)
                        # display(explanation)
                        pass
                        # perm = permutation_importance(estimator, _X, _y, n_repeats=1, n_jobs=4, random_state=42, max_samples=1.0)
                        # sorted_importances_idx = perm.importances_mean.argsort()
                        
                        # for importance, feature in zip(perm.importances[sorted_importances_idx].T, X.columns[sorted_importances_idx]):

                        #     importances[f"{dat}_permutation_importance"][feature].append(importance.mean())
                    if feature_importance in ('gini', 'both'):
                        if features is None:
                            raise ValueError("features must be provided for gini importance")
                            continue
                        for feature, importance in zip(features, estimator.feature_importances_):
                            importances[f"gini_importance"][cv_n][feature] = importance 
                    
    return scores, importances
    
def custom_comb_validate(estimator, X_train, y_train, X_test, y_test, features, scoring=None, feature_importance = 'gini'):
    #feature importance can be 'shap', 'permutation', 'both' or None

    # Initialize the scoring metrics
    if scoring is None:
        scoring = {'accuracy': get_scorer('accuracy')}
    # scorers = {key: get_scorer(value) for key,value in scoring.items()}
    scorers = scoring

    # Perform cross-validation
    scores = defaultdict(list)
    importances = defaultdict(nested_dict)    
        
    # The estimator is already fitted on the training data!!
    # estimator.fit(X_train, y_train)
    
    # Evaluate the estimator on the testing data
    datasets = {
        #'train': (X_train, y_train),
        'test': (X_test, y_test)
    }
    
    for dat, (X, y) in datasets.items():
            
        for key, scorer in scorers.items():
            try:
                score = scorer(estimator, X, y)
            except:
                score = np.NaN
            
            scores[f'{dat}_{key}'].append(score)
        
        if dat == 'train' and feature_importance and False:
        # Collect feature importances
            if isinstance(estimator, RandomForestClassifier):
                if feature_importance in ('gini', 'both'):
                    for feature, importance in zip(features, estimator.feature_importances_):
                        importances[f"gini_importance"][0][feature] = importance

           
    return scores, importances

def transform_nested_dict_list(input_dict, index_name = 'features'):
    '''
    changes foo = 
        {'metric1': {0: {'m1': 1, 'm2': 2, 'm3': 3}, 1: {'m1': 1, 'm2': 2, 'm3': 3}},
         'metric2': {1: {'m1': 4, 'm2': 5, 'm3': 6}, 1: {'m1': 4, 'm2': 5, 'm3': 6}}}
    to bar = {"features": [['m1', 'm2', 'm3'],['m1', 'm2', 'm3']], "metric1": [[1,2,3], [4,5,6]], "metric2": [[1,2,3], [4,5,6]]} 
    
    '''
    
    result = defaultdict(list)
    metrics = list(input_dict.keys())
    
    
    for i, metric in enumerate(metrics):
        cvs = list(input_dict[metric].keys())
        features = list(input_dict[metric][cvs[0]].keys())
        for cv in cvs:
            if i == 0:
                result[index_name].append(features)
            result[metric].append(list(input_dict[metric][cv].values()))
        

    return result

def sensitivity_score(y_true, y_pred):
    TP = sum((y_true == 1) & (y_pred == 1))
    FN = sum((y_true == 1) & (y_pred == 0))
    if (TP + FN) == 0:
        sensitivity = 0
    else:
        sensitivity = TP / (TP + FN)
    return sensitivity

def specificity_score(y_true, y_pred):
    TN = sum((y_true == 0) & (y_pred == 0))
    FP = sum((y_true == 0) & (y_pred == 1))
    if (TN + FP) == 0:
        specificity = 0
    else:
        specificity = TN / (TN + FP)
    return specificity

def mcc_from_cm(confusion_matrix):
    
    TN = confusion_matrix[0, 0]
    FP = confusion_matrix[0, 1]
    FN = confusion_matrix[1, 0]
    TP = confusion_matrix[1, 1]

    denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    if denominator == 0:
        return 0
    
    # Calculate the MCC
    mcc = ((TP * TN) - (FP * FN)) / denominator

    return mcc

def pr_auc_score(y_true, y_scores):
    """
    Compute the area under the precision-recall curve (AUC-PR).
    
    Parameters:
    - y_true: True binary labels.
    - y_scores: Target scores, can either be probability estimates of the positive class, confidence values, or binary decisions.
    
    Returns:
    - AUC-PR value.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    return auc(recall, precision)

def main():
    pass

if __name__ == "__main__":
    main()
    