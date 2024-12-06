import pandas as pd
import os 
from scipy.stats import zscore
import utility
from pathlib import Path
import numpy as np

def prep_data(normalize=False, verbose=False):
    # %%
    parent_dir = './prep_data/raw/'
    ssd_path_in = '/media/niklas/T7/data/FeMAI_RAW/'
    if Path(ssd_path_in).is_dir():   
        parent_dir = os.path.join(ssd_path_in, parent_dir)
        
    output_dir = 'prep_data/'

    if not Path(output_dir).is_dir():
        Path(output_dir).mkdir(exist_ok=True)
    
    # anno_df = pd.read_excel(os.path.join(parent_dir, "metadata_2090_CRC_cohort_20230117.xlsx"), 'metadata_2090_CRC_cohort')
    # expr_df = pd.read_csv(os.path.join(parent_dir, "species_signal_2090_CRC_cohort_20230201.tsv"), sep='\t', index_col=0)
    # func_df = pd.read_csv(os.path.join(parent_dir, "functional_signal_2090_CRC_cohort_20221207.tsv"), sep='\t', index_col=0)
    # taxa_df = pd.read_csv(os.path.join(parent_dir, "species_taxo_phylo_20230201.tsv"), sep='\t', index_col=0)
    # modu_df = pd.read_csv(os.path.join(parent_dir, "modules_definition_GMM_GBM_KEGG_92.tsv"), sep='\t')
    # expr_meta_df = pd.read_csv(os.path.join(parent_dir, "metaphlan2_CRC_2090_species_signal_20230710.tsv"), sep='\t', index_col=0)
    
    # anno_df = pd.read_excel(os.path.join(parent_dir, "metadata_2250_CRC_cohort_20231114.xlsx"), 'metadata_1696_CRC_cohort')
    # expr_df = pd.read_csv(os.path.join(parent_dir, "species_signal_2250_CRC_cohort_20231115.tsv"), sep='\t', index_col=0)
    # func_df = pd.read_csv(os.path.join(parent_dir, "functional_signal_2250_CRC_cohort_20231114.tsv"), sep='\t', index_col=0)
    # taxa_df = pd.read_csv(os.path.join(parent_dir, "species_taxo_phylo_20231115.tsv"), sep='\t', index_col=0)
    # modu_df = pd.read_csv(os.path.join(parent_dir, "modules_definition_GMM_GBM_KEGG_92_20231114.tsv"), sep='\t')
    # expr_meta_df = pd.read_csv(os.path.join(parent_dir, "metaphlan2_CRC_2090_species_signal_20230710.tsv"), sep='\t', index_col=0)
    
    # # input file names 2250
    # anno = "metadata_2250_CRC_cohort_20231114.xlsx"
    # anno_sheet = "metadata_1696_CRC_cohort"
    # expr = "species_signal_2250_CRC_cohort_20231115.tsv"
    # func = "functional_signal_2250_CRC_cohort_20231114.tsv"
    # taxa = "species_taxo_phylo_20231115.tsv"
    # modu = "modules_definition_GMM_GBM_KEGG_92_20231114.tsv"
    # expr_meta = "metaphlan2_CRC_2090_species_signal_20230710.tsv"
    
    datasets = {
        "anno": {
            # "input": "metadata_2340_CRC_cohort_20240426.xlsx", # less strict quality controll
            "input": "metadata_2340_CRC_cohort_20240704.xlsx",
            "sheet": "metadata_2340_CRC_cohort",
            "output": "anno_full_clean.csv",
        },
        "anno_raw": {
            "input": "metadata_2340_CRC_cohort_20240426.xlsx",
            "sheet": "metadata_2340_CRC_cohort",
            "output": "anno_raw.csv",
        },
        "taxa": {
            "input": "species_signal_2340_CRC_cohort_20240322.tsv",
            "output": "taxa.csv",
        },
        "taxa_counts_0": {
            "input": "batch_effect_corrected_species_filtered0_2340_ech/species_signal_2340_CRC_cohort_20240617_no_batch_effect_correction_prev0.csv",
            "output": "taxa_counts_0.csv",
        },
        "taxa_counts_10": {
            "input": "batch_effect_corrected_species_filtered10_2340_ech/species_signal_2340_CRC_cohort_20240617_no_batch_effect_correction_prev10.csv",
            "output": "taxa_counts_10.csv",
        },
        "func": {
            "input": "functional_signal_2340_CRC_cohort_20240329.tsv",
            "output": "func.csv",
        },
        "func_counts_0": {
            "input": "batch_effect_corrected_functions_prev_0_2340_ech/functional_signal_2340_CRC_cohort_20240617_no_batch_effect_correction_prev0.tsv",
            "output": "func_counts_0.csv",
        },
        "func_counts_10": {
            "input": "batch_effect_corrected_functions_prev_10_2340_ech/functional_signal_2340_CRC_cohort_20240617_no_batch_effect_correction_prev10.tsv",
            "output": "func_counts_10.csv",
        },
        "taxa_meta": {
            "input": "metaphlan4_CRC_count_table_2340_CRC_cohort_20240607.tsv",
            "output": "taxa_meta4.csv",
        },
        "modules": {
            "input": "modules_definition_GMM_GBM_KEGG_92_20240329.tsv",
            "output": "modules.csv",
        },
        "species": {
            "input": "species_taxo_phylo_20240322.tsv",
            "output": "species.csv",
        },
        "french_batch_bat_0": {
            "input": "batch_effect_corrected_species_filtered0_2340_ech/species_signal_2340_CRC_cohort_20240617_combat_prev0.csv",
            "output": "mbec_corrected_bat_prev0.csv",
        },
        "french_batch_pls_0": {
            "input": "batch_effect_corrected_species_filtered0_2340_ech/species_signal_2340_CRC_cohort_20240617_plsda_prev0.csv",
            "output": "mbec_corrected_pls_prev0.csv",
        },
        "french_batch_rbe_0": {
            "input": "batch_effect_corrected_species_filtered0_2340_ech/species_signal_2340_CRC_cohort_20240617_rbe_prev0.csv",
            "output": "mbec_corrected_rbe_prev0.csv",
        },
        "french_batch_svd_0": {
            "input": "batch_effect_corrected_species_filtered0_2340_ech/species_signal_2340_CRC_cohort_20240617_svd_prev0.csv",
            "output": "mbec_corrected_svd_prev0.csv",
        },
        "french_batch_bmc_0": {
            "input": "batch_effect_corrected_species_filtered0_2340_ech/species_signal_2340_CRC_cohort_20240617_BMC_prev0.csv",
            "output": "mbec_corrected_bmc_prev0.csv",
        },
        "french_batch_bat_10": {
            "input": "batch_effect_corrected_species_filtered10_2340_ech/species_signal_2340_CRC_cohort_20240617_combat_prev10.csv",
            "output": "mbec_corrected_bat_prev10.csv",
        },
        "french_batch_pls_10": {
            "input": "batch_effect_corrected_species_filtered10_2340_ech/species_signal_2340_CRC_cohort_20240617_plsda_prev10.csv",
            "output": "mbec_corrected_pls_prev10.csv",
        },
        "french_batch_rbe_10": {
            "input": "batch_effect_corrected_species_filtered10_2340_ech/species_signal_2340_CRC_cohort_20240617_rbe_prev10.csv",
            "output": "mbec_corrected_rbe_prev10.csv",
        },
        "french_batch_svd_10": {
            "input": "batch_effect_corrected_species_filtered10_2340_ech/species_signal_2340_CRC_cohort_20240617_svd_prev10.csv",
            "output": "mbec_corrected_svd_prev10.csv",
        },
        "french_batch_bmc_10": {
            "input": "batch_effect_corrected_species_filtered10_2340_ech/species_signal_2340_CRC_cohort_20240617_BMC_prev10.csv",
            "output": "mbec_corrected_bmc_prev10.csv",
        },
        "french_batch_func_bat_0": {
            "input": "batch_effect_corrected_functions_prev_0_2340_ech/functional_signal_2340_CRC_cohort_20240617_combat_correction_prev0.tsv",
            "output": "mbec_corrected_func_bat_prev0.csv",
        },
        "french_batch_func_pls_0": {
            "input": "batch_effect_corrected_functions_prev_0_2340_ech/functional_signal_2340_CRC_cohort_20240617_plsda_correction_prev0.tsv",
            "output": "mbec_corrected_func_pls_prev0.csv",
        },
        "french_batch_func_rbe_0": {
            "input": "batch_effect_corrected_functions_prev_0_2340_ech/functional_signal_2340_CRC_cohort_20240617_rbe_correction_prev0.tsv",
            "output": "mbec_corrected_func_rbe_prev0.csv",
        },
        "french_batch_func_bmc_0": {
            "input": "batch_effect_corrected_functions_prev_0_2340_ech/functional_signal_2340_CRC_cohort_20240617_BMC_correction_prev0.tsv",
            "output": "mbec_corrected_func_bmc_prev0.csv",
        },
        "french_batch_func_svd_0": {
            "input": "batch_effect_corrected_functions_prev_0_2340_ech/functional_signal_2340_CRC_cohort_20240617_svd_correction_prev0.tsv",
            "output": "mbec_corrected_func_svd_prev0.csv",
        },
        "french_batch_func_bat_10": {
            "input": "batch_effect_corrected_functions_prev_10_2340_ech/functional_signal_2340_CRC_cohort_20240617_combat_correction_prev10.tsv",
            "output": "mbec_corrected_func_bat_prev10.csv",
        },
        "french_batch_func_pls_10": {
            "input": "batch_effect_corrected_functions_prev_10_2340_ech/functional_signal_2340_CRC_cohort_20240617_plsda_correction_prev10.tsv",
            "output": "mbec_corrected_func_pls_prev10.csv",
        },
        "french_batch_func_rbe_10": {
            "input": "batch_effect_corrected_functions_prev_10_2340_ech/functional_signal_2340_CRC_cohort_20240617_rbe_correction_prev10.tsv",
            "output": "mbec_corrected_func_rbe_prev10.csv",
        },
        "french_batch_func_bmc_10": {
            "input": "batch_effect_corrected_functions_prev_10_2340_ech/functional_signal_2340_CRC_cohort_20240617_BMC_correction_prev10.tsv",
            "output": "mbec_corrected_func_bmc_prev10.csv",
        },
        "french_batch_func_svd_10": {
            "input": "batch_effect_corrected_functions_prev_10_2340_ech/functional_signal_2340_CRC_cohort_20240617_svd_correction_prev10.tsv",
            "output": "mbec_corrected_func_svd_prev10.csv",
        },
    }

    datasets.update({
        "taxa_func": {
            "taxa": datasets["taxa"]["output"],
            "func": datasets["func"]["output"],
            "output": "taxa_func.csv",
        },
        "taxa_func_counts_0": {
            "taxa": datasets["taxa_counts_0"]["output"],
            "func": datasets["func_counts_0"]["output"],
            "output": "taxa_func_counts_0.csv",
        },
        "taxa_func_counts_10": {
            "taxa": datasets["taxa_counts_10"]["output"],
            "func": datasets["func_counts_10"]["output"],
            "output": "taxa_func_counts_10.csv",
        },
        "french_batch_taxa_func_bat_0": {
            "taxa": datasets["french_batch_bat_0"]["output"],
            "func": datasets["french_batch_func_bat_0"]["output"],
            "output": "mbec_corrected_taxa_func_bat_prev0.csv",
        },
        "french_batch_taxa_func_pls_0": {
            "taxa": datasets["french_batch_pls_0"]["output"],
            "func": datasets["french_batch_func_pls_0"]["output"],
            "output": "mbec_corrected_taxa_func_pls_prev0.csv",
        },
        "french_batch_taxa_func_rbe_0": {
            "taxa": datasets["french_batch_rbe_0"]["output"],
            "func": datasets["french_batch_func_rbe_0"]["output"],
            "output": "mbec_corrected_taxa_func_rbe_prev0.csv",
        },
        "french_batch_taxa_func_svd_0": {
            "taxa": datasets["french_batch_svd_0"]["output"],
            "func": datasets["french_batch_func_svd_0"]["output"],
            "output": "mbec_corrected_taxa_func_svd_prev0.csv",
        },
        "french_batch_taxa_func_bmc_0": {
            "taxa": datasets["french_batch_bmc_0"]["output"],
            "func": datasets["french_batch_func_bmc_0"]["output"],
            "output": "mbec_corrected_taxa_func_bmc_prev0.csv",
        },
        "french_batch_taxa_func_bat_10": {
            "taxa": datasets["french_batch_bat_10"]["output"],
            "func": datasets["french_batch_func_bat_10"]["output"],
            "output": "mbec_corrected_taxa_func_bat_prev10.csv",
        },
        "french_batch_taxa_func_pls_10": {
            "taxa": datasets["french_batch_pls_10"]["output"],
            "func": datasets["french_batch_func_pls_10"]["output"],
            "output": "mbec_corrected_taxa_func_pls_prev10.csv",
        },
        "french_batch_taxa_func_rbe_10": {
            "taxa": datasets["french_batch_rbe_10"]["output"],
            "func": datasets["french_batch_func_rbe_10"]["output"],
            "output": "mbec_corrected_taxa_func_rbe_prev10.csv",
        },
        "french_batch_taxa_func_svd_10": {
            "taxa": datasets["french_batch_svd_10"]["output"],
            "func": datasets["french_batch_func_svd_10"]["output"],
            "output": "mbec_corrected_taxa_func_svd_prev10.csv",
        },
        "french_batch_taxa_func_bmc_10": {
            "taxa": datasets["french_batch_bmc_10"]["output"],
            "func": datasets["french_batch_func_bmc_10"]["output"],
            "output": "mbec_corrected_taxa_func_bmc_prev10.csv",
        },
        # "french_batch_taxa_func_bat_20": {
        #     "taxa": datasets["french_batch_bat_20"]["output"],
        #     "func": datasets["french_batch_func_bat_20"]["output"],
        #     "output": "mbec_corrected_taxa_func_bat_prev20.csv",
        # },
        # "french_batch_taxa_func_pls_20": {
        #     "taxa": datasets["french_batch_pls_20"]["output"],
        #     "func": datasets["french_batch_func_pls_20"]["output"],
        #     "output": "mbec_corrected_taxa_func_pls_prev20.csv",
        # },
        # "french_batch_taxa_func_rbe_20": {
        #     "taxa": datasets["french_batch_rbe_20"]["output"],
        #     "func": datasets["french_batch_func_rbe_20"]["output"],
        #     "output": "mbec_corrected_taxa_func_rbe_prev20.csv",
        # },
    })
    
    clean_cohort_names = {
        'PRJDB4176 - JPN' : 'Japan1',
        'PRJEB10878 - CHN' : 'China1',
        'PRJEB12449 - USA' : 'USA1',
        'PRJEB27928 - DEU' : 'Germany1',
        'PRJEB6070 - FRA' : 'France1', 
        'PRJEB6070 - DEU' : 'Germany2',
        'PRJEB7774 - AUT' : 'Austria1',
        'PRJNA389927 - USA' : 'USA2',
        'PRJNA389927 - CAN' : 'Canada1',
        'PRJNA397112 - IND' : 'India1',
        'PRJNA447983 - ITA' : 'Italy1',
        'PRJNA531273 - IND' : 'India2',
        'PRJNA608088 - CHN' : 'China2',
        'PRJNA429097 - CHN' : 'China3',
        'PRJNA763023 - CHN' : 'China4',
        'PRJNA731589 - CHN' : 'China5',
        'PRJNA961076 - BRA' : 'Brazil1',
    }

    def read_file(dataset, values):
        # read and format file
        if "input" in values:
            if not os.path.isfile(os.path.join(parent_dir, values["input"])):
                if verbose:
                    print(f"Did not find {dataset} input file: {values['input']}")
                return None
                
            if dataset in ["anno", "anno_raw"]:
                _df = pd.read_excel(os.path.join(parent_dir, values["input"]), values["sheet"])
            else:
                # sep = '\t' if values["input"].endswith('.tsv') else ','
                sep = '\t'
                _df = pd.read_csv(os.path.join(parent_dir, values["input"]), sep=sep, index_col=0)

                if dataset not in ["modules", "species"]:
                    _df = _df.T
                
            # if dataset.startswith("functional_"):
            #     pref = dataset.split("_")[2] + "."
            #     _df.index = _df.index.str.replace(pref, '', regex=False)

        else:
            try:
                left = pd.read_csv(os.path.join(output_dir, values["taxa"]), index_col=0)
                right = pd.read_csv(os.path.join(output_dir, values["func"]), index_col=0)
            except Exception as e:
                if verbose:
                    print(f"Error merging base files for {dataset}: {e}")
                return None
            
            _df = pd.merge(left=left, left_index=True, right=right, right_index=True)
        
        return _df    
        
    
    def write_file(_df, dataset, values):
        if "output" in values:
            if os.path.isfile(os.path.join(output_dir, values["output"])):
                if verbose:
                    print(f"overwriting {dataset} output file: {os.path.join(output_dir, values['output'])}")
                    
        _df.to_csv(os.path.join(output_dir, values["output"]))
        
    for dataset, values in datasets.items():
        _df = read_file(dataset, values)
        
        if _df is None:
            continue
        
        if dataset == "anno":
            _df = _df.loc[_df['to_exclude'].isna()]
            _df = _df.dropna(subset=["health_status", "gender", "bmi", "age"])
            
            _df = utility.parse_df(_df)
            _df = _df.loc[_df["host_phenotype"] != "adenoma"]
            
        if dataset in ["anno", "anno_raw"]:
            _df.set_index("sample", inplace=True, drop=True)
            _df['cohort_name'] = _df['study_accession'] + ' - ' + _df['country']
            _df['cohort_name'] = _df['cohort_name'].map(clean_cohort_names).fillna(_df['cohort_name'])
            
        write_file(_df, dataset, values)
        if verbose:
            print(f"Processed {dataset}")


if __name__ == "__main__":
    prep_data()