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
    
    
    # input file names 2340
    anno = "metadata_2340_CRC_cohort_20240426.xlsx"
    anno_sheet = "metadata_2340_CRC_cohort"
    taxa = "species_signal_2340_CRC_cohort_20240322.tsv"
    taxa_counts = "species_signal_count_table_2340_CRC_cohort_20240322.tsv"
    func = "functional_signal_2340_CRC_cohort_20240329.tsv"
    func_counts = "functional_signal_count_2340_CRC_cohort_20240329.tsv"
    species = "species_taxo_phylo_20240322.tsv"
    modu = "modules_definition_GMM_GBM_KEGG_92_20240329.tsv"
    taxa_meta = "metaphlan4_CRC_count_table_2340_CRC_cohort_20240607.tsv"
    
    french_batch_bat_0 = 'mbec_corrected_bat_prev0.csv'
    french_batch_pls_0 = 'mbec_corrected_pls_prev0.csv'
    french_batch_rbe_0 = 'mbec_corrected_rbe_prev0.csv'
    french_batch_svd_0 = 'mbec_corrected_svd_prev0.csv'
    french_batch_bmc_0 = 'mbec_corrected_bmc_prev0.csv'

    french_batch_bat_10 = 'mbec_corrected_bat_prev10.csv'
    french_batch_pls_10 = 'mbec_corrected_pls_prev10.csv'
    french_batch_rbe_10 = 'mbec_corrected_rbe_prev10.csv'
    french_batch_svd_10 = 'mbec_corrected_svd_prev10.csv'
    french_batch_bmc_10 = 'mbec_corrected_bmc_prev10.csv'

    french_batch_bat_20 = 'mbec_corrected_bat_prev20.csv'
    french_batch_pls_20 = 'mbec_corrected_pls_prev20.csv'
    french_batch_rbe_20 = 'mbec_corrected_rbe_prev20.csv'
    french_batch_svd_20 = 'mbec_corrected_svd_prev20.csv'
    french_batch_bmc_20 = 'mbec_corrected_bmc_prev20.csv'

    french_batch_func_pls_0 = 'functional_signal_2250_CRC_cohort_20240307_plsda_correction_prev0.tsv'
    french_batch_func_rbe_0 = 'functional_signal_2250_CRC_cohort_20240307_rbe_correction_prev0.tsv'
    french_batch_func_bat_0 = 'functional_signal_2250_CRC_cohort_20240307_bat_correction_prev0.tsv'
    french_batch_func_bmc_0 = 'functional_signal_2250_CRC_cohort_20240307_bmc_correction_prev0.tsv'

    french_batch_func_bat_10 = 'functional_signal_2250_CRC_cohort_20240307_bat_correction_prev10.tsv'
    french_batch_func_pls_10 = 'functional_signal_2250_CRC_cohort_20240307_plsda_correction_prev10.tsv'
    french_batch_func_rbe_10 = 'functional_signal_2250_CRC_cohort_20240307_rbe_correction_prev10.tsv'
    french_batch_func_bmc_10 = 'functional_signal_2250_CRC_cohort_20240307_bmc_correction_prev10.tsv'
    french_batch_func_svd_10 = 'functional_signal_2250_CRC_cohort_20240307_svd_correction_prev10.tsv'

    french_batch_func_bat_20 = 'functional_signal_2250_CRC_cohort_20240311_bat_correction_prev20.tsv'
    french_batch_func_pls_20 = 'functional_signal_2250_CRC_cohort_20240313_plsda_correction_prev20.tsv'
    french_batch_func_rbe_20 = 'functional_signal_2250_CRC_cohort_20240311_rbe_correction_prev20.tsv'
    
    # reading all files
    anno_df = pd.read_excel(os.path.join(parent_dir, anno), anno_sheet)
    taxa_df = pd.read_csv(os.path.join(parent_dir, taxa), sep='\t', index_col=0)
    func_df = pd.read_csv(os.path.join(parent_dir, func), sep='\t', index_col=0)
    species = pd.read_csv(os.path.join(parent_dir, species), sep='\t', index_col=0)
    modu_df = pd.read_csv(os.path.join(parent_dir, modu), sep='\t')
    expr_meta_df = pd.read_csv(os.path.join(parent_dir, taxa_meta), sep='\t', index_col=0)
    
    df_expr_bat_0 = pd.read_csv(os.path.join(parent_dir, french_batch_bat_0), index_col=0).T
    df_expr_pls_0 = pd.read_csv(os.path.join(parent_dir, french_batch_pls_0), index_col=0).T
    df_expr_rbe_0 = pd.read_csv(os.path.join(parent_dir, french_batch_rbe_0), index_col=0).T
    df_expr_bmc_0 = pd.read_csv(os.path.join(parent_dir, french_batch_bmc_0), index_col=0).T
    df_expr_svd_0 = pd.read_csv(os.path.join(parent_dir, french_batch_svd_0), index_col=0).T
    
    df_expr_bat_10 = pd.read_csv(os.path.join(parent_dir, french_batch_bat_10), index_col=0).T
    df_expr_pls_10 = pd.read_csv(os.path.join(parent_dir, french_batch_pls_10), index_col=0).T
    df_expr_rbe_10 = pd.read_csv(os.path.join(parent_dir, french_batch_rbe_10), index_col=0).T
    df_expr_bmc_10 = pd.read_csv(os.path.join(parent_dir, french_batch_bmc_10), index_col=0).T
    df_expr_svd_10 = pd.read_csv(os.path.join(parent_dir, french_batch_svd_10), index_col=0).T
    
    df_expr_bat_20 = pd.read_csv(os.path.join(parent_dir, french_batch_bat_20), index_col=0).T
    df_expr_pls_20 = pd.read_csv(os.path.join(parent_dir, french_batch_pls_20), index_col=0).T
    df_expr_rbe_20 = pd.read_csv(os.path.join(parent_dir, french_batch_rbe_20), index_col=0).T
    df_expr_bmc_20 = pd.read_csv(os.path.join(parent_dir, french_batch_bmc_20), index_col=0).T
    df_expr_svd_20 = pd.read_csv(os.path.join(parent_dir, french_batch_svd_20), index_col=0).T
    
    df_func_bat_0 = pd.read_csv(os.path.join(parent_dir, french_batch_func_bat_0), index_col=0, sep='\t').T
    df_func_pls_0 = pd.read_csv(os.path.join(parent_dir, french_batch_func_pls_0), index_col=0, sep='\t').T
    df_func_rbe_0 = pd.read_csv(os.path.join(parent_dir, french_batch_func_rbe_0), index_col=0, sep='\t').T
    df_func_bmc_0 = pd.read_csv(os.path.join(parent_dir, french_batch_func_bmc_0), index_col=0, sep='\t').T
    # df_func_svd_0 = pd.read_csv(os.path.join(parent_dir, french_batch_func_svd_0), index_col=0).T
    
    df_func_bat_10 = pd.read_csv(os.path.join(parent_dir, french_batch_func_bat_10), index_col=0, sep='\t').T
    df_func_pls_10 = pd.read_csv(os.path.join(parent_dir, french_batch_func_pls_10), index_col=0, sep='\t').T
    df_func_rbe_10 = pd.read_csv(os.path.join(parent_dir, french_batch_func_rbe_10), index_col=0, sep='\t').T
    df_func_bmc_10 = pd.read_csv(os.path.join(parent_dir, french_batch_func_bmc_10), index_col=0, sep='\t').T
    df_func_svd_10 = pd.read_csv(os.path.join(parent_dir, french_batch_func_svd_10), index_col=0, sep='\t').T
    
    df_func_bat_20 = pd.read_csv(os.path.join(parent_dir, french_batch_func_bat_20), index_col=0, sep='\t').T
    df_func_pls_20 = pd.read_csv(os.path.join(parent_dir, french_batch_func_pls_20), index_col=0, sep='\t').T
    df_func_rbe_20 = pd.read_csv(os.path.join(parent_dir, french_batch_func_rbe_20), index_col=0, sep='\t').T
    # df_func_bmc_20 = pd.read_csv(os.path.join(parent_dir, french_batch_func_bmc_20), index_col=0).T
    # df_func_svd_20 = pd.read_csv(os.path.join(parent_dir, french_batch_func_svd_20), index_col=0).T
    

    df_expr_pls_0.index = df_expr_pls_0.index.str.replace('pls.', '', regex=False)
    df_expr_bat_0.index = df_expr_bat_0.index.str.replace('bat.', '', regex=False)
    df_expr_rbe_0.index = df_expr_rbe_0.index.str.replace('rbe.', '', regex=False)
    df_expr_svd_0.index = df_expr_svd_0.index.str.replace('svd.', '', regex=False)
    df_expr_bmc_0.index = df_expr_bmc_0.index.str.replace('bmc.', '', regex=False)
    
    df_expr_pls_10.index = df_expr_pls_10.index.str.replace('pls.', '', regex=False)
    df_expr_bat_10.index = df_expr_bat_10.index.str.replace('bat.', '', regex=False)
    df_expr_rbe_10.index = df_expr_rbe_10.index.str.replace('rbe.', '', regex=False)
    df_expr_svd_10.index = df_expr_svd_10.index.str.replace('svd.', '', regex=False)
    df_expr_bmc_10.index = df_expr_bmc_10.index.str.replace('bmc.', '', regex=False)
    
    df_expr_pls_20.index = df_expr_pls_20.index.str.replace('pls.', '', regex=False)
    df_expr_bat_20.index = df_expr_bat_20.index.str.replace('bat.', '', regex=False)
    df_expr_rbe_20.index = df_expr_rbe_20.index.str.replace('rbe.', '', regex=False)
    df_expr_svd_20.index = df_expr_svd_20.index.str.replace('svd.', '', regex=False)
    df_expr_bmc_20.index = df_expr_bmc_20.index.str.replace('bmc.', '', regex=False)

    df_french_bat_0 = pd.merge(left=df_expr_bat_0, left_index=True, right=df_func_bat_0, right_index=True)
    df_french_pls_0 = pd.merge(left=df_expr_pls_0, left_index=True, right=df_func_pls_0, right_index=True)
    df_french_rbe_0 = pd.merge(left=df_expr_rbe_0, left_index=True, right=df_func_rbe_0, right_index=True)

    df_french_bat_10 = pd.merge(left=df_expr_bat_10, left_index=True, right=df_func_bat_10, right_index=True)
    df_french_pls_10 = pd.merge(left=df_expr_pls_10, left_index=True, right=df_func_pls_10, right_index=True)
    df_french_rbe_10 = pd.merge(left=df_expr_rbe_10, left_index=True, right=df_func_rbe_10, right_index=True)
    
    df_french_bat_20 = pd.merge(left=df_expr_bat_20, left_index=True, right=df_func_bat_20, right_index=True)
    df_french_pls_20 = pd.merge(left=df_expr_pls_20, left_index=True, right=df_func_pls_20, right_index=True)
    df_french_rbe_20 = pd.merge(left=df_expr_rbe_20, left_index=True, right=df_func_rbe_20, right_index=True)
    
    
    df_taxa_counts = pd.read_csv(os.path.join(parent_dir, taxa_counts), index_col=0)
    df_func_counts = pd.read_csv(os.path.join(parent_dir, func_counts), index_col=0, sep = '\t').T

    # output file names
    exp_name = 'exp_all.csv'
    exp_meta_name = 'exp_meta_all.csv'
    func_name = 'func_all.csv'
    exp_func_name = 'taxa_func_all.csv'
    
    expr_genus_name = 'exp_genus.csv'
    expr_corr85_name = 'exp_corr85.csv'
    expr_corr90_name = 'exp_corr90.csv'
    expr_corr95_name = 'exp_corr95.csv'
    anno_brackets_name = 'anno_brackets.csv'

    simulated_exp = {
        'easy' : 'easy_data.csv',
        'hard' : 'hard_data.csv',
    }

    anno_raw_name = 'anno_full_raw.csv'
    anno_all_name = 'anno_full_clean.csv'
    anno_no_adenoma_name = 'anno_noadenoma_clean.csv'

    taxa_name = 'taxa.csv'
    taxa_meta_name = 'taxa_meta.csv'
    
    taxa_counts = 'taxa_counts.csv'
    func_counts =  'func_counts.csv'
    taxa_func_counts = 'taxa_func_counts.csv'
    
    
    french_batch_func_bat_0 = 'mbec_corrected_func_bat_prev0.csv'
    french_batch_func_pls_0 = 'mbec_corrected_func_pls_prev0.csv'
    french_batch_func_rbe_0 = 'mbec_corrected_func_rbe_prev0.csv'
    french_batch_func_svd_0 = 'mbec_corrected_func_svd_prev0.csv'
    french_batch_func_bmc_0 = 'mbec_corrected_func_bmc_prev0.csv'

    french_batch_func_bat_10 = 'mbec_corrected_func_bat_prev10.csv'
    french_batch_func_pls_10 = 'mbec_corrected_func_pls_prev10.csv'
    french_batch_func_rbe_10 = 'mbec_corrected_func_rbe_prev10.csv'
    french_batch_func_svd_10 = 'mbec_corrected_func_svd_prev10.csv'
    french_batch_func_bmc_10 = 'mbec_corrected_func_bmc_prev10.csv'

    french_batch_func_bat_20 = 'mbec_corrected_func_bat_prev20.csv'
    french_batch_func_pls_20 = 'mbec_corrected_func_pls_prev20.csv'
    french_batch_func_rbe_20 = 'mbec_corrected_func_rbe_prev20.csv'
    french_batch_func_svd_20 = 'mbec_corrected_func_svd_prev20.csv'
    french_batch_func_bmc_20 = 'mbec_corrected_func_bmc_prev20.csv'
    
    
    french_batch_comb_bat_0 = 'mbec_corrected_comb_bat_prev0.csv'
    french_batch_comb_pls_0 = 'mbec_corrected_comb_pls_prev0.csv'
    french_batch_comb_rbe_0 = 'mbec_corrected_comb_rbe_prev0.csv'

    french_batch_comb_bat_10 = 'mbec_corrected_comb_bat_prev10.csv'
    french_batch_comb_pls_10 = 'mbec_corrected_comb_pls_prev10.csv'
    french_batch_comb_rbe_10 = 'mbec_corrected_comb_rbe_prev10.csv'

    french_batch_comb_bat_20 = 'mbec_corrected_comb_bat_prev20.csv'
    french_batch_comb_pls_20 = 'mbec_corrected_comb_pls_prev20.csv'
    french_batch_comb_rbe_20 = 'mbec_corrected_comb_rbe_prev20.csv'

    # format and norm func and exp data
    taxa_df.index = species.index
    taxa_df = taxa_df.T
    
    expr_genus_df = taxa_df.copy()
    expr_genus_df = pd.concat([expr_genus_df, species['genus']], axis=1)
    expr_genus_df.dropna(inplace=True)
    expr_genus_df = expr_genus_df.groupby(by='genus').sum().T
    
    expr_corr_matrix = taxa_df.corr().abs()
    upper = expr_corr_matrix.where(np.triu(np.ones(expr_corr_matrix.shape), k=1).astype(bool))

    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    expr_reduced95_df = taxa_df.drop(to_drop, axis=1)

    to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
    expr_reduced90_df = taxa_df.drop(to_drop, axis=1)

    to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]
    expr_reduced85_df = taxa_df.drop(to_drop, axis=1)

    taxa_meta_df = expr_meta_df.loc[:,['#SampleID']]
    expr_meta_df = expr_meta_df.drop(columns=['#SampleID']).T
    
    func_df = func_df.T
    
    exp_fun_df = pd.concat([taxa_df, func_df],axis=1)

    # fix annotation file and remove NaNs
    anno_df = anno_df.set_index("sample")
    anno_df = anno_df.loc[anno_df['to_exclude'].isna()]
    
    anno_all_df = anno_df.dropna(subset=["health_status", "gender", "bmi", "age"])
    anno_all_df = utility.parse_df(anno_all_df)
        
    anno_brackets_df = anno_all_df.copy()
    anno_brackets_df['age'] = pd.cut(anno_brackets_df['age'], bins=[0,20,30,40,50,60,70,80,100], labels=[15, 25, 35, 45, 55, 65, 75, 85])
    anno_brackets_df['bmi'] = pd.cut(anno_brackets_df['bmi'], bins=[0,18.5,25,30,35,40,100], labels=[15, 22.5, 27.5, 32.5, 37.5, 45])
    
    anno_noadenoma_df = anno_all_df.loc[anno_all_df['host_phenotype'] != "adenoma"]

    df_taxa_func_counts = pd.merge(df_taxa_counts, df_func_counts, left_index=True, right_index=True) 
    exp_fun_df = pd.merge(left=taxa_df, left_index=True, right=func_df, right_index=True)
    #expr_df = expr_df.loc[anno_df.index]
    #func_df = func_df.loc[anno_df.index]
    #exp_fun_df = exp_fun_df.loc[anno_df.index]

    # write files
    species.to_csv(os.path.join(output_dir, taxa_name))
    taxa_df.to_csv(os.path.join(output_dir, exp_name))
    func_df.to_csv(os.path.join(output_dir, func_name))
    exp_fun_df.to_csv(os.path.join(output_dir, exp_func_name))
    
    expr_genus_df.to_csv(os.path.join(output_dir, expr_genus_name))
    expr_reduced85_df.to_csv(os.path.join(output_dir, expr_corr85_name))
    expr_reduced90_df.to_csv(os.path.join(output_dir, expr_corr90_name))
    expr_reduced95_df.to_csv(os.path.join(output_dir, expr_corr95_name))
    
    expr_meta_df.to_csv(os.path.join(output_dir, exp_meta_name))
    taxa_meta_df.to_csv(os.path.join(output_dir, taxa_meta_name))
    
    anno_df.to_csv(os.path.join(output_dir, anno_raw_name))
    anno_all_df.to_csv(os.path.join(output_dir, anno_all_name))
    anno_noadenoma_df.to_csv(os.path.join(output_dir, anno_no_adenoma_name))
    anno_brackets_df.to_csv(os.path.join(output_dir, anno_brackets_name))
    
    df_expr_bat_0.to_csv(os.path.join(output_dir, french_batch_bat_0))
    df_expr_pls_0.to_csv(os.path.join(output_dir, french_batch_pls_0))
    df_expr_rbe_0.to_csv(os.path.join(output_dir, french_batch_rbe_0))
    df_expr_svd_0.to_csv(os.path.join(output_dir, french_batch_svd_0))
    df_expr_bmc_0.to_csv(os.path.join(output_dir, french_batch_bmc_0))
    
    df_expr_bat_10.to_csv(os.path.join(output_dir, french_batch_bat_10))
    df_expr_pls_10.to_csv(os.path.join(output_dir, french_batch_pls_10))
    df_expr_rbe_10.to_csv(os.path.join(output_dir, french_batch_rbe_10))
    df_expr_svd_10.to_csv(os.path.join(output_dir, french_batch_svd_10))
    df_expr_bmc_10.to_csv(os.path.join(output_dir, french_batch_bmc_10))
    
    df_expr_bat_20.to_csv(os.path.join(output_dir, french_batch_bat_20))
    df_expr_pls_20.to_csv(os.path.join(output_dir, french_batch_pls_20))
    df_expr_rbe_20.to_csv(os.path.join(output_dir, french_batch_rbe_20))
    df_expr_svd_20.to_csv(os.path.join(output_dir, french_batch_svd_20))
    df_expr_bmc_20.to_csv(os.path.join(output_dir, french_batch_bmc_20))
    
    df_func_bat_0.to_csv(os.path.join(output_dir, french_batch_func_bat_0))
    df_func_pls_0.to_csv(os.path.join(output_dir, french_batch_func_pls_0))
    df_func_rbe_0.to_csv(os.path.join(output_dir, french_batch_func_rbe_0))
    # df_func_svd_0.to_csv(os.path.join(output_dir, french_batch_func_svd_0))
    df_func_bmc_0.to_csv(os.path.join(output_dir, french_batch_func_bmc_0))
    
    df_func_bat_10.to_csv(os.path.join(output_dir, french_batch_func_bat_10))
    df_func_pls_10.to_csv(os.path.join(output_dir, french_batch_func_pls_10))
    df_func_rbe_10.to_csv(os.path.join(output_dir, french_batch_func_rbe_10))
    df_func_svd_10.to_csv(os.path.join(output_dir, french_batch_func_svd_10))
    df_func_bmc_10.to_csv(os.path.join(output_dir, french_batch_func_bmc_10))
    
    df_func_bat_20.to_csv(os.path.join(output_dir, french_batch_func_bat_20))
    df_func_pls_20.to_csv(os.path.join(output_dir, french_batch_func_pls_20))
    df_func_rbe_20.to_csv(os.path.join(output_dir, french_batch_func_rbe_20))
    # df_func_svd_20.to_csv(os.path.join(output_dir, french_batch_func_svd_20))
    # df_func_bmc_20.to_csv(os.path.join(output_dir, french_batch_func_bmc_20))
    
    df_french_bat_0.to_csv(os.path.join(output_dir, french_batch_comb_bat_0))
    df_french_pls_0.to_csv(os.path.join(output_dir, french_batch_comb_pls_0))
    df_french_rbe_0.to_csv(os.path.join(output_dir, french_batch_comb_rbe_0))
    
    df_french_bat_10.to_csv(os.path.join(output_dir, french_batch_comb_bat_10))
    df_french_pls_10.to_csv(os.path.join(output_dir, french_batch_comb_pls_10))
    df_french_rbe_10.to_csv(os.path.join(output_dir, french_batch_comb_rbe_10))
    
    df_french_bat_20.to_csv(os.path.join(output_dir, french_batch_comb_bat_20))
    df_french_pls_20.to_csv(os.path.join(output_dir, french_batch_comb_pls_20))
    df_french_rbe_20.to_csv(os.path.join(output_dir, french_batch_comb_rbe_20))
    
    
    df_taxa_counts.to_csv(os.path.join(output_dir, taxa_counts))
    df_func_counts.to_csv(os.path.join(output_dir, func_counts))
    df_taxa_func_counts.to_csv(os.path.join(output_dir, taxa_func_counts))


if __name__ == "__main__":
    prep_data()