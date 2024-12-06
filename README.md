# The bigger the better? A critical view on multi-cohort microbiome analysis in colorectal cancer

This git repository contains all code used in the analysis. 

## Instructions
### Running the experiments
By running all cells of `central_analysis.ipynb` or `central_analysis.py`, you will be able to reproduce the results of the paper. 

### Including custom data
To include custom data, place it in the `prep_data/` directory, add the paths to the data in the central_analysis file and add a `'experiment'` to the `'experiments'` dictionary. The experiment should include the count files, the metadata file and the name of the experiment.

### Rerunning the experiments
Results are places in the `data/simulations/` directory. By default, the script will not overwrite existing results. To overwrite existing results, set EXEC_MODE variable to `'redo'` (default `'load'`) or delete the existing results from the simulations directory.

### Runtime
The script will try to automatically determine, if it is executed on a machine with sufficient hardware. If no sufficient hardware is present, it only repeats the experiment once and only executes the first 2 experiments. To disable this, remove the `if_server()` checks. 

Running an experiment on a normal office machine should take around **50 minutes** per repeat. 

### Telegram inclusion
As a convenience feature, the scripts progress bar can sync to a telegram bot. To use this feature, follow [this](https://tqdm.github.io/docs/contrib.telegram/) guide from tqdm.

## Source data
The dat included in this repository to reproduce the results is based on the following sources downloaded from the European Nucleotide Archive: PRJDB4176 (Yachida et al. 2019), PRJEB10878 (Yu et al. 2017), PRJEB12449 (Vogtmann et al. 2016), PRJEB27928 (Wirbel et al. 2019), FPRJEB6070 (Zeller et al. 2014), PRJEB7774 (Feng et al. 2015), RJNA389927 (Hannigan et al. 2018), PRJNA397112 (Dhakan et al. 2019), PRJNA447983 (Thomas et al. 2019), PRJNA531273 (Gupta et al. 2019), PRJNA608088 (Chang et al. 2021), PRJNA429097 (J. Yang et al. 2020), PRJNA731589 (Y. Yang et al. 2021), PRJNA763023 (Liu et al. 2022), and PRJNA961076 (Grion et al. 2023).