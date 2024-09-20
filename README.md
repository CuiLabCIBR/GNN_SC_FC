# GNN_SC_FC_Coupling

Data and codes for our paper ***Group-common and individual-specific effects of structure-function coupling in human brain networks with graph neural network***, including

* Training and validating the proposed Graph neural network(GNN) framework,

* Connectome matrices of  *group-averaged structural (SC), functional (FC), predicted functional connectivity (PredFC)* matrices,  

* Group and individual effects of the regional structure-function coupling, 

for the **HCP-D** and **HCP-YA** (typically referred as **'HCP'**) datasets using `Schaefer-400` (7 Networks order). 

 See [data](data/) for more details. 

## Dependencies

The graph neural network is implemented in Pytorch  for CUDA 1.31.1 (https://pytorch.org/) , pyTorch Geometric  for CUDA 2.2.0 (https://pytorch-geometric.readthedocs.io/en/latest/) at Python 3.10. Scikit-learn (http://scikit-learn.org/stable/), Scipy (https://scipy.org/), and Brainconnectivity toolbox (https://github.com/aestrivex/bctpy) are also necessary for model training and testing.

The visualization is implemented by Python 3.10, Matlab 2023a, and R 4.2.1. 

## `data`
- The [sub_info](data/info/sub_info) folder contains all the subject ID in this study for both datasets as well as the subject ID of the samples in the testing set.
- The [info](data/info) folder contains the Sensorimotor-association axis ranks and the spin test null distributions for Schaefer 200 and 400.
- The [SC_FC_PredFC_matrix](data/SC_FC_PredFC_matrix) folder contains group-averaged SC, FC, PredFC matrices.
- The [preprocessed_data](data/preprocessed_data) folder contains the preprocessed SC, FC data (please put your SC, FC here in pickle)
- The [result_out](data/result_out) folder contains the regional group and individual effect.

## `functions`
The [functions](functions/) folder contains Matlab and Python functions commonly used in `code`.

## `code`

- The [step01_ train_val_test_GNN](step01_train_val_test_GNN/) folder contains our proposed graph neural network model (`model.py`), the process of training, validating, and testing the model(`train_val_test.py`). In our sensitivity test, we run  cross-validation(`cross_validation.py`) for 100 times to validate the robustness of our results. 

  * Run the graph neural network model by

  ```bash
  python train_val_test.py  --epochs 400 --batch-size 2 --lr 0.001 --layer-num 2 --conv-dim 256 --reg 0.0001
  ```

  * Train and validate the GNN model for hyperparameter tuning using `--if-kfold` for 5 fold cross-validation.

  * Take rewired SC for training by  `--rewired`.

  

- The [step02_get_SC_FC_coupling](step02_get_SC_FC_coupling/) folder contains codes to calculate the whole-brain structure-function coupling analysis and the null model test by rewiring SC and  plot *Fig.2*  and  *Fig.S1*.

  - Run `step_01_sc_fc_coupling_group_hcp.m` and `step_01_sc_fc_coupling_group_hcpd.m` to calculate structure-function coupling at the group level and plot the *Fig. S1*.
  - Run `step_02_density_plot.R` to plot the density plot at *Fig2a-d*. 
  - Run `step_03_sc_fc_coupling_rew_ind_hcp.m` amd `step_03_sc_fc_coupling_rew_ind_hcpd.m` to calculate the structure-function coupling at the individual level for the actual and the rewired conditions.
  - Run `step_04_violin_scatter_boxplot.R` to plot the *Fig2e&f*. 
  - Run `step_05_regional_sc_fc_coupling_lin.m` and `step_05_regional_sc_fc_coupling_GNN.m` to calculate the regional structure-function coupling.
  - Run `step_06_surfaceplot_cp_linear.R` and `step_06_surfaceplot_cp_GNN` to plot the  *Fig2g-j*. 

  

- The [step_03_whole_brain_group_ind_effects](step_03_whole_brain_group_ind_effects/) folder contains codes to calculate the whole-brain group-common and individual-specific effects of the structure-function coupling and plot *Fig. 3* and *Fig.S5a*.

  * Run `step_01_whole_brain_group_ind_effects_linear.m` and `step_01_whole_brain_group_ind_effects_GNN.m` to calculate the group and individual effects for the whole brain.
  * Run  `step_02_barplot_group_linear_ind.py` and  `step_02_barplot_group_GNN_ind.py`  to plot the *Fig.3b&c*. 

  * Run  `step_03_split_violin_linear.py` and  `step_03_split_violin_GNN.py` to plot the *Fig.3e&f*. 

  

  Similar procedures are implemented for the schaefer200 parcellation in the sensitivity test for *Fig.S3a* by runing `step_04_whole_brain_group_ind_effects_GNN_schaefer200.m`, `step_04_pFC_eFC_group_ind_schaefer200.py`, and `step_04_split_violin_GNN_schaefer200.py` to get the whole-brain group and individual effects.

  

- The [step_04_regional_group_ind_effects](step_04_regional_group_ind_effects/) folder contains codes to calculate the whole-brain group-common and individual-specific effects of the structure-function coupling and their relationship with the sensorimotor-association axis, and plot *Fig.4*, *Fig.5*,  *Fig.S4* and  *Fig.S5b-g*. 

  * Run `step_01_regional_group_ind_effects_GNN.m` to calculate the regional group and individual effects at the regional level.

  * Run `step_02_surfaceplot_sa_rank.R` to plot the *Fig.4a*.

  * Run `step_02_surfaceplot_group_ind_effect_GNN.R` to plot the *Fig.4b&c* and *Fig. 5b&c*.

  * Run `step_02_scatterplot_group_ind_effect_sa_rank_GNN.R` to plot the *Fig. 4e&f* and *Fig. 5e&f*.

  * For the linear model, run `step_03_regional_group_ind_effects_linear.m` and `step_03_scatterplot_group_ind_effect_sa_rank_linear.R` to plot the *Fig.S3*.

    

  Similar procedures are implemented for the sensitivity test.

  * For the normalized individual effects, run `step_04_surfaceplot_norm_ind_effect_single.R` and `step_04_surfaceplot_norm_ind_effect_single.R` to plot the *Fig.S3*.
  * For the schaefer200 parcellation, run `step_05_regional_group_ind_effects_GNN_schaefer200.m`, `step_05_surfaceplot_sa_rank_GNN_schaefer200.R`, and  `step_05_surfaceplot_sa_rank_GNN_schaefer200.R` plot the *Fig.S4b-g*.
