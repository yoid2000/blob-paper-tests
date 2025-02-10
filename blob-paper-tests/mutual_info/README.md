
`compute_mutual_info.py` computes the mutual information measure for every pair of columns in one dataset.

It places the result of each individual dataset in `results/mutual`. 

It is designed to work with SLURM.

`plot_mutual_info.py` reads in the results in `results/mutual` and produces two plots that compare the 
mutual information score for the original data versus the synthetic data:

`mutual_info_plot.png` orders the per-pair results by original score, and then plots both the original and
synthetic scores. We can see from this that the synthetic scores deviate markedly. One reason for this is
that mutual information can give a high score based on a very small number of perfectly aligned values in
the two columns. SynDiffix suppresses these values, and therefore removes the dependence.

`err_nunique.png` is really just for the purpose of ensuring that there is nothing strange in the number
of unique values produced by SynDiffix that would account for the large deviation.

In any event, based on these results, we decided to look at a lot more measures, which can be found in
`all_stat_tests`.
