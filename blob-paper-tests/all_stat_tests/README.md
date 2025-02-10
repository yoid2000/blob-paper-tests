`stat_tests.py` contain all of the statistical and ML tests that I'm running. They are all normalized to return a score between 0 (no correlation/dependence) and 1 (perfect correlation/dependence).

`run_stat_test.py` creates simple 2-column dataframes with increasing amounts of randomness, and therefore decreasing amounts of correlation, and runs a series of statistical tests on them. It generates a variety of plots and puts them in `simple_results`. The dataframes contain combinations of categorical, integer continuous, and float continuous columns.

`basic_per_swap_frac.png` plots the amount of random swapping (more swapping means less dependence) against the score for all tests. It shows that as we would expect, more randomness means less dependence.

`basic_per_test.png` plots the difference between the given test's scores and the median score for each type of dataset (kindof here assuming that the median score represents some kind of "correct" score). It also plots the execution times for the different tests.

`media_diff.png` shows the difference from the median, but for different dataset types. 

`swap_by_test.png` shows how each test relates to the amount of randomness. 

Overall, this all shows that all the tests track well with the amount of randomness in the datasets. Haven't learned much here except that the tests all appear to do what they are supposed to do.


Note that logistic regression is primarily for binary variables and so I don't use it.