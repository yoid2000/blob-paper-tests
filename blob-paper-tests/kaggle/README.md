Contains the code needed to download kaggle files.

The kaggle files and other metadata are placed in directory blob_tests

By default, blob_tests is local. If you want it to be someplace else, you need to set the following environment BLOB_TEST_DIR to the path to the parent directory where you want blob_tests to be.

The datasets collected by kaggle are placed in blob_tests/datasets/kaggle_parquet. We have pre-populated that directory with a small number of datasets.

get_tables.py retrieves the tables. You will need to setup a kaggle authorization code to use this. See kaggle documentation. You can modify parameters inside get_tables.py to change the number of datasets collected, as well as filter out the datasets you don't want (by number of rows and columns).

change_to_datetime.py reads in all of the collected datasets, modifies the columns that look like datetime columns to dataframe datetime type. The so-modified .parquet files are placed in blob_tests/datasets/kaggle_parquet_out. They are put here so that you can make a manual check that everything looks kosher. If so, you can move them over to kaggle_parquet.

get_stats.py reads in all of the collected datasets, and generates the file blob_tests/datesets_info.json. This contains the kaggle name for each dataset along with the number of rows and columns. It also contains general statistics about the data. It also generates two scatterplots showing the number of rows and columns per dataset. These files are pre-loaded with the 1000 datasets used for the blob paper.

You can ignore check_zips.py.

