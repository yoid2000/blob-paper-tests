import pandas as pd
import time
import numpy as np
import warnings
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, StandardScaler
from sklearn.metrics.cluster import entropy
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr, spearmanr, chi2_contingency, ks_2samp
import os
os.environ['NUMBA_DISABLE_JIT'] = '1'
from dcor import distance_correlation


class StatTests:
    test_types = {
        'mutual_information': {'type': 'mixed', 'routine': '_compute_mutual_information'},
        'pearson_correlation': {'type': 'continuous', 'routine': '_compute_pearson_correlation'},
        'spearman_rank_correlations': {'type': 'mixed', 'routine': '_compute_spearman_rank_correlations'},
        'distance_correlation': {'type': 'mixed', 'routine': '_compute_distance_correlation'},
        'chi_square': {'type': 'categorical', 'routine': '_compute_chi_square'},
        'kolmogorov_smirnov': {'type': 'continuous', 'routine': '_compute_kolmogorov_smirnov'},
        'linear_regression': {'type': 'continuous', 'routine': '_compute_linear_regression'},
        'decision_tree_regressor': {'type': 'mixed', 'routine': '_compute_decision_tree_regressor'},
        'random_forest_regressor': {'type': 'mixed', 'routine': '_compute_random_forest_regressor'},
        'gradient_boosting_regressor': {'type': 'mixed', 'routine': '_compute_gradient_boosting_regressor'},
        'support_vector_regressor': {'type': 'continuous', 'routine': '_compute_support_vector_regressor'},
        'decision_tree_classifier': {'type': 'mixed', 'routine': '_compute_decision_tree_classifier'},
        'logistic_regression': {'type': 'categorical', 'routine': '_compute_logistic_regression'},
        'gradient_boosting_classifier': {'type': 'mixed', 'routine': '_compute_gradient_boosting_classifier'},
        'random_forest_classifier': {'type': 'mixed', 'routine': '_compute_random_forest_classifier'},
    }

    def __init__(self, df_in, col1, col2):
        # df_in is the 2dim synthetic dataframe that we are measuring
        # col1 and col2 are the names of the columns we are measuring
        self.columns = [col1, col2]
        self.col1 = col1
        self.col2 = col2
        self.initial_bins = 50
        self.fallback_bins = 10
        self.fms = {
            'elapsed_time': None,
            'mode_freq_col1': None,
            'mode_freq_col2': None,
            'n_uniques_col1': None,
            'n_uniques_col2': None,
            'n_uniques_both': None,
            'perfect_dependence': False,
            'perfect_hierarchy': None,
            'forced': None,
            'chi_square': None,
            'spearman': None,
            'AMI': None,     # adjusted mutual information
            'DC': None,      # distance correlation
            'reverse_spearman_col1': None,
            'reverse_spearman_col2': None,
        }
        self.df = df_in[[col1, col2]].dropna()
        self.col_processed = [None, None]
        self.col_discretized = [None, None]
        self.col_types = [None, None]
        for col_i, col in enumerate(self.columns):
            self.col_processed[col_i], self.col_types[col_i] = self._preprocess_column(self.df[col])

    def get_full_measure_stats(self):
        return self.fms

    def run_full_measure(self):
        start_time = time.time()
        self.fms['mode_freq_col1'] = self.df[self.col1].value_counts(normalize=True).max()
        self.fms['mode_freq_col2'] = self.df[self.col2].value_counts(normalize=True).max()
        self.fms['n_uniques_col1'] = len(self.df[self.col1].unique())
        self.fms['n_uniques_col2'] = len(self.df[self.col2].unique())
        self.fms['n_uniques_both'] = len(self.df[[self.col1, self.col2]].drop_duplicates())
        if self.fms['n_uniques_col1'] == 1 or self.fms['n_uniques_col2'] == 1:
            self.fms['forced'] = 0.0
            return 0.0
        result = self.run_stat_test('chi_square')
        self.fms['chi_square'] = result['score']
        result = self.run_stat_test('spearman_rank_correlations')
        self.fms['spearman'] = result['score']
        result = self.run_stat_test('mutual_information')
        self.fms['AMI'] = result['score']
        result = self.run_stat_test('distance_correlation')
        self.fms['DC'] = result['score']
        for col_i, col in enumerate(self.columns):
            df_new = self._scramble_column(col)
            st = StatTests(df_new, self.col1, self.col2)
            result = st.run_stat_test('spearman_rank_correlations')
            if col_i == 0:
                self.fms['reverse_spearman_col1'] = result['score']
            else:
                self.fms['reverse_spearman_col2'] = result['score']
        if self._perfect_dependence():
            # with perfect dependence, it is only necessary to synthesize one of the
            # two columns. The other can always be patched in from the 2dim subtable
            self.fms['perfect_dependence'] = True
        elif (self.fms['n_uniques_col1'] == self.fms['n_uniques_both'] or
            self.fms['n_uniques_col2'] == self.fms['n_uniques_both']):
            # The two columns may be hierarchically related
            hierarchy = self._perfect_hierarchy()
            if hierarchy is not None:
                self.fms['perfect_hierarchy'] = hierarchy
        end_time = time.time()
        self.fms['elapsed_time'] = end_time - start_time
        return None

    def run_stat_test(self, test):
        if test not in self.test_types:
            raise ValueError('Invalid test name')
        start_time = time.time()
        routine_name = self.test_types[test]['routine']
        routine = getattr(self, routine_name)
        score, other_names, other_vals = routine()
        end_time = time.time()
        elapsed_time = end_time - start_time
        return {'score': score, 'elapsed_time': elapsed_time, 'other_names': other_names, 'other_vals': other_vals}


    def _perfect_hierarchy(self):
        if not ((self.fms['n_uniques_col1'] == self.fms['n_uniques_both'] or
                self.fms['n_uniques_col2'] == self.fms['n_uniques_both']) and
                self.fms['n_uniques_col1'] != self.fms['n_uniques_col2'] and
                self.fms['n_uniques_both'] < (len(self.df)/10)):
            return None
        # Check if every distinct value in col1 is always paired with a given value in col2
        col1_to_col2 = self.df.groupby(self.col1)[self.col2].nunique()
        if all(col1_to_col2 == 1):
            return 'col2_on_col1'
        # Check if every distinct value in col2 is always paired with a given value in col1
        col2_to_col1 = self.df.groupby(self.col2)[self.col1].nunique()
        if all(col2_to_col1 == 1):
            return 'col1_on_col2'
        return None

    def _perfect_dependence(self):
        if not (self.fms['n_uniques_col1'] == self.fms['n_uniques_col2'] and
                self.fms['n_uniques_col1'] == self.fms['n_uniques_both'] and
                self.fms['n_uniques_col1'] < (len(self.df)/10)):
            # It is possible that the two columns are perfectly dependent even if the
            # number of distinct values is greater than 10% of the rows, but we ignore
            # this possibility
            return False
        # Check if every distinct value in col1 is always paired with a given value in col2
        col1_to_col2 = self.df.groupby(self.col1)[self.col2].nunique()
        if not all(col1_to_col2 == 1):
            return False
        # Check if every distinct value in col2 is always paired with a given value in col1
        col2_to_col1 = self.df.groupby(self.col2)[self.col1].nunique()
        if not all(col2_to_col1 == 1):
            return False
        return True

    def _scramble_column(self, col):
        # make a copy of self.df
        df_copy = self.df.copy()
        distinct_vals = df_copy[col].unique().tolist()
        # shuffle the values in distinct_vals
        np.random.shuffle(distinct_vals)
        # loop through distinct_valls two entries at a time
        for i in range(0, len(distinct_vals)-1, 2):
            val0 = distinct_vals[i]
            val1 = distinct_vals[i+1]
            # replace val0 with val1 and val1 with val0
            df_copy[col] = df_copy[col].replace({val0: val1, val1: val0})
        return df_copy

    def _discretize_with_fallback(self, col_processed):
        ''' I want to discretize with quantile, but it can fail when bins are too small, so we
            fallback to a uniform strategy.
        '''
        def discretize(col, n_bins, strategy):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                # Attempt discretization with the specified strategy
                discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
                col_discretized = pd.Series(discretizer.fit_transform(col.values.reshape(-1, 1)).flatten())

                # Check if the specific warning was raised
                if any("Bins whose width are too small" in str(warning.message) for warning in w):
                    raise ValueError("Bins too small")

                return col_discretized

        try:
            # Try quantile strategy with initial_bins
            return discretize(col_processed, self.initial_bins, 'quantile')
        except ValueError:
            try:
                # If quantile strategy with initial_bins fails, try quantile strategy with fallback_bins
                return discretize(col_processed, self.fallback_bins, 'quantile')
            except ValueError:
                # If quantile strategy with fallback_bins also fails, scale the data and use uniform strategy with fallback_bins
                scaler = StandardScaler()
                col_scaled = scaler.fit_transform(col_processed.values.reshape(-1, 1))
                return discretize(pd.Series(col_scaled.flatten()), self.fallback_bins, 'uniform')

    def _discretize_columns(self):
        for col_i, _ in enumerate(self.columns):
            if self.col_types[col_i] == 'numeric':
                if self.col_processed[col_i].nunique() >= self.initial_bins:
                    self.col_discretized[col_i] = self._discretize_with_fallback(self.col_processed[col_i])
                else:
                    # Assign a distinct integer to each unique value
                    label_encoder = LabelEncoder()
                    self.col_discretized[col_i] = pd.Series(label_encoder.fit_transform(self.col_processed[col_i]))
            else:
                self.col_discretized[col_i] = self.col_processed[col_i]

    def _preprocess_column(self, column):
        if pd.api.types.is_numeric_dtype(column):
            return column, 'numeric'
        elif pd.api.types.is_datetime64_any_dtype(column):
            return column.astype('int64'), 'numeric'
        else:
            le = LabelEncoder()
            return pd.Series(le.fit_transform(column), index=column.index), 'text'

    def _compute_pearson_correlation(self):
        score, _ = pearsonr(self.col_processed[0], self.col_processed[1])
        normalized_score = abs(score)  # Normalize to [0, 1] by taking the absolute value
        return round(float(normalized_score), 3), None, None

    def _compute_spearman_rank_correlations(self):
        score, _ = spearmanr(self.col_processed[0], self.col_processed[1])
        normalized_score = abs(score)  # Normalize to [0, 1] by taking the absolute value
        return round(float(normalized_score), 3), None, None

    def _compute_distance_correlation(self):
       # Standardize the data to reduce numerical instability
        scaler = StandardScaler()
        col1_scaled = scaler.fit_transform(self.col_processed[0].values.reshape(-1, 1)).flatten()
        col2_scaled = scaler.fit_transform(self.col_processed[1].values.reshape(-1, 1)).flatten()
        score = distance_correlation(col1_scaled, col2_scaled)
        return round(float(score), 3), None, None

    def _cramers_v(self, chi2, n, min_dim):
        return np.sqrt(chi2 / (n * (min_dim - 1)))

    def _compute_chi_square(self):
        # Discretize columns if they are continuous
        self._discretize_columns()
        # Create a contingency table
        contingency_table = pd.crosstab(self.col_discretized[0], self.col_discretized[1])
        # Perform the chi-square test
        chi2, p, _, _ = chi2_contingency(contingency_table)

        # Calculate Cramér's V
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape)
        cramers_v_score = self._cramers_v(chi2, n, min_dim)
        
        # Normalize the score using Cramér's V
        normalized_score = cramers_v_score  # Use Cramér's V to determine the strength of dependence
        
        return round(float(normalized_score), 3), None, None

    def _compute_kolmogorov_smirnov(self):
        score, p_value = ks_2samp(self.col_processed[0], self.col_processed[1])
        normalized_score = 1 - float(score)  # Normalize to [0, 1]
        confidence_score = 1 - float(p_value)
        return normalized_score, confidence_score, None

    def _compute_linear_regression(self):
        # Model 1: Predict col2 from col1
        model1 = LinearRegression()
        model1.fit(self.col_processed[0].values.reshape(-1, 1), self.col_processed[1])
        predictions1 = model1.predict(self.col_processed[0].values.reshape(-1, 1))
        score1 = r2_score(self.col_processed[1], predictions1)

        # Model 2: Predict col1 from col2
        model2 = LinearRegression()
        model2.fit(self.col_processed[1].values.reshape(-1, 1), self.col_processed[0])
        predictions2 = model2.predict(self.col_processed[1].values.reshape(-1, 1))
        score2 = r2_score(self.col_processed[0], predictions2)

        # Return the best score and normalize to [0, 1]
        best_score = max(score1, score2)
        return round(best_score, 3), None, None

    def _compute_regressor(self, model):
        # Model 1: Predict col2 from col1
        model1 = model()
        scores1 = cross_val_score(model1, self.col_processed[0].values.reshape(-1, 1), self.col_processed[1], cv=5, scoring='r2')
        score1 = np.mean(scores1)

        # Model 2: Predict col1 from col2
        model2 = model()
        scores2 = cross_val_score(model2, self.col_processed[1].values.reshape(-1, 1), self.col_processed[0], cv=5, scoring='r2')
        score2 = np.mean(scores2)

        # Determine baseline for col2
        dummy_reg1 = DummyRegressor(strategy='mean')
        baseline_scores1 = cross_val_score(dummy_reg1, self.col_processed[0].values.reshape(-1, 1), self.col_processed[1], cv=5, scoring='r2')
        baseline_score1 = np.mean(baseline_scores1)

        # Determine baseline for col1
        dummy_reg2 = DummyRegressor(strategy='mean')
        baseline_scores2 = cross_val_score(dummy_reg2, self.col_processed[1].values.reshape(-1, 1), self.col_processed[0], cv=5, scoring='r2')
        baseline_score2 = np.mean(baseline_scores2)

        # Normalize scores relative to the baseline
        normalized_score1 = (score1 - baseline_score1) / (1 - baseline_score1) if score1 > baseline_score1 else 0.0
        normalized_score2 = (score2 - baseline_score2) / (1 - baseline_score2) if score2 > baseline_score2 else 0.0

        # Return the best normalized score
        best_normalized_score = max(normalized_score1, normalized_score2)
        return round(float(best_normalized_score), 3), None, None

    def _compute_decision_tree_regressor(self):
        return self._compute_regressor(DecisionTreeRegressor)

    def _compute_random_forest_regressor(self):
        return self._compute_regressor(RandomForestRegressor)

    def _compute_gradient_boosting_regressor(self):
        return self._compute_regressor(GradientBoostingRegressor)

    def _compute_support_vector_regressor(self):
        return self._compute_regressor(SVR)

    def _compute_classifier(self, model):
        # Discretize columns if they are continuous
        self._discretize_columns()

        # Model 1: Predict col2 from col1
        model1 = model()
        model1.fit(self.col_discretized[0].values.reshape(-1, 1), self.col_discretized[1])
        predictions1 = model1.predict(self.col_discretized[0].values.reshape(-1, 1))
        score1 = accuracy_score(self.col_discretized[1], predictions1)

        # Model 2: Predict col1 from col2
        model2 = model()
        model2.fit(self.col_discretized[1].values.reshape(-1, 1), self.col_discretized[0])
        predictions2 = model2.predict(self.col_discretized[1].values.reshape(-1, 1))
        score2 = accuracy_score(self.col_discretized[0], predictions2)

        # Determine baseline for col2
        dummy_clf1 = DummyClassifier(strategy='most_frequent')
        dummy_clf1.fit(self.col_discretized[0].values.reshape(-1, 1), self.col_discretized[1])
        dummy_predictions1 = dummy_clf1.predict(self.col_discretized[0].values.reshape(-1, 1))
        baseline_score1 = accuracy_score(self.col_discretized[1], dummy_predictions1)

        # Determine baseline for col1
        dummy_clf2 = DummyClassifier(strategy='most_frequent')
        dummy_clf2.fit(self.col_discretized[1].values.reshape(-1, 1), self.col_discretized[0])
        dummy_predictions2 = dummy_clf2.predict(self.col_discretized[1].values.reshape(-1, 1))
        baseline_score2 = accuracy_score(self.col_discretized[0], dummy_predictions2)

        # Normalize scores relative to the baseline
        normalized_score1 = (score1 - baseline_score1) / (1 - baseline_score1) if score1 > baseline_score1 else 0.0
        normalized_score2 = (score2 - baseline_score2) / (1 - baseline_score2) if score2 > baseline_score2 else 0.0

        # Return the best normalized score
        best_normalized_score = max(normalized_score1, normalized_score2)
        return round(float(best_normalized_score), 3), None, None

    def _compute_decision_tree_classifier(self):
        return self._compute_classifier(DecisionTreeClassifier)

    def _compute_logistic_regression(self):
        return self._compute_classifier(LogisticRegression)

    def _compute_gradient_boosting_classifier(self):
        return self._compute_classifier(GradientBoostingClassifier)

    def _compute_random_forest_classifier(self):
        return self._compute_classifier(RandomForestClassifier)

    def _compute_mutual_information(self):
        self._discretize_columns()

        # Compute adjusted mutual information
        ami = adjusted_mutual_info_score(self.col_discretized[0], self.col_discretized[1])

        return round(float(ami), 3), None, None