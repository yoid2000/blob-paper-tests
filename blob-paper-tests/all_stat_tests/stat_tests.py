import pandas as pd
import time
import numpy as np
import warnings
from sklearn.metrics import mutual_info_score
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
#import os
#os.environ['NUMBA_DISABLE_JIT'] = '1'
from dcor import distance_correlation


class StatTests:
    test_types = {
        'mutual_information': {'type': 'mixed', 'routine': '_compute_mutual_information'},
        'pearson_correlation': {'type': 'continuous', 'routine': '_compute_pearson_correlation'},
        'spearman_rank_correlations': {'type': 'mixed', 'routine': '_compute_spearman_rank_correlations'},
        'distance_correlation': {'type': 'continuous', 'routine': '_compute_distance_correlation'},
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
        self.df = df_in[[col1, col2]].dropna()
        self.col1_processed, self.col1_type = self._preprocess_column(self.df[col1])
        self.col2_processed, self.col2_type = self._preprocess_column(self.df[col2])

    def run_stat_test(self, test):
        if test not in self.test_types:
            raise ValueError('Invalid test name')
        if self.test_types[test]['type'] == 'continuous' and (self.col1_type == 'text' or self.col2_type == 'text'):
            return None
        start_time = time.time()
        routine_name = self.test_types[test]['routine']
        routine = getattr(self, routine_name)
        score, other_names, other_vals = routine()
        end_time = time.time()
        elapsed_time = end_time - start_time
        return {'score': score, 'elapsed_time': elapsed_time, 'other_names': other_names, 'other_vals': other_vals}

    def _discretize_with_fallback(self, col_processed, initial_bins=10):
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                # Attempt discretization with quantile strategy
                discretizer = KBinsDiscretizer(n_bins=initial_bins, encode='ordinal', strategy='quantile')
                col_discretized = pd.Series(discretizer.fit_transform(col_processed.values.reshape(-1, 1)).flatten())

                # Check if the specific warning was raised
                if any("Bins whose width are too small" in str(warning.message) for warning in w):
                    raise ValueError("Bins too small")

                return col_discretized

        except ValueError:
            # If quantile strategy fails, scale the data and use uniform strategy
            scaler = StandardScaler()
            col_scaled = scaler.fit_transform(col_processed.values.reshape(-1, 1))
            discretizer = KBinsDiscretizer(n_bins=initial_bins, encode='ordinal', strategy='uniform')
            col_discretized = pd.Series(discretizer.fit_transform(col_scaled).flatten())
            return col_discretized

    def _discretize_columns(self):
        # Possible strategy are 'uniform' and 'quantile'
        if self.col2_type == 'numeric' and self.col2_processed.nunique() >= 20:
            self.col2_discretized = self._discretize_with_fallback(self.col2_processed)
        else:
            self.col2_discretized = self.col2_processed

        if self.col1_type == 'numeric' and self.col1_processed.nunique() >= 20:
            self.col1_discretized = self._discretize_with_fallback(self.col1_processed)
        else:
            self.col1_discretized = self.col1_processed


    def _preprocess_column(self, column):
        if pd.api.types.is_numeric_dtype(column):
            return column, 'numeric'
        elif pd.api.types.is_datetime64_any_dtype(column):
            return column.astype('int64'), 'numeric'
        else:
            le = LabelEncoder()
            return pd.Series(le.fit_transform(column), index=column.index), 'text'

    def _compute_pearson_correlation(self):
        score, _ = pearsonr(self.col1_processed, self.col2_processed)
        normalized_score = abs(score)  # Normalize to [0, 1] by taking the absolute value
        return round(float(normalized_score), 3), None, None

    def _compute_spearman_rank_correlations(self):
        score, _ = spearmanr(self.col1_processed, self.col2_processed)
        normalized_score = abs(score)  # Normalize to [0, 1] by taking the absolute value
        return round(float(normalized_score), 3), None, None

    def _compute_distance_correlation(self):
        score = distance_correlation(self.col1_processed, self.col2_processed)
        return round(float(score), 3), None, None  # Distance correlation is already in [0, 1]

    def _cramers_v(self, chi2, n, min_dim):
        return np.sqrt(chi2 / (n * (min_dim - 1)))

    def _compute_chi_square(self):
        # Discretize columns if they are continuous
        self._discretize_columns()
        # Create a contingency table
        contingency_table = pd.crosstab(self.col1_discretized, self.col2_discretized)
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
        score, p_value = ks_2samp(self.col1_processed, self.col2_processed)
        normalized_score = 1 - float(score)  # Normalize to [0, 1]
        confidence_score = 1 - float(p_value)
        return normalized_score, confidence_score, None

    def _compute_linear_regression(self):
        # Model 1: Predict col2 from col1
        model1 = LinearRegression()
        model1.fit(self.col1_processed.values.reshape(-1, 1), self.col2_processed)
        predictions1 = model1.predict(self.col1_processed.values.reshape(-1, 1))
        score1 = r2_score(self.col2_processed, predictions1)

        # Model 2: Predict col1 from col2
        model2 = LinearRegression()
        model2.fit(self.col2_processed.values.reshape(-1, 1), self.col1_processed)
        predictions2 = model2.predict(self.col2_processed.values.reshape(-1, 1))
        score2 = r2_score(self.col1_processed, predictions2)

        # Return the best score and normalize to [0, 1]
        best_score = max(score1, score2)
        return round(best_score, 3), None, None

    def _compute_regressor(self, model):
        # Model 1: Predict col2 from col1
        model1 = model()
        scores1 = cross_val_score(model1, self.col1_processed.values.reshape(-1, 1), self.col2_processed, cv=5, scoring='r2')
        score1 = np.mean(scores1)

        # Model 2: Predict col1 from col2
        model2 = model()
        scores2 = cross_val_score(model2, self.col2_processed.values.reshape(-1, 1), self.col1_processed, cv=5, scoring='r2')
        score2 = np.mean(scores2)

        # Determine baseline for col2
        dummy_reg1 = DummyRegressor(strategy='mean')
        baseline_scores1 = cross_val_score(dummy_reg1, self.col1_processed.values.reshape(-1, 1), self.col2_processed, cv=5, scoring='r2')
        baseline_score1 = np.mean(baseline_scores1)

        # Determine baseline for col1
        dummy_reg2 = DummyRegressor(strategy='mean')
        baseline_scores2 = cross_val_score(dummy_reg2, self.col2_processed.values.reshape(-1, 1), self.col1_processed, cv=5, scoring='r2')
        baseline_score2 = np.mean(baseline_scores2)

        # Normalize scores relative to the baseline
        normalized_score1 = (score1 - baseline_score1) / (1 - baseline_score1) if score1 > baseline_score1 else 0.0
        normalized_score2 = (score2 - baseline_score2) / (1 - baseline_score2) if score2 > baseline_score2 else 0.0

        # Return the best normalized score
        best_normalized_score = max(normalized_score1, normalized_score2)
        return round(float(best_normalized_score), 3), None, None

    def _compute_regressor_old(self, model):
        # Model 1: Predict col2 from col1
        model1 = model()
        model1.fit(self.col1_processed.values.reshape(-1, 1), self.col2_processed)
        predictions1 = model1.predict(self.col1_processed.values.reshape(-1, 1))
        score1 = r2_score(self.col2_processed, predictions1)

        # Model 2: Predict col1 from col2
        model2 = model()
        model2.fit(self.col2_processed.values.reshape(-1, 1), self.col1_processed)
        predictions2 = model2.predict(self.col2_processed.values.reshape(-1, 1))
        score2 = r2_score(self.col1_processed, predictions2)

        # Determine baseline for col2
        dummy_reg1 = DummyRegressor(strategy='mean')
        dummy_reg1.fit(self.col1_processed.values.reshape(-1, 1), self.col2_processed)
        dummy_predictions1 = dummy_reg1.predict(self.col1_processed.values.reshape(-1, 1))
        baseline_score1 = r2_score(self.col2_processed, dummy_predictions1)

        # Determine baseline for col1
        dummy_reg2 = DummyRegressor(strategy='mean')
        dummy_reg2.fit(self.col2_processed.values.reshape(-1, 1), self.col1_processed)
        dummy_predictions2 = dummy_reg2.predict(self.col2_processed.values.reshape(-1, 1))
        baseline_score2 = r2_score(self.col1_processed, dummy_predictions2)

        # Normalize scores relative to the baseline
        normalized_score1 = (score1 - baseline_score1) / (1 - baseline_score1) if baseline_score1 != 1 else score1
        normalized_score2 = (score2 - baseline_score2) / (1 - baseline_score2) if baseline_score2 != 1 else score2

        # Return the best normalized score
        best_normalized_score = max(normalized_score1, normalized_score2)
        return best_normalized_score, None, None

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
        model1.fit(self.col1_discretized.values.reshape(-1, 1), self.col2_discretized)
        predictions1 = model1.predict(self.col1_discretized.values.reshape(-1, 1))
        score1 = accuracy_score(self.col2_discretized, predictions1)

        # Model 2: Predict col1 from col2
        model2 = model()
        model2.fit(self.col2_discretized.values.reshape(-1, 1), self.col1_discretized)
        predictions2 = model2.predict(self.col2_discretized.values.reshape(-1, 1))
        score2 = accuracy_score(self.col1_discretized, predictions2)

        # Determine baseline for col2
        dummy_clf1 = DummyClassifier(strategy='most_frequent')
        dummy_clf1.fit(self.col1_discretized.values.reshape(-1, 1), self.col2_discretized)
        dummy_predictions1 = dummy_clf1.predict(self.col1_discretized.values.reshape(-1, 1))
        baseline_score1 = accuracy_score(self.col2_discretized, dummy_predictions1)

        # Determine baseline for col1
        dummy_clf2 = DummyClassifier(strategy='most_frequent')
        dummy_clf2.fit(self.col2_discretized.values.reshape(-1, 1), self.col1_discretized)
        dummy_predictions2 = dummy_clf2.predict(self.col2_discretized.values.reshape(-1, 1))
        baseline_score2 = accuracy_score(self.col1_discretized, dummy_predictions2)

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

        # Compute mutual information
        mi = mutual_info_score(self.col1_discretized, self.col2_discretized)

        # Compute entropy for normalization
        h_col1 = entropy(self.col1_discretized)
        h_col2 = entropy(self.col2_discretized)
        
        # Check for zero entropy to avoid division by zero
        if h_col1 == 0 or h_col2 == 0:
            return 0  # or handle this case as needed
            
        normalized_mi = mi / min(h_col1, h_col2)
        
        return round(float(normalized_mi), 3), ('entropy_col1', 'entropy_col2'), (h_col1, h_col2)