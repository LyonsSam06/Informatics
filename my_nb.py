from mixed_naive_bayes import MixedNB
import numpy as np
import pandas as pd
import copy

class NaiveBayes:
    def __init__(self, class_info: tuple[str, list[str]], feature_info: dict[str, list[str]], categorical_columns: list[int]):
        self.class_info = class_info # (class_name, permitted_values)
        self.feature_info = feature_info
        self.inner_nb = MixedNB(categorical_features=categorical_columns)

    def fit(self, x: pd.DataFrame, y: pd.Series):
        # FIX: Use the passed x and y instead of 'training_data'
        # The library expects numpy arrays
        features_train = x.values
        class_train = y.values
        self.inner_nb.fit(features_train, class_train)

    def predict(self, testing_data: pd.DataFrame) -> pd.DataFrame:
        class_name = self.class_info[0]
        # Drop class column if present to get features only
        features_test = testing_data.drop(class_name, axis=1, errors='ignore').values
        
        predicted_class = self.inner_nb.predict(features_test)
        
        classified_data = testing_data.copy()
        classified_data['PredictedClass'] = predicted_class
        return classified_data

    def retrieve_class_probability(self, class_value: str) -> float:
        # Find the index of the class value from the model's classes_ attribute
        class_index = np.where(self.inner_nb.classes_ == class_value)[0][0]
        # MixedNB stores log priors in 'class_log_prior_'
        return np.exp(self.inner_nb.class_log_prior_[class_index])

    def retrieve_conditional_probability(self, class_value: str, feature_index: int, feature_value_index: int) -> float:
        class_index = np.where(self.inner_nb.classes_ == class_value)[0][0]
        # Access the categorical distributions; MixedNB uses 'category_count_' for counts
        # To get probability: count of value / total count for that class
        feature_counts = self.inner_nb.category_count_[feature_index]
        class_total = np.sum(feature_counts[class_index])
        return feature_counts[class_index, feature_value_index] / class_total