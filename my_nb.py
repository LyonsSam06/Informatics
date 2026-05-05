from mixed_naive_bayes import MixedNB
import numpy as np
import pandas as pd
import copy
from sklearn.preprocessing import LabelEncoder

#needs to be passes categorical_columns now (list of indicies of the categorical features)   = [1, 2, 5, 6, 8, 10, 12]
class NaiveBayes:
    def __init__(self, class_info: tuple[str, list[str]], feature_info: dict[str, list[str]], categorical_columns: list[int]):
        self.class_info = class_info # (class_name, permitted_values)
        self.feature_info = feature_info
        self.inner_nb = MixedNB(categorical_features=categorical_columns)

    def fit(self, training_data: pd.DataFrame):
        #need to turn df into a 2d list for MixedNB to use it
        data_copy = training_data.copy()
        label_encoder = LabelEncoder()
        
        #need to encode the text based categorical features
        columns_to_encode = [1, 2, 6, 10, 12] 
        for col in columns_to_encode:
            col_name = data_copy.columns[col]
            data_copy[col_name] = label_encoder.fit_transform(data_copy[col_name])
        #split the data into 2.
        # x is a 2D list of all the values
        # y is a list of just the class column

        #first split them
        x_df = data_copy.iloc[:, :-1]
        y_df = data_copy.iloc[:, -1]
        # for MixedNB's 'fit' to work we need to assign numeric values to the target column
        y_numbered = label_encoder.fit_transform(y_df)

        #fit only takes list so chnage to right format
        x = x_df.values.tolist()
        y = y_numbered.tolist() 
        self.inner_nb.fit(x, y)

    def predict(self, testing_data: pd.DataFrame) -> pd.DataFrame:
        data_copy = testing_data.copy()
        #we encoded the target field so need to call it back to unconvert them
        label_encoder = LabelEncoder()
        #encode the values for the model again
        columns_to_encode = [1, 2, 6, 10, 12]
        for col in columns_to_encode:
            col_name = data_copy.columns[col]
            data_copy[col_name] = label_encoder.fit_transform(data_copy[col_name])
        x_test = data_copy.iloc[:, :-1].values.tolist()
        #getting the number representations of the predicted classes
        numeric_predictions = self.inner_nb.predict(x_test)
        #make the key
        class_labels = self.class_info[1]
        #convert back and send
        string_predictions = [class_labels[int(i)] for i in numeric_predictions]
        classified_data = testing_data.copy()
        classified_data['PredictedClass'] = string_predictions
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