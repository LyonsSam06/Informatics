import pandas as pd


# Simple function stub for balancing the dataset according to a particular feature.
# Modify as you see fit, including the function name/template, etc!

def balance_dataset(dataset: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    """
    designed to work with any number of values to allow application in multiple features with differing numbers of values

    for duplcates a random value from the smaller groups x times, where x is how many fewer there are
    adds these to a NEW dataframe not the same one to prevent a single value being duplicated repeatedly as it takes up more of the pool
    """

    #define the value to stop duplicating at
    vc = dataset.value_counts(feature_name)
    goal = vc.max()

    #define copy of dataset
    new_dataset = dataset.copy()

    for k in vc.keys():
        for _ in range(goal-vc[k]):
            #kinda long line so commenting for future me
            # gets a sample (random row) from the sliced dataset so that the key matches the feature value and concats that with the dataset
            # one at a time is very ineffici
            new_dataset = pd.concat([new_dataset, dataset[dataset[feature_name] == k].sample()])

    #debug
    # print(vc)
    # print(new_dataset.value_counts(feature_name))

    return new_dataset
