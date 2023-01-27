import pandas as pd

def read_data(datasets):
    """
    :param datasets: Takes list of strings
    :return: Returns read df
    """
    if datasets.count(True) != 1:
        raise ValueError("Select exactly one dataset")
    if datasets[0]:
        path = 'Data/toy_example.csv'
        separator = ';'
        df = pd.read_csv(path, sep=str(separator))
    if datasets[1]:
        path = 'Data/dataset_1.csv'
        separator = ';'
        df = pd.read_csv(path, sep=str(separator))
        raise TypeError("This dataset is not publicly available. Please select \"dataset_3\" in 1.Settings")
    if datasets[2]:
        path = 'Data/dataset_2.csv'
        separator = ';'
        df = pd.read_csv(path, sep=str(separator))
        raise TypeError("This dataset is not publicly available. Please select \"dataset_3\" in 1.Settings")
    if datasets[3]:
        path = 'Data/dataset_3.csv'
        separator = ';'
        df = pd.read_csv(path, sep=str(separator))
        df=df.rename(columns={'id':'case:concept:name','act':'concept:name'})

    # Surprise package works with score on 5, so objective needs to be rescaled by 5. Afterwards, downscaled again.
    df['objective'] = df['objective'].apply(lambda x: x * 5)

    return df

def rename_to_surprise_notation(df):
    df.rename(
        columns={
            'case:concept:name': 'userId',
            'concept:name':'movieId',
            'objective':'rating',
            'time_start':'timestamp'},
        inplace=True)
    return df

def spearman_surprise(pred):
    """
    :param pred: Takes the predictions by surprise package as input
    :return: Returns average spearman correlation
    """
    import math
    from scipy.stats import spearmanr
    from statistics import mean
    df1 = pd.DataFrame(pred, columns=['case:concept:name', 'concept:name', 'r_ui', 'objective', 'details'])

    # create an empty dictionary to store the lists
    result_dict = {}

    # group the dataframe by 'concept:name'
    grouped_df = df1.groupby('concept:name')

    # iterate over the groups
    for name, group in grouped_df:
        # get the values of 'r_ui' and 'objective' for this group
        r_ui_values = group['r_ui'].values
        objective_values = group['objective'].values

        # add the lists to the dictionary using the name as the key
        result_dict[name] = (r_ui_values, objective_values)

    # create a list to store the correlations
    correlation_list = []

    # iterate over the items in the dictionary
    for name, lists in result_dict.items():
        # unpack the lists
        r_ui_values, objective_values = lists

        # check if the lists have at least 3 elements
        if len(r_ui_values) >= 3 and len(objective_values) >= 3:
            # calculate the Spearman rank correlation
            correlation, pvalue = spearmanr(r_ui_values, objective_values, nan_policy='omit')

            # add the correlation to the list if it is not nan
            if not math.isnan(correlation):
                correlation_list.append(correlation)

    #        print(f"For group {name}, the Spearman rank correlation is {correlation:.3f} with p-value {pvalue:.3f}")
    #    else:
    #        print(f"For group {name}, the lists do not have at least 3 elements, so the correlation was not calculated.")

    # calculate the mean of the correlations
    mean_correlation = mean(correlation_list)

    #print(f"The mean of the calculated correlations is {mean_correlation:.3f}.")
    return round(mean_correlation, 4)


def kendall_surprise(pred):
    """
    :param pred: Takes the predictions by surprise package as input
    :return: Returns average kendall correlation
    """
    import math
    from scipy.stats import kendalltau
    from statistics import mean
    df1 = pd.DataFrame(pred, columns=['case:concept:name', 'concept:name', 'r_ui', 'objective', 'details'])

    # create an empty dictionary to store the lists
    result_dict = {}

    # group the dataframe by 'concept:name'
    grouped_df = df1.groupby('concept:name')

    # iterate over the groups
    for name, group in grouped_df:
        # get the values of 'r_ui' and 'objective' for this group
        r_ui_values = group['r_ui'].values
        objective_values = group['objective'].values

        # add the lists to the dictionary using the name as the key
        result_dict[name] = (r_ui_values, objective_values)

    # create a list to store the correlations
    correlation_list = []

    # iterate over the items in the dictionary
    for name, lists in result_dict.items():
        # unpack the lists
        r_ui_values, objective_values = lists

        # check if the lists have at least 3 elements
        if len(r_ui_values) >= 3 and len(objective_values) >= 3:
            # calculate the Spearman rank correlation
            correlation, pvalue = kendalltau(r_ui_values, objective_values, nan_policy='omit')

            # add the correlation to the list if it is not nan
            if not math.isnan(correlation):
                correlation_list.append(correlation)

    #        print(f"For group {name}, the Spearman rank correlation is {correlation:.3f} with p-value {pvalue:.3f}")
    #    else:
    #        print(f"For group {name}, the lists do not have at least 3 elements, so the correlation was not calculated.")

    # calculate the mean of the correlations
    mean_correlation = mean(correlation_list)

    #print(f"The mean of the calculated correlations is {mean_correlation:.3f}.")
    return round(mean_correlation, 4)


def RMSE_surprise(pred):
    from sklearn.metrics import mean_squared_error
    import numpy as np
    """
    :param pred: Takes the predictions by surprise package as input
    :return: Returns RMSE
    """

    df1 = pd.DataFrame(pred, columns=['case:concept:name', 'concept:name', 'r_ui', 'objective', 'details'])
    df1['r_ui'] = df1['r_ui'].div(5)
    df1['objective'] = df1['objective'].div(5)

    # Calculate the RMSE between the 'r_ui' and 'objective' columns
    mse = mean_squared_error(df1['objective'], df1['r_ui'])
    rmse = np.sqrt(mse)

    return round(rmse, 4)


def MAE_surprise(pred):
    from sklearn.metrics import mean_absolute_error
    import numpy as np
    """
    :param pred: Takes the predictions by surprise package as input
    :return: Returns RMSE
    """

    df1 = pd.DataFrame(pred, columns=['case:concept:name', 'concept:name', 'r_ui', 'objective', 'details'])
    df1['r_ui'] = df1['r_ui'].div(5)            #score in [0,1] instead of [0,5]
    df1['objective'] = df1['objective'].div(5)  #score in [0,1] instead of [0,5]

    # Calculate the RMSE between the 'r_ui' and 'objective' columns
    mae = mean_absolute_error(df1['objective'], df1['r_ui'])

    return round(mae, 4)