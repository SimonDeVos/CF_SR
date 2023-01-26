import math
from statistics import mean

import numba as numba
import pandas as pd
import numpy as np
import scipy
from scipy.stats import stats
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
from datetime import datetime


def calc_results(R_pred, R_test):
    rmse, mae = metrics_sparse(R_pred, R_test)
    return


def df_to_piv(df, df_train_val, df_train, df_val, df_test):
    """
    :param df:
    :param df_train_val:
    :param df_train:
    :param df_val:
    :param df_test:
    :return: piv, piv_train_val,piv_train,piv_val,piv_test
    """
    # get selected indices from subselections
    index_arr_train_val = df_train_val.index.values.tolist()
    index_arr_train = df_train.index.values.tolist()
    index_arr_val = df_val.index.values.tolist()
    index_arr_test = df_test.index.values.tolist()

    # initialize subsamples as df/df_train_val, then y-value of non-selected observations to NaN
    # 1. train_val is df without test
    df_train_val = df.copy()
    df_train_val.loc[df_train_val.index.isin(index_arr_test), 'objective'] = np.nan
    # 2. test is df without train_val
    df_test = df.copy()
    df_test.loc[df_test.index.isin(index_arr_train_val), 'objective'] = np.nan
    # 3. train is train_val without val
    df_train = df_train_val.copy()
    df_train.loc[df_train.index.isin(index_arr_val), 'objective'] = np.nan
    # 4. val is train_val without train
    df_val = df_train_val.copy()
    df_val.loc[df_val.index.isin(index_arr_train), 'objective'] = np.nan

    values_log = 'objective'
    index_log = ['case:concept:name']
    columns_log = ['concept:name']

    # convert dataframes into pivot tables
    piv = pd.pivot_table(df, values=values_log, index=index_log, columns=columns_log, dropna=False)
    piv_train_val = pd.pivot_table(df_train_val, values=values_log, index=index_log, columns=columns_log, dropna=False)
    piv_train = pd.pivot_table(df_train, values=values_log, index=index_log, columns=columns_log, dropna=False)
    piv_val = pd.pivot_table(df_val, values=values_log, index=index_log, columns=columns_log, dropna=False)
    piv_test = pd.pivot_table(df_test, values=values_log, index=index_log, columns=columns_log, dropna=False)

    return piv, piv_train_val, piv_train, piv_val, piv_test


def func_one_in_train_val(df, df_train_val, df_test):
    """
    :param df: dataframe that is used as input
    :param df_train_val: subset of df that contains train+validation samples
    :param df_test: subset of df that contains test samples
    :return:
    """
    # make sure all instances have at least one observation in train_val (otherwise no info to make prediction)
    df = df.sort_values(by=['time_start'])
    empl_train_val = df_train_val['case:concept:name'].unique()
    to_add = []
    rows_to_drop = []
    # run through elements of test set
    for i in range(len(df_test)):
        # if this employee has no observations in train_val:
        if df_test.iloc[i]['case:concept:name'] not in empl_train_val:
            # save row to be added to train_val as np array
            to_add.append(df_test.iloc[i].values)
            # this person is now present in train_val set
            empl_train_val = np.append(empl_train_val, df_test.iloc[i]['case:concept:name'])
            # this observation should be dropped from the test set
            rows_to_drop.append(df_test.iloc[i].name)
            print('employee ' + str(
                df_test.iloc[i]['case:concept:name']) + ' has an observation moved from test to train_val')
    # the values of train_val are the values of the old dataframe, plus the values that should be added
    if to_add != []:
        arr_train_val = np.concatenate((df_train_val.values, to_add), axis=0)
        # re-initialize df_train_val with observations from test set
        df_train_val = pd.DataFrame(arr_train_val, columns=df.columns)
        # remove rows from test set that moved to df_train_val
        df_test = df_test.drop(rows_to_drop).copy()

    return df, df_train_val, df_test


def hyperpara_tuning(R_train, R_val, sr, objective, sm, lambda_array, beta_array, alpha_array, L_array, steps_array):
    """
    :param R_train:
    :param R_val:
    :param sr:
    :param objective: The metric on which you optimize hyperaparameters. 1 for rmse, 2 for mae, 3 for spearman, 4 for kendall
    :param sm:
    :param lambda_array:
    :param beta_array:
    :param alpha_array:
    :param L_array:
    :param steps_array:
    :return:
    """

    rmse_array = []
    mae_array = []
    spearman_array = []
    kendall_array = []

    N = len(R_train)  # N: num of employees
    M = len(R_train[0])  # M: num of jobs

    run = 1
    for lam_para in lambda_array:
        for beta_para in beta_array:
            for alpha_para in alpha_array:
                for L_para in L_array:
                    for steps_para in steps_array:
                        len_grid = len(lambda_array) * len(beta_array) * len(alpha_array) * len(L_array) * len(
                            steps_array)

                        # print('Grid run '+str(run)+'/'+str(len_grid), end='\r')
                        print(f'Grid run {run + 1}/{len_grid}', end='\r')

                        # P = np.random.rand(N,L_para)
                        # Q = np.random.rand(M,L_para)
                        P = np.full([N, L_para], 0.5)
                        Q = np.full([M, L_para], 0.5)

                        nP_val, nQ_val = matrix_factorization(R_train, P, Q, K=L_para, steps=steps_para,
                                                              alpha=alpha_para, lambda1=lam_para, beta=beta_para, SR=sr,
                                                              similarity_matrix=sm)
                        nR_val = np.dot(nP_val, nQ_val.T)
                        rmse, mae = metrics_sparse(nR_val, R_val)
                        spearman_value = spearman(nR_val, R_val)
                        kendall_value = kendall(nR_val, R_val)

                        rmse_array.append(rmse)
                        mae_array.append(mae)
                        spearman_array.append(spearman_value)
                        kendall_array.append(kendall_value)
                        run = run + 1

                        # Keep track of best hyperparameter settings
                        if objective == 1:
                            if rmse == min(rmse_array):
                                lambda_optimal = lam_para
                                beta_optimal = beta_para
                                alpha_optimal = alpha_para
                                L_optimal = L_para
                                steps_optimal = steps_para
                        if objective == 2:
                            if mae == min(mae_array):
                                lambda_optimal = lam_para
                                beta_optimal = beta_para
                                alpha_optimal = alpha_para
                                L_optimal = L_para
                                steps_optimal = steps_para
                        if objective == 3:
                            if spearman_value == max(spearman_array):
                                lambda_optimal = lam_para
                                beta_optimal = beta_para
                                alpha_optimal = alpha_para
                                L_optimal = L_para
                                steps_optimal = steps_para
                        if objective == 4:
                            if kendall_value == max(spearman_array):
                                lambda_optimal = lam_para
                                beta_optimal = beta_para
                                alpha_optimal = alpha_para
                                L_optimal = L_para
                                steps_optimal = steps_para

    print('array rmse: ' + str(rmse_array))
    print('array mae : ' + str(mae_array))
    print('array spearman: ' + str(spearman_array))
    print('array kendall: ' + str(kendall_array))

    # Display message of best run
    if objective == 1:
        best_run = rmse_array.index(min(rmse_array)) + 1
        print('best rmse value: ' + str(min(rmse_array)) + ' with run ' + str(best_run) + '/' + str(len_grid))
    if objective == 2:
        best_run = mae_array.index(min(mae_array)) + 1
        print('best mae value: ' + str(min(mae_array)) + ' with run ' + str(best_run) + '/' + str(len_grid))
    if objective == 3:
        best_run = spearman_array.index(max(spearman_array)) + 1
        print('best spearman value: ' + str(max(spearman_array)) + ' with run ' + str(best_run) + '/' + str(len_grid))
    if objective == 4:
        best_run = kendall_array.index(max(kendall_array)) + 1
        print('best kendall value: ' + str(max(kendall_array)) + ' with run ' + str(best_run) + '/' + str(len_grid))

    return lambda_optimal, beta_optimal, alpha_optimal, L_optimal, steps_optimal


def kendall(predictions, test):
    """
    :param predictions:
    :param test:
    :return:
    """
    pred = predictions.T
    test = test.T
    corr_arr = []
    for i in range(pred.shape[0]):
        if np.count_nonzero(
                ~np.isnan(test[i])) > 2:  # only take into consideration jobs that encountered at least 3 matches
            spcorr = stats.kendalltau(pred[i], test[i], nan_policy='omit')[0]
            corr_arr.append(spcorr)
    matrix_corr = mean(corr_arr)
    return round(matrix_corr, 4)


def matrix_factorization(R, P, Q, K, steps, alpha, lambda1, beta, SR, similarity_matrix):
    '''
    R: rating matrix
    P: |U| * K (User features matrix)
    Q: |D| * K (Item features matrix)
    K: latent features
    steps: iterations
    alpha: learning rate
    lambda1: regularization parameter
    beta: regularization parameter for SR
    '''

    Q = Q.T

    # Case without social regularization:
    if SR == False:
        for step in tqdm(range(steps), leave=False, colour='gray'):  # tqdm inserts nice loading bar
            for i in range(len(R)):
                for j in range(len(R[i])):
                    if R[i][j] > 0:
                        # calculate error
                        eij = R[i][j] - np.dot(P[i, :], Q[:, j])
                        for k in range(K):
                            # calculate gradient with a and beta parameter
                            P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - lambda1 * P[i][k])
                            Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - lambda1 * Q[k][j])
            eR = np.dot(P, Q)
            e = 0
            for i in range(len(R)):
                for j in range(len(R[i])):
                    if R[i][j] > 0:
                        e = e + pow(R[i][j] - np.dot(P[i, :], Q[:, j]), 2)
                        for k in range(K):
                            e = e + (beta / 2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))
            # 0.001: local minimum
            if e < 0.001:
                break

            # print('step: '+str(step) + "/" + str(steps) + ' - Error: '+ str(e))

    # Case with social regularization
    if SR == True:
        for step in tqdm(range(steps), leave=False, colour='gray'):  # tqdm inserts nice loading bar
            for i in range(len(R)):
                for j in range(len(R[i])):
                    if R[i][j] > 0:
                        # calculate error
                        eij = R[i][j] - np.dot(P[i, :], Q[:, j])
                        for k in range(K):
                            soc_reg = 0
                            for f in range(len(R[i])):
                                soc_reg = soc_reg + similarity_matrix[i][f] * (P[i][k] - P[f][k])
                            # calculate gradient with a and beta parameter
                            P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - lambda1 * P[i][
                                k] - beta * soc_reg)  # TODO change to - beta*soc_reg
                            Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - lambda1 * Q[k][j])

            eR = np.dot(P, Q)
            e = 0

            for i in range(len(R)):
                for j in range(len(R[i])):
                    if R[i][j] > 0:
                        e = e + pow(R[i][j] - np.dot(P[i, :], Q[:, j]), 2)
                        for k in range(K):
                            e = e + (lambda1 / 2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))
            # 0.001: local minimum
            if e < 0.001:
                break
            # print('step: '+str(step) + "/" + str(steps) + ' - Error: '+ str(e))

    return P, Q.T


"""
#Poging tot efficienter maken van code - loops vermijden
import numpy as np
import numba
@numba.jit
def matrix_factorization_opt(R, P, Q, K, steps, alpha,lambda1, beta, SR, similarity_matrix):
    Q = Q.T
    if SR == False:
        for step in range(steps):
            eij = R - np.dot(P,Q)
            P += alpha * (2 * eij * Q - lambda1 * P)
            Q += alpha * (2 * eij * P - lambda1 * Q)
            e = np.sum((R - np.dot(P,Q))**2)
            if e < 0.001:
                break
    else:
        for step in range(steps):
            eij = R - np.dot(P,Q)
            soc_reg = np.dot(similarity_matrix, P)
            P += alpha * (2 * eij * Q - lambda1 * P - beta*soc_reg)
            Q += alpha * (2 * eij * P - lambda1 * Q)
            e = np.sum((R - np.dot(P,Q))**2)
            if e < 0.001:
                break
    return P, Q.T
"""


def metrics_sparse(matrix_1, matrix_2):
    """
    :param matrix_1: this is the prediction matrix
    :param matrix_2: this is part of the observations
    :return: return RMSE, MAE for predicted elements with observed elements in matrix_2
    """
    MSE = 0  # initiate MSE to 0
    RMSE = 0  # initiate RMSE to 0
    MAE = 0  # initiate MAE to 0

    y_actual = []
    y_predicted = []
    # for each non-null elements in matrix_2:
    for i in range(len(matrix_1)):
        for j in range(len(matrix_1[i])):
            if not np.isnan(matrix_2[i][j]):
                y_actual.append(matrix_2[i][j])
                y_predicted.append(matrix_1[i][j])

    MSE = mean_squared_error(y_actual, y_predicted)
    RMSE = math.sqrt(MSE)
    MAE = mean_absolute_error(y_actual, y_predicted)

    return round(RMSE, 4), round(MAE, 4)


def piv_to_R(piv, piv_train_val, piv_train, piv_val, piv_test):
    """
    :param piv:
    :param piv_train_val:
    :param piv_train:
    :param piv_val:
    :param piv_test:
    :return:
    """
    R = piv.to_numpy()
    R_train = piv_train.to_numpy()
    R_test = piv_test.to_numpy()
    R_train_val = piv_train_val.to_numpy()
    R_val = piv_val.to_numpy()
    return R, R_train_val, R_train, R_val, R_test


def prep(df, test_size, out_of_time, one_in_train_val):
    """
    :param df: dataframe that is used as input
    :param test_size: test size as fraction in [0,1]
    :param out_of_time: 1 if train val test split should be made out of time. 0 otherwise
    :param one_in_train_val: 1 if train_val needs at least one observation per employee. 0 otherwise
    :return: df_train_val,df_train,df_val,df_test
    """
    if one_in_train_val == 0:
        if out_of_time == 0:
            df_train_val, df_test = train_test_split(df, test_size=test_size)
            df_train, df_val = train_test_split(df_train_val, test_size=test_size / (1 - test_size))
        if out_of_time == 1:
            # split df into train_val and test
            # rank on time_start
            df = df.sort_values(by=['time_start'])
            # take first (1-test_size)%. Add to df_train_val
            df_train_val = df.head(int(len(df.index) * (1 - test_size))).copy()
            # take the remaining (test_size)% and put in df_test
            df_test = df.tail(int(len(df.index) * test_size)).copy()

            # split train_val into train and val
            df_train_val = df_train_val.sort_values(by=['time_start'])
            df_train = df_train_val.head(int(len(df_train_val.index) * (1 - (test_size / (1 - test_size))))).copy()
            df_val = df_train_val.tail(int(len(df_train_val.index) * (test_size / (1 - test_size)))).copy()
    if one_in_train_val == 1:
        if out_of_time == 0:
            # split df into train_val and test as before
            df_train_val, df_test = train_test_split(df, test_size=test_size)

            # make sure train_val has at least one observation per employee
            df, df_train_val, df_test = func_one_in_train_val(df, df_train_val, df_test)

            # split df_train_val into df_train and df_val
            df_train, df_val = train_test_split(df_train_val, test_size=test_size / (1 - test_size))

        if out_of_time == 1:
            # split df into train_val and test
            # rank on time_start
            df = df.sort_values(by=['time_start'])
            # take first 75%, add to df_train_val
            df_train_val = df.head(int(len(df.index) * (1 - test_size))).copy()
            # take the remaining 25% and put in df_test
            df_test = df.tail(int(len(df.index) * test_size)).copy()

            # make sure train_val has at least one observation per employee
            df, df_train_val, df_test = func_one_in_train_val(df, df_train_val, df_test)

            # split train_val into train and val
            df_train_val = df_train_val.sort_values(by=['time_start'])
            df_train = df_train_val.head(int(len(df_train_val.index) * (1 - (test_size / (1 - test_size))))).copy()
            df_val = df_train_val.tail(int(len(df_train_val.index) * (test_size / (1 - test_size)))).copy()

    return df, df_train_val, df_train, df_val, df_test


def print_results(R_pred, R_test):
    rmse, mae = metrics_sparse(R_pred, R_test)

    print('mae: ' + str(mae))
    print('rmse: ' + str(rmse))
    print('spearman: ' + str(spearman(R_pred, R_test)))
    print('kendall: ' + str(kendall(R_pred, R_test)))

    return mae, rmse, spearman(R_pred, R_test), kendall(R_pred, R_test)


def read_data(datasets):
    """
    :param datasets: Takes list of strings
    :return: Returns read df, cov_array_personal, cov_array_personal_num, cov_array_personal_cat
    """
    if datasets.count(True) != 1:
        raise ValueError("Select exactly one dataset")

    if datasets[0]:
        path = 'Data/toy_example.csv'
        separator = ';'
        cov_array_personal = ['X1', 'X2', 'X3', 'X4']
        cov_array_personal_num = ['X1', 'X2', 'X3']
        cov_array_personal_cat = ['X4']
        df = pd.read_csv(path, sep=str(separator))

    if datasets[1]:
        path = 'Data/dataset_1.csv'
        separator = ';'
        df = pd.read_csv(path, sep=str(separator))
        cov_array_personal = df.columns[[22, 23]].tolist()
        cov_array_personal_num = []
        cov_array_personal_cat = df.columns[[22, 23]].tolist()

        raise TypeError("This dataset is not publicly available. Please select \"dataset_3\" in 1.Settings")

    if datasets[2]:
        path = 'Data/dataset_2.csv'
        separator = ";"
        df = pd.read_csv(path, sep=str(separator))
        cov_array_personal = df.columns[[6, 61]].tolist()
        cov_array_personal_num = []
        cov_array_personal_cat = df.columns[[6, 61]].tolist()
        raise TypeError("This dataset is not publicly available. Please select \"dataset_3\" in 1.Settings")

    if datasets[3]:
        path = 'Data/dataset_3.csv'
        separator = ';'
        cov_array_personal = ['V06', 'V08']
        cov_array_personal_num = ['V08']
        cov_array_personal_cat = ['V06']
        df = pd.read_csv(path, sep=str(separator))
        df = df.rename(columns={'id': 'case:concept:name', 'act': 'concept:name'})

    return df, cov_array_personal, cov_array_personal_num, cov_array_personal_cat


def results_to_txt(nR_cf, nR_cf_sr, R_test, start_time_experiment, toy_data, dataset_1, dataset_2, dataset_3,
                   cov_array_personal, test_size, out_of_time, one_in_train_val, lambda_array, beta_array, alpha_array,
                   L_array, steps_array, lambda_optimal_cf, beta_optimal_cf, L_optimal_cf, alpha_optimal_cf,
                   steps_optimal_cf, lambda_optimal_cf_sr, beta_optimal_cf_sr, alpha_optimal_cf_sr, L_optimal_cf_sr,
                   steps_optimal_cf_sr):
    # Get the current date and time
    now = datetime.now()
    duration = now - start_time_experiment

    # Format the date and time to be used in the file name
    file_name = "Results/results_{}.txt".format(now.strftime("%Y-%m-%d %H-%M-%S"))

    mae_cf, rmse_cf, spearman_cf, kendall_cf = calc_results(nR_cf, R_test)
    mae_cf_sr, rmse_cf_sr, spearman_cf_sr, kendall_cf_sr = calc_results(nR_cf_sr, R_test)

    # Open a new text file in write mode
    with open(file_name, "w") as file:
        file.write("\nstart time experiment:" + str(start_time_experiment)
                   + "\nend time experiment:" + str(now)
                   + "\ntime elapsed:" + str(duration)

                   # Write the contents of the variable to the file
                   + "\n\n1. DATASET"
                   + "\nToy_data: " + str(toy_data)
                   + "\ndataset_1: " + str(dataset_1)
                   + "\ndataset_2: " + str(dataset_2)
                   + "\ndataset_3: " + str(dataset_3)

                   + "\n\ncov_array_personal: ".join(cov_array_personal)

                   + "\n\n2. Train/val/test split:"
                   + "\ntest_size:" + str(test_size)
                   + "\nout_of_time:" + str(out_of_time)
                   + "\none_in_train_val:" + str(one_in_train_val)

                   + "\n\n3. Hyperparameters:"
                   + "\nlambda_array: " + str(lambda_array)
                   + "\nbeta_array: " + str(beta_array)
                   + "\nalpha_array: " + str(alpha_array)
                   + "\nL_array: " + str(L_array)
                   + "\nsteps_array: " + str(steps_array)

                   + "\n\noptimal hyperpara CF:"
                   + "\nlambda: " + str(lambda_optimal_cf)
                   + "\nbeta: " + str(beta_optimal_cf)
                   + "\nalpha: " + str(alpha_optimal_cf)
                   + "\nL: " + str(L_optimal_cf)
                   + "\nsteps: " + str(steps_optimal_cf)

                   + "\n\noptimal hyperpara CF+SR:"
                   + "\nlambda: " + str(lambda_optimal_cf_sr)
                   + "\nbeta: " + str(beta_optimal_cf_sr)
                   + "\nalpha: " + str(alpha_optimal_cf_sr)
                   + "\nL: " + str(L_optimal_cf_sr)
                   + "\nsteps: " + str(steps_optimal_cf_sr)

                   + "\n\n4. Results:"

                   + "\n\nresults CF:"
                   + "\nmae_cf: " + str(mae_cf)
                   + "\nrmse_cf: " + str(rmse_cf)
                   + "\nspearman_cf: " + str(spearman_cf)
                   + "\nkendall_cf: " + str(kendall_cf)

                   + "\n\nresults CF+SR:"
                   + "\nmae_cf_sr: " + str(mae_cf_sr)
                   + "\nrmse_cf_sr: " + str(rmse_cf_sr)
                   + "\nspearman_cf_sr: " + str(spearman_cf_sr)
                   + "\nkendall_cf_sr: " + str(kendall_cf_sr))
    return


def sim(df, N, cov_array_personal, cov_array_personal_cat, cov_array_personal_num):
    """
    :param df:
    :param N:
    :param cov_array_personal:
    :param cov_array_personal_cat:
    :param cov_array_personal_num:
    :return:
    """
    df_sim = df.drop_duplicates(subset=["case:concept:name"])[cov_array_personal].copy(deep=True).reset_index(drop=True)

    # initialize similarity_matrix
    similarity_matrix = np.zeros((N, N))

    nr_features = len(cov_array_personal_cat) + len(cov_array_personal_num)
    fract_num = len(cov_array_personal_num) / nr_features

    if len(cov_array_personal_num) != 0:  # rescale numerical values to domain [0,1]
        x = df_sim[cov_array_personal_num].values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df_sim_num = pd.DataFrame(x_scaled)

    for i in tqdm(range(N), leave=False, colour='gray'):
        for j in range(N):
            if len(cov_array_personal_num) != 0:
                dist_num = scipy.spatial.distance.cosine(df_sim_num.iloc[i].to_numpy(), df_sim_num.iloc[j].to_numpy())
                similarity_num = 1 - dist_num  # value between 0 and 1
            else:
                similarity_num = 0
            if len(cov_array_personal_cat) != 0:
                similarity_cat = np.sum(
                    df_sim[cov_array_personal_cat].iloc[i] == df_sim[cov_array_personal_cat].iloc[j]) / len(
                    cov_array_personal_cat)
            else:
                similarity_cat = 0
            similarity = fract_num * similarity_num + (1 - fract_num) * similarity_cat
            similarity_matrix[i][j] = similarity
    return similarity_matrix


def spearman(predictions, test):
    """
    :param predictions:
    :param test:
    :return:
    """
    pred = predictions.T
    test = test.T
    corr_arr = []
    for i in range(pred.shape[0]):
        if np.count_nonzero(
                ~np.isnan(test[i])) > 2:  # only take into consideration jobs that encountered at least 3 matches
            spcorr = stats.spearmanr(pred[i], test[i], nan_policy='omit')[0]
            corr_arr.append(spcorr)
    matrix_corr = mean(corr_arr)
    return round(matrix_corr, 4)
