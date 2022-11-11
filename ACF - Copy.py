import matplotlib.pyplot as plt
import os

import scipy.integrate
import seaborn as sns
import math

import statsmodels
from statsmodels.graphics.tsaplots import plot_acf
from statistics import stdev
import numpy as np
import pandas as pd
import seaborn as sns
import statistics

#Try messing with alpha changed to one

def df_manual_acf(df, label, cwd):
    # print("alpha",  (df['cumulated strain'].max() - df['cumulated strain'].min()), df['cumulated strain'].shape[0])
    alphaValue = (df['cumulated strain'].max() - df['cumulated strain'].min()) / df['cumulated strain'].shape[0]
    lags = df.shape[0] // 2
    resultList = [0] * lags
    deltaM = 1
    shearAverage = df['shear rate'].mean()
    # shift shear function to x=0
    df['shear rate shifted'] = df['shear rate'] - shearAverage

    # calculate the denominator
    df['acf denom'] = df['shear rate shifted'] ** 2
    denom = df['acf denom'].sum()

    # calculate autocorrelation
    shear_rate_shifted = list(df["shear rate shifted"])
    for i in range(lags):
        for j in range(df.shape[0] - i):
            resultList[i] += shear_rate_shifted[j] * shear_rate_shifted[j + i * deltaM]
        resultList[i] = resultList[i] / denom

    df_lags = pd.DataFrame(resultList, columns=['lags'])
    #print(df_lags)


    return df_lags, alphaValue

#def df_shearRate(df):

def df_dataCalculation(df):
    df['pressure'] = ((df['total stress tensor (xx)'] + df['total stress tensor (zz)'])) / 2
    df['normalStress'] = df['total stress tensor (xx)'] - df['total stress tensor (zz)']
    df['muReal'] = ((df['normalStress'] ** 2) + (4 * (df['total stress tensor (xz)'] ** 2)) / (
            2 * df['pressure'])) ** .5
    cols = ['total stress tensor (xx)', 'total stress tensor (zz)', 'total stress tensor (xz)', 'pressure',
            'normalStress', 'muReal', 'time']
    return df


def fileNameParser(fileName):
    # st_D2N1000VF0.8Bidi1.4_0.5Square_1_preshear_nobrownian_2D_stress1cl_shear
    packing = "Packing" + fileName.split("VF")[1].split('B')[0]
    stress = "Stress" + fileName.split("stress")[1].split('cl')[0]
    return packing + stress


def dat_to_df(file_name, samples_skip=0):
    '''
    Converts the data in a dat file to a dataframe.
    Args:
        file_name(Str): File name of .dat file data
    Return:
        df(DataFrame): Dataframe object holding .dat file data
    '''
    cols = ['time', 'cumulated strain', 'shear rate',
            'total stress tensor (xx)', 'total stress tensor (xy)', 'total stress tensor (xz)',
            'total stress tensor (yz)', 'total stress tensor (yy)', 'total stress tensor (zz)',
            'contact stress tensor (xx)', 'contact stress tensor (xy)', 'contact stress tensor (xz)',
            'contact stress tensor (yz)', 'contact stress tensor (yy)', 'contact stress tensor (zz)',
            'dashpot stress tensor (xx)', 'dashpot stress tensor (xy)', 'dashpot stress tensor (xz)',
            'dashpot stress tensor (yz)', 'dashpot stress tensor (yy)', 'dashpot stress tensor (zz)',
            'hydro stress tensor (xx)', 'hydro stress tensor (xy)', 'hydro stress tensor (xz)',
            'hydro stress tensor (yz)', 'hydro stress tensor (yy)', 'hydro stress tensor (zz)',
            'repulsion stress tensor (xx)', 'repulsion stress tensor (xy)', 'repulsion stress tensor (xz)',
            'repulsion stress tensor (yz)', 'repulsion stress tensor (yy)', 'repulsion stress tensor (zz)']
    file = open(file_name)
    lines = file.readlines()[17 + samples_skip:]
    lines_split = []
    for line in lines:
        lines_split.append(line.split())
    df = pd.DataFrame(lines_split, columns=cols).astype(float)
    return df


def interpolate(df, maxLength):
    array = list(df['lags'])
    num_nulls = maxLength - len(array)
    if num_nulls != 0:
        placement = maxLength / num_nulls
    for i in range(1, num_nulls + 1):
        if ((i * round(placement)) % len(array) == 0):
            array.insert(((i + 1) * round(placement)) % len(array), np.nan)
        else:
            array.insert((i * round(placement)) % len(array), np.nan)
    return pd.DataFrame(array, columns=['lags']).interpolate(method='linear')


def plot_all(data, alphas, file_name, cwd, run_name):
    for key in data.keys():
        alphaValue = alphas[key]
        plt.scatter(data[key].index * alphas[key], data[key]['lags'], s=1)
    plt.title(file_name)
    plt.xlabel("Cumulative Strain Lag", fontsize=12)
    plt.ylabel("Shear Rate Autocorrelation", fontsize=12)
    plt.xlim(0, 1)
    # plt.savefig(cwd + "/results/" + "Runs_" +fileNameParser(run_name) + ".png")
    #plt.show()
    #plt.close()


def average_ACF(data, alpha, maxLength, file_name, cwd, run_name,df2, plot=True, plotall=False):
    plot_avg = pd.DataFrame([0] * maxLength, columns=['lags'])
    #time_measure = pd.DataFrame([0] * maxLength, columns = ['time'])
    #print(time_measure)
    for key in data.keys():
        plot_avg['lags'] = plot_avg['lags'] + data[key]['lags']
    print(plot_avg['lags'])
    plot_avg['lags'] = plot_avg['lags'] / 10
    if (plot):
        plt.title(file_name)
        plt.scatter(plot_avg.index * alpha, plot_avg['lags'], s=0.5, linewidths=3)
        plt.xlabel("Cumulative Strain Lag", fontsize=12)
        plt.ylabel("Shear Rate Autocorrelation", fontsize=12)
        plt.xlim(0, 1)
        # plt.savefig(cwd + "/results/" + "Avg_" +fileNameParser(run_name) + ".png")
        #plt.show()
        #plt.close()
    if (plotall):
        plt.scatter(plot_avg.index * alphaValue, plot_avg['lags'], s=0.5, linewidths=3)
    return plot_avg


def variance_ACF(data, alphas, avg_df, maxLength, file_name, cwd,df, run_name, plotall=True):
    var_data = {}
    var_sum = pd.DataFrame([0] * maxLength, columns=['lags'])
    x_axis_data = {}
    x_axis = pd.DataFrame([0] * maxLength, columns=['time'])
    print(data.keys())
    for i in data.keys():
        x_axis_data[i] = int(df['time'])
        var_data[i] = (np.power(data[i]['lags'] - avg_df['lags'], 2), alphas[i])
        var_sum['lags'] = var_sum['lags'] + np.power(data[i]['lags'] - avg_df['lags'], 2)
        # plt.scatter(var_data[i][0].index*var_data[i][1], var_data[i][0], s=0.5)
    print(x_axis_data)
    var_sum['lags'] = var_sum['lags'] / 10
    plt.scatter(var_sum.index * alphas[1], var_sum, s=1, alpha=1, linewidths=5, c='black')
    plt.title(file_name)
    plt.xlabel("Cumulative Strain Lag", fontsize=12)
    plt.ylabel("Autocorrelation Variance", fontsize=12)
    plt.xlim(0, 1)
    # plt.savefig(cwd + "/results/" + "VarAll_" + fileNameParser(run_name) + ".png")
    #plt.show()
    #plt.close()
    #print(var_sum)
    return var_sum


def integrate(data, alphas, avg_df, maxLength, file_name, cwd, run_name,df, plotall=True):
    var_data = {}
    var_sum = pd.DataFrame([0] * maxLength, columns=['lags'])
    for i in data.keys():
        var_data[i] = (np.power(data[i]['lags'] - avg_df['lags'], 2), alphas[i])
        var_sum['lags'] = var_sum['lags'] + np.power(data[i]['lags'] - avg_df['lags'], 2)
        # plt.scatter(var_data[i][0].index*var_data[i][1], var_data[i][0], s=0.5)
    var_sum['lags'] = var_sum['lags'] / 10
    var_sum = np.array(var_sum)
    var_sum = var_sum[np.logical_not(np.isnan(var_sum))]
    variance_integrated = scipy.integrate.trapz(var_sum) * (1/len(var_sum))
    #print(variance_integrated)
    return variance_integrated
    #print(var_sum)

    #print(type(var_sum))

def main():
    plt.style.use("seaborn-deep")
    cwd = os.getcwd()
    files = os.listdir(cwd + "/data/0.77/")
    mergePlot = True
    data_avg = {}
    data_var = {}
    data_alpha = {}

    stresses = []
    for i in os.listdir(cwd + "/data/0.77/" + files[0]):

        stresses.append(fileNameParser(i).split("Stress")[1])

    # stresses = [stresses[3], stresses[6], stresses[9]]
    for stress in stresses:
        data = {}
        alphas = {}
        maxLength = 0
        maxTime = 0
        for file_name in files:
            #print(stress, file_name)
            #run_name = "st_D2N1000VF0.76Bidi1.4_0.5Square_"+file_name.split("run")[1]+"_preshear_nobrownian_2D_stress"+stress+"cl_shear.dat"

            run_name = "st_D2N1000VF0.77Bidi1.4_0.5Square_" + file_name.split("run")[
                1] + "_preshear_nobrownian_2D_stress" + stress + "cl_shear.dat"
            df = dat_to_df(cwd + "/data/0.77/" + file_name + "/" + run_name)
            df_dataCalculation(df)
            df2 = df
            df, alpha = df_manual_acf(df, fileNameParser(run_name).split("stress"), cwd)
            maxLength = max(maxLength, df.shape[0])
            maxTime =  max(maxTime,df2.shape[0])
            if (file_name in data_alpha.keys()):
                data_alpha[stress] = min(data_alpha[file_name], alpha)
            else:
                data_alpha[stress] = alpha
            df['time'] = df2['time']
            data[int(file_name.split("run")[1])] = df
            alphas[int(file_name.split("run")[1])] = alpha
        for key in data.keys():
            df_lags = interpolate(data[key], maxLength)
        #print(df['time'])
        plot_all(data, alphas, file_name, cwd, run_name)
        data_avg[stress] = average_ACF(data, alpha, maxLength, file_name, cwd, run_name,df, plot=True)
        data_var[stress] = variance_ACF(data, alphas, data_avg[stress], maxLength, file_name, cwd, run_name,df)
        integrate(data, alphas, data_avg[stress], maxLength, file_name, cwd, run_name,df)
        #print(stress)
        # print(np.trapz(variance_ACF(data, alphas, data_avg[stress], maxLength, file_name, cwd, run_name)))
    if (mergePlot):
        for key in stresses:
            plt.scatter(data_avg[key].index * data_alpha[key], data_avg[key]['lags'], s=0.5)
        plt.title('Autocorrelation')
        plt.legend(stresses, markerscale=6)
        plt.axhline(y=0, color='grey', linestyle='-', lw=1)
        plt.xlabel("Cumulative Strain Lag", fontsize=12)
        plt.ylabel("Average Shear Rate Autocorrelation", fontsize=12)
        plt.xlim(0, 1)
        # plt.savefig(cwd + "/results/"+ "TotalAvg_" + stress + ".png")
        plt.close()

        for key in data_avg.keys():
            plt.scatter(data_var[key].index * data_alpha[key], data_var[key]['lags'], s=1)
        plt.title('Variance')
        plt.legend(stresses, markerscale=6)
        plt.xlabel("Cumulative Strain Lag", fontsize=12)
        plt.ylabel("Average Autocorrelation Variance", fontsize=12)
        plt.xlim(0, 1)
        # plt.savefig(cwd + "/results/"+ "TotalVar_" + stress + ".png")
        plt.close()


main()
'''
Notes:

For each dataset subtract the average.

*y axis is c
1) [x] Create ten new lines with the average subtracted.
2) [x] Square the resulting values
3) [x] sum all 10 squared values
4) [x] divide by 10 to normalize

How does the height

P: What does the variance look like for each overall dataset.


Final two plots:
a) averages at different stresses, for the same packing fraction, on the same plot
b) variance at different stresses, for the same packing fracion, on the same plot
'''
