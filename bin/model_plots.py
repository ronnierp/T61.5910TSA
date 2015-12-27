#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Plot Settings
pd.options.display.mpl_style = 'default'

mean_hist = pd.read_csv('../results/findicators/mean_hist.csv', header=0, index_col=0)
arima_hist = pd.read_csv('../results/findicators/arima_hist.csv', header=0, index_col=0)
var_hist = pd.read_csv('../results/findicators/var_hist.csv', header=0, index_col=0)
rf_hist = pd.read_csv('../results/findicators/rf_hist.csv', header=0, index_col=0)

run_artificial = 0

filenames = [['1step_1st.png', '1step_5th.png', '1step_10th.png'], ['varstep_1st.png', 'varstep_5th.png', 'varstep_10th.png'], ['hstep_1st.png', 'hstep_5th.png', 'hstep_10th.png']]
cols = [[0, 4, 9], [11, 16], [22, 26, 31]]
color_code = ['b', 'g', 'r', 'y']

for w in range(3):
    for p in range(len(cols[w])):

        pflag = 1

        if w == 0:
            # 1-step method
            one_step = pd.concat([mean_hist.ix[:, cols[w][p]], arima_hist.ix[:, cols[w][p]], var_hist.ix[:, cols[w][p]], rf_hist.ix[:, cols[w][p]]], axis=1)
            one_step.columns = ['Mean', 'ARIMA', 'VAR', 'RF']

            # Variable step method
        elif w == 1:
            if p == 1 and np.isnan(var_hist.ix[var_hist.index[0], cols[w][p]]):

                if np.isnan(var_hist.ix[var_hist.index[0], 16]):
                    # The variable step prediction didn't reach the end of the prediction window (17, 16 columns)
                    pflag = 0
                else:
                    # Variable step final prediction
                    one_step = pd.concat([mean_hist.ix[:, cols[0][2]], var_hist.ix[:, 16], rf_hist.ix[:, 16]], axis=1)
                    one_step.columns = ['Mean', 'VAR', 'RF']

            else:
                # Variable step final prediction
                one_step = pd.concat([mean_hist.ix[:, cols[0][2]], var_hist.ix[:, cols[w][p]], rf_hist.ix[:, cols[w][p]]], axis=1)
                one_step.columns = ['Mean', 'VAR', 'RF']

        else:
            # h-step method
            one_step = pd.concat([mean_hist.ix[:, cols[0][p]], arima_hist.ix[:, cols[w][p]], var_hist.ix[:, cols[w][p]], rf_hist.ix[:, cols[w][p]]], axis=1)
            one_step.columns = ['Mean', 'ARIMA', 'VAR', 'RF']

        check_nan = one_step.isnull().all()

        if pflag and not check_nan.all():

            if check_nan.any():
                one_step.drop(one_step.columns[check_nan == True], axis=1, inplace=True)

            try:
                if run_artificial:
                    one_step[one_step <= 1000].plot(kind='hist', alpha=0.5, color=color_code[0:sum(check_nan == False)], stacked=True)
                else:

                    med_out = one_step.median() > 3*one_step.median().median()

                    if np.sum(one_step.ix[:, one_step.columns[med_out == True]] < 500000).values > 0:
                        one_step[one_step <= 500000].plot(kind='hist', alpha=0.5, color=color_code[0:sum(check_nan == False)], stacked=True)
                    else:
                        if w == 2 and p == 1:
                            one_step.ix[one_step.index[((one_step <= 50000).sum(axis=1) == sum(med_out == False))], one_step.columns[med_out == False]].plot(kind='hist', alpha=0.5, color=color_code[0:sum(check_nan == False)], stacked=True)
                        else:
                            if p == 2:
                                one_step.ix[one_step.index[((one_step <= 1.5e7).sum(axis=1) > 2)], one_step.columns[med_out == False]].plot(kind='hist', alpha=0.5, color=color_code[0:sum(check_nan == False)], stacked=True)
                            else:
                                one_step.ix[one_step.index[((one_step <= 500000).sum(axis=1) > 2)], one_step.columns[med_out == False]].plot(kind='hist', alpha=0.5, color=color_code[0:sum(check_nan == False)], stacked=True)

            except ValueError:
                one_step.plot(kind='hist', alpha=0.5, color=color_code[0:sum(check_nan == False)], stacked=True)

            plt.xlabel('Sum of Squared Errors')
            plt.ylabel('Frequency')
            plt.savefig('../results/findicators/' + filenames[w][p], dpi=200)
            plt.close()