#!/usr/bin/env python

import argparse
import numpy as np
import pandas as pd
import sklearn.ensemble as ske
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import date
import sys
from cStringIO import StringIO


def parse_data(file_path, DT):
    # Parse the data and create TimeStamp indexes
    # Usage:    parse_data(file_path)
    #           file_path - string containing the path to the CSV data, mandatory
    #           DT - Boolean: 1 - the file contains DateTime index; 0 - Convert TimeStamp

    # Import the Artificial Time Series
    data = pd.read_csv(file_path,
                      index_col=0,
                      header=0)

    if DT:
        t_span = data.index

        # Mask to divide the data in 90 days periods
        t_breaks = pd.DatetimeIndex(t_span).to_period(freq='Q-Jan')

    else:
        # Retrieve and convert dates
        t_span = []
        for i in range(len(data.index)):
            t_span.append(pd.to_datetime(date.fromtimestamp(data.index[i])))
        t_span = pd.DatetimeIndex(t_span)

        data.index = t_span

        # Mask to divide the data in 90 days periods
        t_breaks = data.index.to_period(freq='Q-Jan')

    return data, t_span, t_breaks


def parse_artificial_data(file_path, cols, col_header):
    # Parse the data and create TimeStamp indexes
    # Usage:    parse_artificial_data(file_path)
    #           file_path - string containing the path to the CSV data, mandatory
    #           cols - list containing the feature column numbers, mandatory
    #           col_header - list of strings containing the column headers, mandatory

    # Import the Artificial Time Series
    data = pd.read_csv(file_path,
                      usecols=cols,
                      index_col=0,
                      names=col_header,
                      header=0)

    # Create DateTime array
    t_span = data.index

    # Mask to divide the data in 90 days periods
    t_breaks = np.repeat(np.linspace(0, len(data)/90 - 1, len(data)/90), 90)
    t_breaks = pd.Series(t_breaks)

    return data, t_span, t_breaks


def divide_sets(w_set):
    # Divide the quarter data into training and test sets.
    #   Usage:  divide_sets(w_set)
    #           w_set - Stationary Data (Pandas Data Frame)

    # Retrieve season data and Drop NaN
    w_set.dropna(axis=0, inplace=True)

    n_obs = len(w_set)

    # Choose Training and Test sets
    n_train = int(float(n_obs)*beta)        # Split based on the Training Fraction (beta)

    # Create Indexes (Time stamped)
    train = w_set.index[:n_train]

    test = w_set.index[n_train:]

    return w_set, train, test, n_train


def evaluate_trend(probe):
    # Perform the Augmented Dickey-Fuller test for Stationarity and remove
    # 1st order polynomial trends

    # Test for stationarity and perform differencing
    st_flag, integrated_diff, feature_data = stationarity(probe)

    if not st_flag:
        # Independent Variable
        x = sm.add_constant(np.linspace(1, len(probe.index), len(probe.index)))

        # Ordinary Least Squares (linear model)
        line_model = sm.OLS(probe, x)
        line_res = line_model.fit()

        # t-statistics for the parameters (Exact zero corresponds to a 1st or 0th order polynomial)
        if (line_res.pvalues == 0).all() or (probe.ix[probe.index[-1]] - probe.ix[probe.index[0]]) < 1e-3:
            st_flag = 0
            integrated_diff = None
            feature_data = None

        else:
            # There is a linear trend, therefore remove it from the data
            if (line_res.pvalues <= 0.05).all():
                dt_data = probe - np.dot(x, line_res.params)

            else:
                dt_data = probe

            # Test for stationarity and perform differencing
            st_flag, integrated_diff, feature_data = stationarity(dt_data)

    return st_flag, integrated_diff, feature_data


def stationarity(probe):
    # Perform the Augmented Dickey-Fuller test for Stationarity and choose
    # the optimum number of differences up to a maximum of 5

    st = 0              # Stationarity Flag
    difference = 0      # Number of differences
    v_flag = 0          # Pointer for overdifference correction
    while st == 0 and difference <= 5:

        # Perform the test
        try:
            test_res = sm.tsa.adfuller(probe, maxlag=difference)[1]
        except (ValueError, np.linalg.LinAlgError):
            st = 0
            difference = 11

        if test_res == 0:
            st = 0
            difference += 1
        elif test_res < alpha:
            st = 1
            og_var = probe.var()        # Original Variance
        else:
            difference += 1

    # Perform the differences
    if st and (difference > 0):

        # Check for overdifferencing by evaluating the variance
        v_check = 0     # Variance check
        c_diff = 1      # Difference counter
        while v_check == 0:

            # Difference the data
            intermediate = np.diff(probe.values, n=c_diff)

            if c_diff != 1:

                # A higher variance indicates overdifferencing
                if np.var(intermediate) >= og_var:
                    difference = c_diff - 1
                    stn_data = np.diff(probe.values, n=difference)

                    v_flag = 1
                    v_check = 1     # Break the loop

                else:
                    if c_diff == difference:
                        stn_data = intermediate
                        v_check = 1

                    else:
                        c_diff += 1

            # End of the loop
            elif c_diff == difference:
                stn_data = intermediate
                v_check = 1

            else:
                c_diff += 1

            og_var = np.var(intermediate)

        # The final data set is reduced by the number of differences
        stn_data = pd.Series(stn_data, index=probe.index[difference:len(probe)])

    else:
        stn_data = probe

    # Further processing and stationarity tests
    if st:
        # Subtract the mean and divide by the variance
        if np.abs(stn_data.mean()) > 0.001:
            aux = stn_data - stn_data.mean()
            stn_data = aux.copy()/aux.var()

        # Normalize to the maximum value in the dataset
        else:
            stn_data = stn_data/stn_data.abs().max()

        # Autocorrelation Analysis
        # Confidence Interval Limits
        acf_limit = 1.96/np.sqrt(len(stn_data))

        # ACF of the first 20 lags
        acf = sm.tsa.acf(stn_data, nlags=20)

        # Count the number of lags outside of the confidence interval
        acf_check = sum(np.abs(acf) > acf_limit)

        # Check for control of overdifferencing
        if v_flag:
            if acf_check > 8:
                st = 0
                stn_data = None
                difference = None

        else:
            # The variable is not stationary if more than 40% of the lags fall outside of the CI
            if acf_check > 8:
                st = 0
                stn_data = None
                difference = None

    return st, difference, stn_data


def ar(train_data, astep):
    # Fit the training data to an AR model and perform out of sample prediction
    # Usage: ar(train_data)
    #           train_data - pandas Series of the desired variable
    #           astep - number of steps

    # Model construction
    ar_mod = sm.tsa.AR(train_data)

    # Order selection (AIC) and fit
    ar_res = ar_mod.fit(ic='AIC', disp=False)

    # Out of sample prediction
    ar_pred = ar_res.predict(start=len(train_data), end=len(train_data) + astep - 1)

    return ar_pred


def arma(train_data, fstep, art_flag):
    # Perform ARMA model selection, fit the training data and forecast the test data
    #   Stationary Data is purely ARMA
    #   Differenced data is ARIMA, where the I has already been processed
    #
    # Usage: arma(train_data)
    #             train_data - pandas Series of the desired variable
    #             fstep - number of steps
    #             art_flag - artificial data flag (Boolean)

    # Include timestamps in the artificial data set
    if art_flag:
        train_data.index = pd.date_range('2015/01/01', periods=len(train_data), freq='D')

    # ---------------- Determine AR(p) and MA(q) parameters------------------
    xtr = train_data.index[:int(0.75*len(train_data)) + 1]    # Training set
    xte = train_data.index[int(0.75*len(train_data)):]        # Evaluation set

    # Some models are not appropriate and might raise warnings (silenced) or print messages (captured)
    old_stderr = sys.stderr
    err = sys.stderr = StringIO()

    try:
        # Autocorrelation Function (ACF)
        macf = sm.tsa.acf(train_data.ix[xtr], nlags=20)

        # Partial-Autocorrelation Function (PACF)
        pacf = sm.tsa.pacf(train_data.ix[xtr], nlags=20)

        q_idx = np.abs(macf[1:5]).min()
        p_idx = np.abs(pacf[1:5]).min()

        # Find the index corresponding to the coefficients q_idx and p_idx
        for i in range(5):
            if p_idx == np.abs(pacf[i]):
                a = i
            if q_idx == np.abs(macf[i]):
                b = i

    except (ValueError, np.linalg.LinAlgError):
        a = 4
        b = 4

    # First try pure MA or pure AR models (Parsimony Criteria)
    smodel = []
    # Pure AR model
    try:
        ar_model = sm.tsa.ARMA(train_data.ix[xte], order=[a, 0])
        ar_result = ar_model.fit(solver='bfgs', disp=False, warn_convergence=False)

        smodel.append((ar_result.aic, a, 0))

    except (ValueError, np.linalg.LinAlgError):
        smodel.append((-1, 0, 0))

    # Pure MA model
    try:
        ma_model = sm.tsa.ARMA(train_data.ix[xte], order=[0, b])
        ma_result = ma_model.fit(solver='bfgs', disp=False, warn_convergence=False)

        smodel.append((ma_result.aic, 0, b))

    except (ValueError, np.linalg.LinAlgError):
        smodel.append((-1, 0, 0))

    # Parsimony (no penalty)
    if smodel[0][0] < smodel[1][0]:
        # Run AR
        try:
            ar_model = sm.tsa.ARMA(train_data, order=[a, 0])
            ar_result = ar_model.fit(solver='bfgs', disp=False, warn_convergence=False)
            arma_forecast = ar_result.forecast(steps=fstep, alpha=0.05)[0]

        except (ValueError, np.linalg.LinAlgError):
            arma_forecast = np.nan*np.ones([1, fstep])

    elif smodel[0][0] > smodel[1][0]:
        # Run MA
        try:
            ma_model = sm.tsa.ARMA(train_data, order=[0, b])
            ma_result = ma_model.fit(solver='bfgs', disp=False, warn_convergence=False)
            arma_forecast = ma_result.forecast(steps=fstep, alpha=0.05)[0]

        except (ValueError, np.linalg.LinAlgError):
            arma_forecast = np.nan*np.ones([1, fstep])

    # Mixed models (penalized models)
    else:
        # Evaluate all possible mixed configurations of p and q - ARMA(p, q)
        smodel = []
        for p in np.linspace(1, a, a):
            for q in np.linspace(1, b, b):
                try:
                    # Assess model performance on test set
                    arima_model = sm.tsa.ARMA(train_data.ix[xte], order=[int(p), int(q)])
                    arima_result = arima_model.fit(solver='bfgs', disp=False, warn_convergence=False)

                    smodel.append((arima_result.aic, int(p), int(q)))

                except (ValueError, np.linalg.LinAlgError):
                    dummy = 0

        # Create structured array
        smodel = np.array(smodel, dtype=[('score', float), ('p', int), ('q', int)])

        # Sort models by score
        smodel = np.sort(smodel, order='score')
        # -----------------------------------------------------------------------

        # -----------------------------Forecast----------------------------------
        # Check for successful models
        if len(smodel) > 0:

            # Evaluate up to the 4th best score
            if len(smodel) > 4:
                fmodel = 3
            else:
                fmodel = len(smodel) - 1

            # Select model by the Akaike Information Criterion score
            true_prediction = 0
            x = 0

            if fmodel != 0:
                while true_prediction == 0 and x <= fmodel:
                    if len(smodel[x]) > 2:
                        # Configure model and forecast
                        try:
                            arma_model = sm.tsa.ARMA(train_data, order=[smodel[x][1], smodel[x][2]])
                            arma_result = arma_model.fit(solver='bfgs', disp=False, warn_convergence=False)
                            arma_forecast = arma_result.forecast(steps=fstep, alpha=0.05)[0]

                            true_prediction = 1

                        except (ValueError, np.linalg.LinAlgError):
                            x += 1
                            arma_forecast = np.nan*np.ones([1, fstep])
                    else:
                        x += 1
                        arma_forecast = np.nan*np.ones([1, fstep])
            else:
                arma_forecast = np.nan*np.ones([1, fstep])

        # No model fits the data
        else:
            arma_forecast = np.nan*np.ones([1, fstep])

    # Return the user's prompt
    err = err.getvalue()
    sys.stderr = old_stderr

    return arma_forecast


def run_var(train_data, var_step, artificial):
    # Perform Vector Autoregression
    # Usage:
    #           train_data - training data (Pandas Data Frame)
    #           var_step - number of out of sample predictions
    #           artificial - artificial data set flag (Boolean)


    if artificial:
        d_idx = pd.date_range('2015/01/01', periods=len(rf_season), freq='D')
        d_idx = d_idx[train_data.index - (train_data.index[0] - 1)]

        idx_1 = d_idx[:0.5*len(d_idx)]
        idx_2 = d_idx[0.5*len(d_idx):]

        var_model = sm.tsa.VAR(train_data.ix[train_data.index[:0.75*len(d_idx)]], idx_1)
    else:

        idx_1 = train_data.index[:0.5*len(train_data)]
        idx_2 = train_data.index[0.5*len(train_data):]

        var_model = sm.tsa.VAR(train_data.ix[idx_1])

    if not train_data.ix[train_data.index[-1]].isnull().any():

        # Train the classifier and calculate AIC for the test set
        var_order = []
        var_lag = 1
        vsuc = 0
        while var_lag <= 5:

            try:
                # Fit the data (Maximum Lag of 5)
                var_res = var_model.fit(maxlags=var_lag)
                check = var_res.aic

                try:
                    if artificial:
                        m = sm.tsa.VAR(train_data.ix[train_data.index[0.5*len(d_idx):]], idx_2)

                    else:
                        m = sm.tsa.VAR(train_data.ix[idx_2])

                    r = m.fit(maxlags=var_lag)

                    var_order.append((r.aic, var_lag))

                    vsuc = 1
                    var_lag += 1

                except (ValueError, np.linalg.LinAlgError):
                    var_lag += 1

            except (ValueError, np.linalg.LinAlgError):
                var_lag += 1

        if vsuc == 1:
            # Create structured array
            var_order = np.array(var_order, dtype=[('score', float), ('lag', int)])

            # Sort models by score
            var_order = np.sort(var_order, order='score')

            var_suc = 0
            v = 0
            while var_suc == 0 and v <= len(var_order) - 1:

                try:
                    if artificial:
                        vmodel = sm.tsa.VAR(train_data, d_idx)
                    else:
                        vmodel = sm.tsa.VAR(train_data)

                    rmodel = vmodel.fit(maxlags=var_order[v][1])

                    # Forecast var-step
                    var_predict = rmodel.forecast(train_data.values, var_step)

                    var_suc = 1
                except (ValueError, np.linalg.LinAlgError):
                    v += 1
                    var_predict = np.nan*np.ones([1, len(train_data.columns)])

            if var_suc == 0:
                var_predict = np.nan*np.ones([1, len(train_data.columns)])

        else:
            var_predict = np.nan*np.ones([1, len(train_data.columns)])

    else:
        var_predict = np.nan*np.ones([1, len(train_data.columns)])

    return var_predict


def error_calc(dataset, Y_true):
    # Calculate the Sum of Squared Errors
    # Usage:
    #           dataset - predictions
    #           Y_true - correct labels

    dataset.dropna(axis=1, inplace=True)
    u2 = dataset.values

    u1 = Y_true.ix[:, dataset.columns].values

    if len(dataset.columns) >= 1:
        try:
            score = np.sum((u1 - u2)**2)
        except ValueError:
            score = np.nan
    else:
        score = np.nan

    return score


def r2_score(dataset, Y_true):
    # Calculate the coefficient of determination (R^2) as described in Alpaydin (2014)
    # R^2 = 1 - Relative Square Error, RSE = sum of squared errors/sum of squared residuals
    # Usage:
    #           dataset - predictions
    #           Y_true - correct labels

    dataset.dropna(axis=0, inplace=True)

    u1 = Y_true.ix[dataset.index, dataset.columns].values
    u2 = dataset.values

    # Residual sum of squares
    # Evaluate from 0 to "move"
    # RSQ = sum((Y_true - prediction)^2)
    u = np.sum((u1 - u2)**2)

    # Variance = sum((Y_true - Y_mean)^2)
    v = np.sum(np.sum((Y_true.ix[:, dataset.columns] - Y_true.ix[:, dataset.columns].mean())**2))

    try:        # Avoid zero division
        score = 1 - u/v

    except ValueError:
        # The curve is described by a constant and the prediction is accurate
        if u == 0:
            score = 1
        else:
            score = np.nan

    return score


def step_calc(lstep):
    # Calculate the position of the variable steps depending on the total number of steps (odd/even)

        if (lstep % 2) == 0:
            variable = [1]
        else:
            variable = [0]

        increment = 2

        out = 0
        while out == 0:
            if variable[-1] + increment <= lstep - 1:
                variable.append(variable[-1] + increment)
            else:
                out = 1

        if variable[-1] != lstep - 1:
            variable.append(lstep - 1)

        return np.array(variable).reshape(len(variable), 1), 1


# ------------------------------Control Panel--------------------------------
def usage_msg():
    return '''
            Please provide at least one flag for the analysis:
                    -sd: Simulated Data
                    -fi: Financial Indicators
                    '''

parser = argparse.ArgumentParser(description='A comparison of univariate and multivariate methods of forecasting the economy', usage=usage_msg())
parser.add_argument('-sd', action='store_true', dest='run_artificial', default=True,
                    help='Forecast the variables of the Simulated Data')
parser.add_argument('-fi', action='store_true', dest='run_fi', default=False,
                    help='Forecast the Financial Indicators')
args = parser.parse_args()

if args.run_artificial == 0 or args.run_fi == 1:
    run_artificial = 0
else:
    run_artificial = 1

ARIMA = 1
VAR = 1
RF = 1

# ---------------------------------------------------------------------------

# -----------------Constants and Statistical Parameter Thresholds------------
alpha = 0.01  # Dickey-Fuller Stationarity Test

beta = float(80)/float(90)  # Training fraction

n_trees = 100
methods = ['Mean', '1-ARIMA', '1-VAR', '1-RF', 'var-ARIMA', 'var-VAR', 'var-RF', 'h-ARIMA', 'h-VAR', 'h-RF']
# ---------------------------------------------------------------------------

# -----------------------------Parse the Data--------------------------------
# Artificial Dataset
ad_filename = '../data/artificial/artificial_timeseries.csv'
ad_cols = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ad_col_headers = ['idx', 'lt', 'mt', 'wn', 'ar', 'ma', 'cg', 'wf', 'mf', 'vf']

if run_artificial:
    ats, dates, periods = parse_artificial_data(ad_filename,
                                                ad_cols,
                                                ad_col_headers)

# Real Data
rd_filename = '../data/real/complete_dataset.csv'

if not run_artificial:
    ats, dates, periods = parse_data(rd_filename, 0)

# ---------------------------------------------------------------------------

# ----------------------------Allocate Variables-----------------------------
# ---------------RF
rf_hist = pd.DataFrame(index=periods.unique(), columns=np.linspace(1, 33, num=33))
rf_feats = pd.DataFrame(np.zeros([len(periods.unique()), 3*len(ats.columns)]), index=periods.unique(), columns=pd.MultiIndex.from_product([['1-RF', 'var-RF', 'h-RF'], ats.columns.tolist()]))

# ---------------VAR
var_hist = pd.DataFrame(index=periods.unique(), columns=np.linspace(1, 33, num=33))

# ---------------ARIMA
arima_hist = pd.DataFrame(index=periods.unique(), columns=np.linspace(1, 33, num=33))

# ---------------Mean
mean_hist = pd.DataFrame(index=periods.unique(), columns=np.linspace(1, 11, num=11))

# ---------------Track Number of Variable Steps/Quarter
quarter_features = pd.DataFrame(np.zeros([len(periods.unique()), len(ats.columns)]), index=periods.unique(), columns=ats.columns)
# ---------------------------------------------------------------------------

# Analyze the data 90 days at a time
for quarter in periods.unique():

    # Stationary Data Holder
    st_ats = ats.ix[dates[periods == quarter]].copy()

    # Stationary Variable
    st_var = []

    # Non-Stationary Variable
    nst_var = []

    # ---------------------Stationary Feature Selection----------------------
    for feature in ats.columns.values:

        # -----------------------Stationarity Test---------------------------
        # Create Series array of the "feature" column
        series = ats.ix[dates[periods == quarter], feature]
        stn_test, d, stn = evaluate_trend(series)
        # -------------------------------------------------------------------

        # --------------------------Stationary Data--------------------------
        # Check if variable is Stationary
        if stn_test:

            # Save the differenced data. The dth initial values are NaN
            st_ats.loc[dates[periods == quarter], feature] = stn

            # Store the stationary variable name
            st_var.append(feature)
        else:
            # Store the non-stationary variable name
            nst_var.append(feature)
        # -------------------------------------------------------------------
    # -----------------------------------------------------------------------

    # Delete columns that are not Stationary
    if len(st_var) >= 1:
        st_ats.drop(nst_var, inplace=True, axis=1)
    # -----------------------------------------------------------------------

    # ----------------------------Initialize Models--------------------------
    # ------General Settings
    # Divide training and test sets
    season, idx_train, idx_test, n = divide_sets(st_ats.ix[dates[periods == quarter]])

    # Y true values
    y = season.ix[idx_test].copy()

    # Window Size
    steps = len(season) - n      # Possible number of steps in the 90 day interval

    # Variable steps
    var_steps, comp_step = step_calc(steps)

    # Track the number of variable steps for the given quarter
    for feat in st_var:
        quarter_features.loc[quarter, feat] = True
    # ----------------------

    # -------Random Forest
    # Allocation for the retroactive model
    rf_season = season      # Linked (no copy)

    # Allocation for the variable step model
    if run_artificial:
        mid_rf = rf_season.ix[idx_train.tolist() + idx_test[var_steps].tolist()].copy()
    else:
        mid_rf = rf_season.ix[idx_train.tolist() + pd.DatetimeIndex(idx_test[var_steps]).tolist()].copy()

    rf_comp = comp_step

    # Feature importance within the quarter
    rf_qfeats = pd.DataFrame(index=idx_test, columns=pd.MultiIndex.from_product([['1-RF', 'var-RF'], st_var]))
    rf_one = pd.MultiIndex.from_product([['1-RF'], st_var])
    rf_var = pd.MultiIndex.from_product([['var-RF'], st_var])
    # ----------------------

    # -------Vector Auto Regression
    # Allocation for the retroactive model
    var_season = rf_season.copy()

    # Allocation for the variable step model
    mid_var = mid_rf.copy()

    var_comp = comp_step
    # ----------------------

    # -------Autoregressive Integrated Moving Average
    # Allocation for the retroactive model
    arima_season = rf_season.copy()

    arima_pred = pd.DataFrame(columns=y.columns, index=y.index)
    # ----------------------

    # -------Mean
    mean_season = st_ats.ix[dates[periods == quarter]].mean()
    mean_season = pd.DataFrame(np.tile(mean_season, (len(rf_season), 1)), columns=rf_season.columns, index=rf_season.index)
    # ----------------------
    # -----------------------------------------------------------------------

    # ---------------------------------Move Window---------------------------
    # Move one step at a time in the forecasting window
    for move in range(steps):

        # --------------------------------Mean ------------------------------
        # Calculate the prediction error
        mean_hist.loc[quarter, move+1] = error_calc(mean_season.ix[idx_test[move:move+1]], y.ix[idx_test[move:move+1]])

        # Evaluate the Coefficient of Determination
        if move == var_steps[-1]:
            mean_hist.loc[quarter, 11] = r2_score(mean_season.loc[idx_test], y)
        # -------------------------------------------------------------------

        # ------------------ARIMA Forecast and Residual Calculation----------
        if ARIMA:
            for indicator in st_var:
                # Perform model selection, fit and prediction
                arima_season.loc[idx_test[move], indicator] = arma(arima_season.ix[arima_season.index[0:n + move], indicator], 1, run_artificial)

                if move == (steps - 1):
                    # h-step
                    arima_pred.loc[idx_test, indicator] = arma(arima_season.ix[arima_season.index[0:n], indicator], steps, run_artificial)

            # Calculate the prediction score
            arima_hist.loc[quarter, move + 1] = error_calc(arima_season.ix[idx_test[move:move+1]], y.ix[idx_test[move:move+1]])

            if move == (steps - 1):
                # Calculate the prediction score
                for z in range(move + 1):
                    arima_hist.loc[quarter, 23+z] = error_calc(arima_pred.ix[idx_test[z:z+1]], y.ix[idx_test[z:z+1]])
                # --------------------------

                # Evaluate the Determination Coefficient
                arima_hist.loc[quarter, 11] = r2_score(arima_season.ix[idx_test], y)
                arima_hist.loc[quarter, 33] = r2_score(arima_pred, y)
        # -------------------------------------------------------------------

        # ---------------------Vector Auto Regression (VAR)------------------
        if VAR:

            # Perform Vector Auto Regression (1-step forecast) and
            # Feed the predicted data back to the training set
            var_season.loc[idx_test[move]] = run_var(var_season.ix[var_season.index[0:n + move]], 1, run_artificial)

            # Calculate the prediction score
            var_hist.loc[quarter, move + 1] = error_calc(var_season.ix[idx_test[move:move+1]], y.ix[idx_test[0:move+1]])

            if move in var_steps:

                # Forecast variable steps
                if move == 0 or move == 1:
                    if run_artificial:
                        mid_var.loc[idx_test[move]] = run_var(mid_var.ix[idx_train.tolist() + idx_test[var_steps[0:var_comp-1]].tolist()], 2**move, run_artificial)[-1]
                    else:
                        mid_var.loc[idx_test[move]] = run_var(mid_var.ix[idx_train.tolist() + pd.DatetimeIndex(idx_test[var_steps[0:var_comp-1]]).tolist()], 2**move, run_artificial)[-1]
                else:
                    if run_artificial:
                        mid_var.loc[idx_test[move]] = run_var(mid_var.ix[idx_train.tolist() + idx_test[var_steps[0:var_comp-1]].tolist()], var_steps[var_comp-1] - var_steps[var_comp-2], run_artificial)[-1]
                    else:
                        mid_var.loc[idx_test[move]] = run_var(mid_var.ix[idx_train.tolist() + pd.DatetimeIndex(idx_test[var_steps[0:var_comp-1]]).tolist()], var_steps[var_comp-1] - var_steps[var_comp-2], run_artificial)[-1]

                # Calculate the prediction score
                var_hist.loc[quarter, 11 + var_comp] = error_calc(mid_var.ix[idx_test[move:move+1]], y.ix[idx_test[move:move+1]])

                var_comp += 1

            if move == (steps - 1):
                # h-step -------------------
                var_pred = run_var(var_season.ix[var_season.index[0:n]], steps, run_artificial)

                if np.isnan(var_pred).all():
                    var_pred = np.repeat(var_pred, len(y.index), 0)

                var_pred = pd.DataFrame(var_pred, columns=y.columns, index=y.index)

                # Calculate the prediction score
                for z in range(move + 1):
                    var_hist.loc[quarter, 23+z] = error_calc(var_pred.ix[idx_test[z:z+1]], y.ix[idx_test[z:z+1]])
                # --------------------------

                # Evaluate the Determination Coefficient
                var_hist.loc[quarter, 11] = r2_score(var_season.ix[idx_test], y)
                var_hist.loc[quarter, 22] = r2_score(mid_var.ix[idx_test], y)
                var_hist.loc[quarter, 33] = r2_score(var_pred, y)
        # -------------------------------------------------------------------

        # -----------------------------Random Forest-------------------------
        if RF:
            wt_offset = 8
            # Build 10 decision trees
            rf_trees = ske.RandomForestRegressor(n_estimators=n_trees)

            # Fit the training data
            rf_trees.fit(rf_season.ix[rf_season.index[0:n+move]].values, rf_season.ix[rf_season.index[0:n+move]].values)

            # Predict the next point and feed it back to the training set (avg of the last three points)
            rf_prediction = rf_trees.predict(np.average(rf_season.ix[rf_season.index[n+move-(1 + wt_offset):n+move-1]].values, axis=0, weights=np.linspace(1,wt_offset,wt_offset))) #rf_season.ix[rf_season.index[n+move-1]].values) #np.mean(rf_season.ix[rf_season.index[n+move-4:n+move-1]].values, axis=1))

            if np.iscomplex(rf_prediction).any():
                rf_hist.loc[quarter, move + 1] = np.nan*np.ones([1, len(st_var)])
            else:
                rf_season.loc[idx_test[move]] = rf_prediction

                # Calculate the prediction score
                rf_hist.loc[quarter, move + 1] = error_calc(rf_season.loc[idx_test[move:move+1]], y.ix[idx_test[0:move+1]])

            # Feature importance
            rf_qfeats.loc[idx_test[move], rf_one] = rf_trees.feature_importances_

            if move in var_steps:
                # Build a tree with a maximum depth of #features/3
                rf_trees = ske.RandomForestRegressor(n_estimators=n_trees)

                # Fit the training data
                if run_artificial:
                    rf_trees.fit(mid_rf.ix[idx_train.tolist() + idx_test[var_steps[0:rf_comp-1]].tolist()].values, mid_rf.ix[idx_train.tolist() + idx_test[var_steps[0:rf_comp-1]].tolist()].values)
                else:
                    rf_trees.fit(mid_rf.ix[idx_train.tolist() + pd.DatetimeIndex(idx_test[var_steps[0:rf_comp-1]]).tolist()].values, mid_rf.ix[idx_train.tolist() + pd.DatetimeIndex(idx_test[var_steps[0:rf_comp-1]]).tolist()].values)

                # Predict the next point and feed it back to the training set
                if move == 0 or move == 1:
                    mid_rf.loc[idx_test[move]] = rf_trees.predict(np.average(mid_rf.ix[idx_train[-wt_offset:]].values, axis=0, weights=np.linspace(1,wt_offset,wt_offset)))
                else:
                    if run_artificial:
                        iidx = np.array(idx_train.tolist() + idx_test[var_steps[0:rf_comp - 2]].tolist())
                    else:
                        iidx = np.array(idx_train.tolist() + pd.DatetimeIndex(idx_test[var_steps[0:rf_comp - 2]]).tolist())

                    mid_rf.loc[idx_test[move]] = rf_trees.predict(np.average(mid_rf.ix[iidx[-wt_offset:]].values, axis=0, weights=np.linspace(1,wt_offset,wt_offset)))

                # Calculate the prediction score
                rf_hist.loc[quarter, 11 + rf_comp] = error_calc(mid_rf.loc[idx_test[move:move+1]], y.ix[idx_test[move:move+1]])

                # Feature importance
                rf_qfeats.loc[idx_test[move], rf_var] = rf_trees.feature_importances_

                rf_comp += 1

            if move == (steps - 1):

                # h-step -------------------
                rf_trees = ske.RandomForestRegressor(n_estimators=n_trees)

                # Fit the training data
                rf_trees.fit(rf_season.ix[mid_rf.index[0:n]], rf_season.ix[mid_rf.index[0:n]])

                # Predict the next point
                hpred = rf_season.ix[rf_season.index[n - wt_offset:n]].values
                for h in np.linspace(wt_offset,1,wt_offset):
                    haux = np.average(hpred[-h:, :], axis=0, weights=np.linspace(1, h, h))
                    hpred = np.vstack((hpred, haux))

                rf_pred = rf_trees.predict(hpred[-len(y.index):])

                rf_pred = pd.DataFrame(rf_pred, columns=y.columns, index=y.index)

                # Calculate the prediction score
                for z in range(steps):
                    rf_hist.loc[quarter, 23+z] = error_calc(rf_pred.ix[idx_test[z:z+1]], y.ix[idx_test[z:z+1]])

                # Feature importance
                rf_feats.loc[quarter, pd.MultiIndex.from_product([['h-RF'], st_var])] = rf_trees.feature_importances_
                # --------------------------

                # Summarize Feature Importance for 1-step and var-step
                rf_feats.loc[quarter, rf_one] = rf_qfeats.ix[:, rf_one].median()
                rf_feats.loc[quarter, rf_var] = rf_qfeats.ix[:, rf_var].dropna(axis=0).median()

                # Evaluate the Determination Coefficient
                rf_hist.loc[quarter, 11] = r2_score(rf_season.ix[idx_test], y)
                rf_hist.loc[quarter, 22] = r2_score(mid_rf.ix[idx_test], y)
                rf_hist.loc[quarter, 33] = r2_score(rf_pred, y)
                # --------------------------
        # -------------------------------------------------------------------
    # -----------------------------------------------------------------------
    print quarter
# ---------------------------------------------------------------------------

##############################

arima_hist = var_hist.copy()
rf_hist = var_hist.copy()

##############################

if run_artificial:
    file_path = '../results/simulated_data/'
else:
    file_path = '../results/findicators/'

# Save Files
rf_hist.to_csv(file_path + 'rf_hist.csv')
rf_feats.to_csv(file_path + 'rf_feats.csv')
var_hist.to_csv(file_path + 'var_hist.csv')
arima_hist.to_csv(file_path + 'arima_hist.csv')
mean_hist.to_csv(file_path + 'mean_hist.csv')
quarter_features.to_csv(file_path + 'quarter_features.csv')

# Plot Settings
pd.options.display.mpl_style = 'default'

# Feature Distribution (Bar plot)
if run_artificial:
    quarter_features.sum(axis=0).plot(kind='barh', figsize=(30, 10))
    plt.xlabel('Features')
    plt.ylabel('Frequency')
    plt.savefig(file_path + 'feat_distr.png', dpi=200)
    plt.close()

# Feature Importance (Summary)
feat_sum = rf_feats.median()

feat_one = feat_sum.ix['1-RF'].copy()
feat_one.sort(ascending=False)
feat_one.to_csv(file_path + 'feat_importance_1rf.csv')

feat_var = feat_sum.ix['var-RF'].copy()
feat_var.sort(ascending=False)
feat_var.to_csv(file_path + 'feat_importance_varrf.csv')

feat_h = feat_sum.ix['h-RF'].copy()
feat_h.sort(ascending=False)
feat_h.to_csv(file_path + 'feat_importance_hrf.csv')

# ---------------------------Summarizing the Data----------------------------
# Create placeholder for the median
median = pd.DataFrame(index=np.linspace(1, 11, 11), columns=methods)

# Store the median of the predictions from the mean
median.loc[:, 'Mean'] = mean_hist.median()

# Store the Direc (retroactive) method step predictions
median.loc[:, '1-ARIMA'] = arima_hist.ix[:, 1:11].median()
median.loc[:, '1-VAR'] = var_hist.ix[:, 1:11].median()
median.loc[:, '1-RF'] = rf_hist.ix[:, 1:11].median()

# Store the variable step method
median.loc[:, 'var-ARIMA'] = arima_hist.ix[:, 12:22].median().values
median.loc[:, 'var-VAR'] = var_hist.ix[:, 12:22].median().values
median.loc[:, 'var-RF'] = rf_hist.ix[:, 12:22].median().values

# Store the direct method
median.loc[:, 'h-ARIMA'] = arima_hist.ix[:, 23:33].median().values
median.loc[:, 'h-VAR'] = var_hist.ix[:, 23:33].median().values
median.loc[:, 'h-RF'] = rf_hist.ix[:, 23:33].median().values

median.to_csv(file_path + 'median.csv')

if run_artificial:
    filenames = [['1step_1st.png', '1step_5th.png', '1step_10th.png'], ['varstep_1st.png', 'varstep_5th.png', 'varstep_10th.png'], ['hstep_1st.png', 'hstep_5th.png', 'hstep_10th.png']]
    cols = [[1, 5, 10], [12, 17], [23, 27, 32]]
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
                        one_step[one_step <= 500000].plot(kind='hist', alpha=0.5, color=color_code[0:sum(check_nan == False)], stacked=True)
                except ValueError:
                    one_step.plot(kind='hist', alpha=0.5, color=color_code[0:sum(check_nan == False)], stacked=True)

                plt.xlabel('Sum of Squared Errors')
                plt.ylabel('Frequency')
                plt.savefig(file_path + filenames[w][p], dpi=200)
                plt.close()
else:
    import model_plots
# ---------------------------------------------------------------------------