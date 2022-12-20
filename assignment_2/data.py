# TBD: check for missing data (-99.99 or -999)

# Import the necessary libraries
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
from sklearn.metrics import mean_squared_error

def choose_model_via_general_to_specific(dep_variable, ind_variables, data, alpha, mode):

    # General-to-Specific (using p-values)
    industries_to_drop = []
    while True:

        # Creating the concatenation of all independent variables (all industries that haven't been excluded, yet)
        ind_variables_concat = " + ".join(
            x for x in ind_variables if x not in industries_to_drop)
        # print(ind_variables_concat)

        # Creating OLS ("-1" tells function to exclude intercept)
        formula = "{} ~ {} -1".format(dep_variable, ind_variables_concat)
        results = sm.ols(formula=formula, data=data).fit()
        # print(results.summary())

        if mode == "p-value":
            # Dropping industry with highest p-value
            # print(results.pvalues.max())
            if results.pvalues.max() > alpha:
                # print(results.pvalues.idxmax())
                industries_to_drop.append(results.pvalues.idxmax())
            else:
                break
        elif mode == "aic":
            # TBD
            break
        elif mode == "sic":
            # TBD
            break
        else:
            # TBD
            break

    # print(results.summary())
    return results

def split_data_into_train_and_test(data, train_start_date, train_end_date, test_start_date, test_end_date):
    # The dates have to be in the format of YEARMONTH, e.g. 192601 for "January 1926"
    train_data = data[(data["months"] >= train_start_date)
                      & (data["months"] <= train_end_date)]
    test_data = data[(data["months"] >= test_start_date)
                     & (data["months"] <= test_end_date)]
    return [train_data, test_data]

def create_value_factor_data():

    # The Value return is constructed from the 6 Size-Value portfolios as HML = 1/2(SV + BV ) − 1/2(SG + BG)
    size_value_portfolio = pd.read_csv(
        'csv_downloads/6_Portfolios_2x3.CSV', skiprows=15).iloc[:(1171-15), :7].astype(float)

    # Naming the "months" column
    size_value_portfolio = size_value_portfolio.rename(columns={"Unnamed: 0": "months"})

    # Stripping whitespace off of column names
    size_value_portfolio.columns.str.strip()

    # Calculating the value factor
    # HML = 1/2(SV           + BV       ) − 1/2(SG           + BG       )
    # HML = 1/2*(small value + big value) - 1/2(small growth + big growth)
    # High book-to-market: value stocks
    # Low book-to-market: growth stocks
    size_value_portfolio["value_returns"] = 0.5 * (size_value_portfolio["SMALL HiBM"] + size_value_portfolio["BIG HiBM"]) - 0.5 * (size_value_portfolio["SMALL LoBM"] + size_value_portfolio["BIG LoBM"])

    return size_value_portfolio

def create_momentum_factor_data():

    # MOM = 1/2(SH + BH) − 1/2(SL + BL)
    # MOM = 1/2(small high momentum + big high momentum) − 1/2(small low momentum + big low momentum)
    size_momentum_portfolio = pd.read_csv(
        'csv_downloads/6_Portfolios_ME_Prior_12_2.CSV', skiprows=11).iloc[:(1161-11), :7].astype(float)

    # Calculating the momentum factor
    size_momentum_portfolio["momentum_returns"] = 0.5 * (size_momentum_portfolio["SMALL HiPRIOR"] + size_momentum_portfolio["BIG HiPRIOR"]) - \
        0.5 * (size_momentum_portfolio["SMALL LoPRIOR"] + size_momentum_portfolio["BIG LoPRIOR"])

    # Naming the "months" column
    size_momentum_portfolio = size_momentum_portfolio.rename(columns={"Unnamed: 0": "months"})

    return size_momentum_portfolio

def create_industry_data():

    # INDUSTRY PORTFOLIOS

    # Industry data frame
    industry_returns = pd.read_csv("csv_downloads/17_Industry_Portfolios.CSV", skiprows=11).iloc[:(1167-11), :17].astype(float)

    # Stripping whitespace off of column names
    industry_returns = industry_returns.rename(columns=lambda x: x.strip())

    # Naming the "months" column
    industry_returns = industry_returns.rename(columns={"Unnamed: 0": "months"})

    # Putting industry names into list
    industries = industry_returns.columns.values.tolist()[1:]

    return industries, industry_returns

def merge_into_model_data(size_value_portfolio, size_momentum_portfolio, industry_returns):

    model_data = industry_returns

    # Adding the value returns data 
    model_data = pd.merge(model_data, size_value_portfolio[["months", "value_returns"]], on="months", how="outer")

    # Adding the momentum returns data 
    model_data = pd.merge(model_data, size_momentum_portfolio[["months", "momentum_returns"]], on="months", how="outer")

    return model_data

def create_models_and_mses_for_all_periods(alpha, start_period, training_duration, holding_duration, dep_variable, ind_variables):

    # "411" makes sense because we are always adding 4 years and 11 months; if we want to go in increments of 10 years, we use "911"; increments of 20 years: "1911"

    # 194001
    # 194412
    # 194501
    # 194912
    # 195001

    while True:

        # Creating train and test data set (using 89 because from e.g. 194412 to 194501, which is one month later, we need to add 89)
        train_data, test_data = split_data_into_train_and_test(
            data=model_data,
            train_start_date=start_period,
            train_end_date=start_period+training_duration,
            test_start_date=start_period+training_duration+89,
            test_end_date=start_period+training_duration+89+holding_duration
            )

        # Selection best model based on GsT
        results = choose_model_via_general_to_specific(
            dep_variable=dep_variable, ind_variables=ind_variables, data=train_data, alpha=alpha, mode="p-value")

        # Calculating MSE
        true_values = test_data["momentum_returns"]
        prediction_values = results.predict(test_data)
        mse = mean_squared_error(true_values, prediction_values)
        print("Created model w/ start period {}, {} parameters, and MSE of {}".format(start_period, len(results.params), mse))
        
        # Setting new start_period for (potential) next iteration and setting the potential test end date, so break condition can be tested
        start_period=start_period+training_duration+89+holding_duration+89
        potential_test_end_date = start_period+training_duration+89+holding_duration

        # Break condition
        if potential_test_end_date > 202210:
            break


size_value_portfolio = create_value_factor_data()
size_momentum_portfolio = create_momentum_factor_data()
industries, industry_returns = create_industry_data()
model_data = merge_into_model_data(size_value_portfolio, size_momentum_portfolio, industry_returns)

create_models_and_mses_for_all_periods(
    alpha=0.05,
    start_period=194001,
    training_duration=911,
    holding_duration=411,
    dep_variable="value_returns",
    ind_variables=industries
    )

