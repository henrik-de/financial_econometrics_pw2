# TBD: check for missing data (-99.99 or -999)

# Import the necessary libraries
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm



# VALUE FACTOR

# The Value return is constructed from the 6 Size-Value portfolios as HML = 1/2(SV + BV ) − 1/2(SG + BG)
size_value = pd.read_csv(
    'csv_downloads/6_Portfolios_2x3.CSV', skiprows=15).iloc[:(1171-15), :7].astype(float)

# Naming the "months" column
size_value = size_value.rename(columns={"Unnamed: 0": "months"})

# Stripping whitespace off of column names
size_value.columns.str.strip()

# Calculating the value factor
# HML = 1/2(SV           + BV       ) − 1/2(SG           + BG       )
# HML = 1/2*(small value + big value) - 1/2(small growth + big growth)
# High book-to-market: value stocks
# Low book-to-market: growth stocks
size_value["value_returns"] = 0.5 * (size_value["SMALL HiBM"] + size_value["BIG HiBM"]) - 0.5 * (size_value["SMALL LoBM"] + size_value["BIG LoBM"])



# MOMENTUM FACTOR

# MOM = 1/2(SH + BH) − 1/2(SL + BL)
# MOM = 1/2(small high momentum + big high momentum) − 1/2(small low momentum + big low momentum)
size_momentum = pd.read_csv(
    'csv_downloads/6_Portfolios_ME_Prior_12_2.CSV', skiprows=11).iloc[:(1161-11), :7].astype(float)

# Calculating the momentum factor
size_momentum["momentum_returns"] = 0.5 * (size_momentum["SMALL HiPRIOR"] + size_momentum["BIG HiPRIOR"]) - \
    0.5 * (size_momentum["SMALL LoPRIOR"] + size_momentum["BIG LoPRIOR"])

# Naming the "months" column
size_momentum = size_momentum.rename(columns={"Unnamed: 0": "months"})



# INDUSTRY PORTFOLIOS

# Industry data frame
industry_returns = pd.read_csv("csv_downloads/17_Industry_Portfolios.CSV", skiprows=11).iloc[:(1167-11), :17].astype(float)


# Stripping whitespace off of column names
industry_returns = industry_returns.rename(columns=lambda x: x.strip())

print(industry_returns.columns)


# Naming the "months" column
industry_returns = industry_returns.rename(columns={"Unnamed: 0": "months"})

# Putting industry names into list
industries = industry_returns.columns.values.tolist()[1:]



# MODEL

model_data = industry_returns

# Adding the value returns data 
model_data = pd.merge(
    model_data, size_value[["months", "value_returns"]], on="months", how="outer")

# Adding the momentum returns data 
model_data = pd.merge(
    model_data, size_momentum[["months", "momentum_returns"]], on="months", how="outer")

# print(model_data)

# Creating OLS ("-1" tells function to exclude intercept)
ind_concat = " + ".join(industries)
formula = "value_returns ~ {} -1".format(ind_concat)
print(formula)
result = sm.ols(formula=formula, data=model_data).fit()

print(result.summary())







