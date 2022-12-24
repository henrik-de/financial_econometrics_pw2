import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
from sklearn.metrics import mean_squared_error

# 1. The momentum factor from Ken French’s website (monthly).
# Removed the table below (years)
momentum_factor_monthly = pd.read_csv("csv_downloads/F-F_Momentum_Factor.CSV")
momentum_factor_monthly['Unnamed: 0'] = pd.to_datetime(momentum_factor_monthly['Unnamed: 0'],format='%Y%m').dt.date
momentum_factor_monthly = momentum_factor_monthly.rename(columns={'Unnamed: 0':'DATE'})

# 2. The 1, 5, and 10-year constant maturity series from FRED.
one_year_yield_monthly = pd.read_csv("csv_downloads/GS1.csv")
one_year_yield_monthly['DATE'] = pd.to_datetime(one_year_yield_monthly['DATE'],format='%Y-%m-%d').dt.date

five_year_yield_monthly = pd.read_csv("csv_downloads/GS5.csv")
five_year_yield_monthly['DATE'] = pd.to_datetime(five_year_yield_monthly['DATE'],format='%Y-%m-%d').dt.date

ten_year_yield_monthly = pd.read_csv("csv_downloads/GS10.csv")
ten_year_yield_monthly['DATE'] = pd.to_datetime(ten_year_yield_monthly['DATE'],format='%Y-%m-%d').dt.date

# 3. The AAA and BAA (Moody’s) from FRED.
aaa_yield_monthly = pd.read_csv("csv_downloads/AAA.csv")
aaa_yield_monthly['DATE'] = pd.to_datetime(aaa_yield_monthly['DATE'],format='%Y-%m-%d').dt.date

baa_yield_monthly = pd.read_csv("csv_downloads/BAA.csv")
baa_yield_monthly['DATE'] = pd.to_datetime(baa_yield_monthly['DATE'],format='%Y-%m-%d').dt.date

# 4. The monthly returns on the VWM from Ken French’s site.
# VWM = Mkt-RF -- renamed in original file to VWM
# Remove table below in the data file (years)
vwm_monthly = pd.read_csv("csv_downloads/F-F_Research_Data_Factors.CSV", skiprows=3)
vwm_monthly['Unnamed: 0'] = pd.to_datetime(vwm_monthly['Unnamed: 0'],format='%Y%m').dt.date
vwm_monthly = vwm_monthly.rename(columns={'Unnamed: 0':'DATE'})
#print(vwm_monthly)

# 5. Monthly data on core CPI from FRED.
core_cpi_monthly = pd.read_csv("csv_downloads/CPILFESL.csv")
core_cpi_monthly['DATE'] = pd.to_datetime(core_cpi_monthly['DATE'],format='%Y-%m-%d').dt.date

# 6. Monthly data in the unemployment rate from FRED.
unemployment_monthly = pd.read_csv("csv_downloads/UNRATE.csv")
unemployment_monthly['DATE'] = pd.to_datetime(unemployment_monthly['DATE'],format='%Y-%m-%d').dt.date

# 7. Monthly data on Industrial Productivity from FRED.
industrial_productivity_monthly = pd.read_csv("csv_downloads/INDPRO.csv")
industrial_productivity_monthly['DATE'] = pd.to_datetime(industrial_productivity_monthly['DATE'],format='%Y-%m-%d').dt.date


## Variable creation
# TERM - The difference between the 10-year and the 1-year
term = pd.DataFrame()
term['DATE']= ten_year_yield_monthly['DATE']
term['Term'] = ten_year_yield_monthly['GS10'] - one_year_yield_monthly['GS1']
#print(ten_year_yield_monthly)
#print(term)

# CURVE - The 10-year yield plus the 1-year yield minus 2 times the 5-year yield
curve = pd.DataFrame()
curve['DATE'] = ten_year_yield_monthly['DATE']
curve['Curve'] = ten_year_yield_monthly['GS10'] + one_year_yield_monthly['GS1'] - 2*five_year_yield_monthly['GS5']
#print(curve)

# DEFAULT - The difference between the AAA and the BAA yields
default = pd.DataFrame()
default['DATE'] = aaa_yield_monthly['DATE']
default['Default'] = aaa_yield_monthly['AAA'] - baa_yield_monthly['BAA']
#print(default)

# INFLATION - The year-overyear difference of log CPI
inflation = pd.DataFrame()
#print(core_cpi_monthly)
log_inf_lag_12 = np.log(core_cpi_monthly['CPILFESL'].shift(periods=12))
#print(log_inf_lag_12)
inflation['DATE']=core_cpi_monthly['DATE']
inflation['Inflation']= np.log(core_cpi_monthly['CPILFESL']) - log_inf_lag_12
#print(inflation)


## Merging data
merged_data = default
merged_data = pd.merge(merged_data, term, on = 'DATE', how = 'outer')
merged_data = pd.merge(merged_data, curve, on = 'DATE', how = 'outer')
merged_data = pd.merge(merged_data, inflation, on = 'DATE', how = 'outer')
merged_data = pd.merge(merged_data, unemployment_monthly, on = 'DATE', how = 'outer')
merged_data = pd.merge(merged_data, industrial_productivity_monthly, on = 'DATE', how = 'outer')
merged_data = pd.merge(merged_data, vwm_monthly, on = 'DATE', how = 'outer')
merged_data = pd.merge(merged_data, momentum_factor_monthly, on = 'DATE', how = 'outer')
#print(merged_data)


## Regression on non-lagged data
# Creating the concatenation of all independent variables
#to_drop = ['DATE','VWM','SMB','HML','RF','Mom']
#ind_variables_concat = " + ".join(x for x in merged_data if x not in to_drop)
#print(ind_variables_concat)

# Creating OLS
#formula = "{} ~ {}".format('Mom', ind_variables_concat)
#print(formula)
#results = sm.ols(formula=formula, data=merged_data).fit()
#print(results.summary())

## Making lagged independent variables by 1 period 
def_lag = pd.DataFrame()
def_lag['DATE'] = default['DATE']
def_lag['Def_lag'] = default['Default'].shift(periods=1)
#print(def_lag)

term_lag = pd.DataFrame()
term_lag['DATE'] = term['DATE']
term_lag['Term_lag'] = term['Term'].shift(periods=1)
#print(term_lag)

curve_lag = pd.DataFrame()
curve_lag['DATE'] = curve['DATE']
curve_lag['Curve_lag'] = curve['Curve'].shift(periods=1)
#print(curve_lag)

inf_lag = pd.DataFrame()
inf_lag['DATE'] = inflation['DATE']
inf_lag['Inf_lag'] = inflation['Inflation'].shift(periods=1)
#print(inf_lag)

unemp_lag = pd.DataFrame()
unemp_lag['DATE'] = unemployment_monthly['DATE']
unemp_lag['Unemp_lag'] = unemployment_monthly['UNRATE'].shift(periods=1)
#print(unemp_lag)

ind_pro_lag = pd.DataFrame()
ind_pro_lag['DATE'] = industrial_productivity_monthly['DATE']
ind_pro_lag['Ind_pro_lag'] = industrial_productivity_monthly['INDPRO'].shift(periods=1)
#print(ind_pro_lag)

vwm_monthly_lag = pd.DataFrame()
vwm_monthly_lag['DATE'] = vwm_monthly['DATE']
vwm_monthly_lag['VWM_lag'] = vwm_monthly['VWM'].shift(periods=1)
vwm_monthly_lag['SMB_lag'] = vwm_monthly['SMB'].shift(periods=1)
vwm_monthly_lag['HML_lag'] = vwm_monthly['HML'].shift(periods=1)
vwm_monthly_lag['RF_lag'] = vwm_monthly['RF'].shift(periods=1)
#print(vwm_monthly_lag)

momentum_factor_monthly_lag = pd.DataFrame()
momentum_factor_monthly_lag['DATE'] = momentum_factor_monthly['DATE']
momentum_factor_monthly_lag['Mom_lag'] = momentum_factor_monthly['Mom'].shift(periods=1)
#print(momentum_factor_monthly_lag)

## Merging independent variables
ind_var_lagged = def_lag
ind_var_lagged = pd.merge(ind_var_lagged, term_lag, on = 'DATE', how = 'outer')
ind_var_lagged = pd.merge(ind_var_lagged, curve_lag, on = 'DATE', how = 'outer')
ind_var_lagged = pd.merge(ind_var_lagged, inf_lag, on = 'DATE', how = 'outer')
ind_var_lagged = pd.merge(ind_var_lagged, unemp_lag, on = 'DATE', how = 'outer')
ind_var_lagged = pd.merge(ind_var_lagged, ind_pro_lag, on = 'DATE', how = 'outer')
ind_var_lagged = pd.merge(ind_var_lagged, vwm_monthly_lag, on = 'DATE', how = 'outer')
ind_var_lagged = pd.merge(ind_var_lagged, momentum_factor_monthly_lag, on = 'DATE', how = 'outer')
ind_var_lagged = pd.merge(ind_var_lagged, vwm_monthly, on = 'DATE', how = 'outer')
ind_var_lagged = pd.merge(ind_var_lagged, momentum_factor_monthly, on = 'DATE', how = 'outer')
ind_var_lagged = ind_var_lagged.dropna()
#print(ind_var_lagged)

## Regression with lagged variables period 1
# Creating the concatenation of all independent variables
# If dep_var is VWM, drop Mom_lag; if dep_var is Mom, drop VWM_lag
to_drop = ['DATE','VWM','SMB', 'SMB_lag','HML', 'HML_lag', 'RF', 'RF_lag','Mom','Mom_lag']
ind_variables_concat = " + ".join(x for x in ind_var_lagged if x not in to_drop)
#print(ind_variables_concat)

# Creating OLS
formula = "{} ~ {}".format('VWM', ind_variables_concat)
#print(formula)
results = sm.ols(formula=formula, data=ind_var_lagged).fit()
#print(results.summary())


## Making lagged independent variables by 12 periods
def_lag_12 = pd.DataFrame()
def_lag_12['DATE'] = default['DATE']
def_lag_12['Def_lag_12'] = default['Default'].shift(periods=12)
#print(def_lag_12)

term_lag_12 = pd.DataFrame()
term_lag_12['DATE'] = term['DATE']
term_lag_12['Term_lag_12'] = term['Term'].shift(periods=12)
#print(term_lag_12)

curve_lag_12 = pd.DataFrame()
curve_lag_12['DATE'] = curve['DATE']
curve_lag_12['Curve_lag_12'] = curve['Curve'].shift(periods=12)
#print(curve_lag)

inf_lag_12 = pd.DataFrame()
inf_lag_12['DATE'] = inflation['DATE']
inf_lag_12['Inf_lag_12'] = inflation['Inflation'].shift(periods=12)
#print(inf_lag_12)

unemp_lag_12 = pd.DataFrame()
unemp_lag_12['DATE'] = unemployment_monthly['DATE']
unemp_lag_12['Unemp_lag_12'] = unemployment_monthly['UNRATE'].shift(periods=12)
#print(unemp_lag_12)

ind_pro_lag_12 = pd.DataFrame()
ind_pro_lag_12['DATE'] = industrial_productivity_monthly['DATE']
ind_pro_lag_12['Ind_pro_lag'] = industrial_productivity_monthly['INDPRO'].shift(periods=12)
#print(ind_pro_lag_12)

vwm_monthly_lag_12 = pd.DataFrame()
vwm_monthly_lag_12['DATE'] = vwm_monthly['DATE']
vwm_monthly_lag_12['VWM_lag_12'] = vwm_monthly['VWM'].shift(periods=12)
vwm_monthly_lag_12['SMB_lag_12'] = vwm_monthly['SMB'].shift(periods=12)
vwm_monthly_lag_12['HML_lag_12'] = vwm_monthly['HML'].shift(periods=12)
vwm_monthly_lag_12['RF_lag_12'] = vwm_monthly['RF'].shift(periods=12)
#print(vwm_monthly_lag_12)

momentum_factor_monthly_lag_12 = pd.DataFrame()
momentum_factor_monthly_lag_12['DATE'] = momentum_factor_monthly['DATE']
momentum_factor_monthly_lag_12['Mom_lag_12'] = momentum_factor_monthly['Mom'].shift(periods=12)
#print(momentum_factor_monthly_lag_12)

## Merging independent variables
ind_var_lagged_12 = def_lag_12
ind_var_lagged_12 = pd.merge(ind_var_lagged_12, term_lag_12, on = 'DATE', how = 'outer')
ind_var_lagged_12 = pd.merge(ind_var_lagged_12, curve_lag_12, on = 'DATE', how = 'outer')
ind_var_lagged_12 = pd.merge(ind_var_lagged_12, inf_lag_12, on = 'DATE', how = 'outer')
ind_var_lagged_12 = pd.merge(ind_var_lagged_12, unemp_lag_12, on = 'DATE', how = 'outer')
ind_var_lagged_12 = pd.merge(ind_var_lagged_12, ind_pro_lag_12, on = 'DATE', how = 'outer')
ind_var_lagged_12 = pd.merge(ind_var_lagged_12, vwm_monthly_lag_12, on = 'DATE', how = 'outer')
ind_var_lagged_12 = pd.merge(ind_var_lagged_12, momentum_factor_monthly_lag_12, on = 'DATE', how = 'outer')
ind_var_lagged_12 = pd.merge(ind_var_lagged_12, vwm_monthly, on = 'DATE', how = 'outer')
ind_var_lagged_12 = pd.merge(ind_var_lagged_12, momentum_factor_monthly, on = 'DATE', how = 'outer')
ind_var_lagged_12 = ind_var_lagged_12.dropna()
#print(ind_var_lagged_12)

## Regression with lagged variables period 12
# Creating the concatenation of all independent variables
# If dep_var is VWM, drop Mom_lag; if dep_var is Mom, drop VWM_lag
to_drop_l12 = ['DATE','VWM','SMB', 'SMB_lag_12','HML', 'HML_lag_12', 'RF', 'RF_lag_12','Mom','Mom_lag_12']
ind_variables_concat_l12 = " + ".join(x for x in ind_var_lagged_12 if x not in to_drop_l12)
#print(ind_variables_concat_l12)

# Creating OLS
formula = "{} ~ {}".format('VWM', ind_variables_concat_l12)
#print(formula)
results = sm.ols(formula=formula, data=ind_var_lagged_12).fit()
#print(results.summary())


## Quick testing individual variables in formt ('dependent_variable', 'independent_variable')
form = "{}~{}".format('VWM', 'VWM_lag_12')
test_result = sm.ols(formula = form, data = ind_var_lagged_12).fit()
#print(test_result.summary())


## Trying dependent variable in log returns
## Did not work like this due to negative ln for negative returns being NaN
#log_mom = pd.DataFrame()
#log_mom['DATE'] = momentum_factor_monthly['DATE']
#print(momentum_factor_monthly)
#log_mom['Mom_log'] = np.log(momentum_factor_monthly['Mom'])
#print(log_mom)


## Trying a rolling regression
# Set the window size for the rolling regression
window_size = 60

# Define variables
# Remember to go back to ind_variables_concat_l12 to drop other variables when changing dependent_var
dependent_var = 'VWM'
rolling_formula = "{}~{}".format(dependent_var, ind_variables_concat_l12)

# Perform the rolling regression
results = []
for i in range(window_size, len(ind_var_lagged_12)):
    subset = ind_var_lagged_12.iloc[i-window_size:i]
    model = sm.ols(rolling_formula, data=subset).fit()
    results.append(model)
# Print the summary of the first and last window
#print(results[0].summary())
#print(results[-1].summary())


## printing correlation table
corr_matrix = ind_var_lagged.corr()
#print(corr_matrix)


## Calculating Sharpe ratios
#Sharpe ratio = (Return of portfolio - risk free rate of return)/standard deviation of the portfolio's return
# A general formula to calculate Sharpe ratio

def calculate_sharpe_ratio(returns, risk_free_rate):
    # Calculate exces returns
    excess_returns = returns - risk_free_rate

    # Calculate the standard deviation of excess returns
    std_excess_returns = np.std(excess_returns)

    # Calculate the mean of the excess returns
    mean_excess_returns = np.mean(excess_returns)

    # Calculate the Sharpe ratio
    sharpe_ratio = mean_excess_returns / std_excess_returns

    return sharpe_ratio

# Load input data into the formula
# Can use this formula for all portfolios and future data set with our own portfolio returns
# Need to have a merged data set to run this; will merge own portfolio when available
sharpe_ratio_data = vwm_monthly
sharpe_ratio_data = pd.merge(sharpe_ratio_data, momentum_factor_monthly, on = 'DATE', how = 'outer')
sharpe_ratio_data = sharpe_ratio_data.dropna()
#print(sharpe_ratio_data)

# Input VWM, Mom or our own portfolio
portfolio = 'VWM'
sharpe_ratio = calculate_sharpe_ratio(sharpe_ratio_data[portfolio],sharpe_ratio_data['RF'])

print(f'The Sharpe ratio of the {portfolio} portfolio is: {sharpe_ratio:.2f}')
 