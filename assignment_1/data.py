import pandas as pd

# 1. The momentum factor from Ken French’s website (monthly).
# Haven't removed the table below (years)
momentum_factor_monthly = pd.read_csv(
    "csv_downloads/F-F_Momentum_Factor.CSV", skiprows=13)

# 2. The 1, 5, and 10-year constant maturity series from FRED.
one_year_yield_monthly = pd.read_csv("csv_downloads/GS1.csv")
five_year_yield_monthly = pd.read_csv("csv_downloads/GS5.csv")
ten_year_yield_monthly = pd.read_csv("csv_downloads/GS10.csv")

# 3. The AAA and BAA (Moody’s) from FRED.
aaa_yield_monthly = pd.read_csv("csv_downloads/AAA.csv")
baa_yield_monthly = pd.read_csv("csv_downloads/BAA.csv")

# 4. The monthly returns on the VWM from Ken French’s site.
# VWM = HML
# Haven't removed the table below (years)
vwm_monthly = pd.read_csv("csv_downloads/F-F_Research_Data_Factors.CSV", skiprows=3)

# 5. Monthly data on core CPI from FRED.
core_cpi_monthly = pd.read_csv("csv_downloads/CPILFESL.csv")

# 6. Monthly data in the unemployment rate from FRED.
unemployment_monthly = pd.read_csv("csv_downloads/UNRATE.csv")

# 7. Monthly data on Industrial Productivity from FRED.
industrial_productivity_monthly = pd.read_csv("csv_downloads/INDPRO.csv")

print(industrial_productivity_monthly)