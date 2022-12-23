# Out-of-sample R^2
# Produce a function that will compute the out-of-sample R^2
# R^2(OOS) = 1 - (Sum(Yt - Yt|t-1)^2)/(Sum(Yt - muhatt|t-1)^2)
# In other words, R^2 = 1 - SSE/SST

import numpy as np
import pandas as pd
def oos_rsquared(y, yhat, mu):
    y = pd.Series(y)
    yhat = pd.Series(yhat)
    
    sse = np.sum((yhat-y)**2)
    sst = np.sum((y-mu)**2)

    r2 = 1 - sse/sst

    return r2

