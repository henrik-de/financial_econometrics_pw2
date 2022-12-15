"""
This file contains very basic templates for the functions that should be provided
in PW2. Feel free to use this file as a starting point for your code. You can
test this using the demo autograder by placing this file and demo-autograder-pw2.py
in the same folder, and then running

python demo-autograder-pw2.py

If you see nothing, then things are working (but you still need to check your
answers are correct).
"""

import numpy as np
import pandas as pd


def oos_rsquared(y, yhat, mu):
    return 0.0


def oos_residuals(y, x, beta, first, last):
    return y[first:last]


def hypothesis_tests(y, unrestricted, restricted):
    return 0.0, 0.0
