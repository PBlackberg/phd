

import numpy as np




def calc_variance(aList, reference_point):
    squared_difference = (aList - reference_point)**2
    variance = squared_difference.mean()
    return variance

def calc_mae(aList, reference_point):
    ''' Mean Absolute Error'''
    abs_difference = np.abs(aList - reference_point)
    mae = (abs_difference).mean()
    return mae

def calc_abs_diff(aList, reference_point):
    ''' Absolute Error'''
    abs_difference = np.abs(aList - reference_point)
    return abs_difference












