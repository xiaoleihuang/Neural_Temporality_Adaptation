'''
This script is to compute the correlation between w. distances and overlaps
'''

import json
import pickle
import os
from scipy.stats import pearsonr

amazon = {
    'usage':[
        0.538, 0.530, 0.539, 0.531, 0.524, 0.798, 0.802, 
        0.766, 0.708, 0.839, 0.783, 0.700, 0.811, 0.749, 0.764],
    'ctt':[
        0.110, 0.070, 0.085, 0.111, 0.140, 0.206, 0.211, 0.208,
        0.198, 0.208, 0.184, 0.162, 0.208, 0.188, 0.216],
    'dist_mi':[
        0.408, 0.518, 0.441, 0.356, 0.309, 0.181, 0.117, 
        0.113, 0.129, 0.133, 0.182, 0.250, 0.134, 0.198, 0.118],
    'dist_freq':[.0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0],
    'dist_mi_re':[
        0.484, 0.592, 0.527, 0.467, 0.387, 0.163, 0.107, 
        0.097, 0.115, 0.114, 0.165, 0.233, 0.140, 0.179, 0.114],
    'dist_freq_re':[
        0.455, 0.557, 0.491, 0.403, 0.339, 0.176, 0.122, 0.117,
        0.143, 0.138, 0.193, 0.270, 0.141, 0.208, 0.130],
}

print('Amazon')
print('Usage, Dist_mi', pearsonr(amazon['usage'], amazon['dist_mi']))
print('Context, Dist_mi', pearsonr(amazon['ctt'], amazon['dist_mi']))
print('Usage, Context', pearsonr(amazon['usage'], amazon['ctt']))
print('Usage, Dist_mi_re', pearsonr(amazon['usage'], amazon['dist_mi_re']))
print('Context, Dist_mi_re', pearsonr(amazon['ctt'], amazon['dist_mi_re']))
print()


dianping = {
    'usage':[0.842, 0.815, 0.765, 0.899, 0.809, 0.848],
    'ctt':[0.316, 0.264, 0.374, 0.367, 0.276, 0.235],
    'dist_mi':[0.134, 0.177, 0.055, 0.088, 0.162, 0.207],
    'dist_freq':[.0, .0, .0, .0, .0, .0],
    'dist_mi_re':[0.119, 0.140, 0.047, 0.061, 0.144, 0.173],
    'dist_freq_re':[0.141, 0.181, 0.066, 0.108, 0.157, 0.204],
}

print('Dianping')
print('Usage, Dist_mi', pearsonr(dianping['usage'], dianping['dist_mi']))
print('Context, Dist_mi', pearsonr(dianping['ctt'], dianping['dist_mi']))
print('Usage, Context', pearsonr(dianping['usage'], dianping['ctt']))
print('Usage, Dist_mi_re', pearsonr(dianping['usage'], dianping['dist_mi_re']))
print('Context, Dist_mi_re', pearsonr(dianping['ctt'], dianping['dist_mi_re']))
print()


economy = {
    'usage':[
        0.380, 0.363, 0.354, 0.373, 0.340, 0.384, 0.382, 
        0.371, 0.324, 0.377, 0.368, 0.322, 0.388, 0.359, 0.378],
    'ctt':[
        0.356, 0.342, 0.345, 0.337, 0.336, 0.353, 0.358,
        0.348, 0.342, 0.362, 0.349, 0.343, 0.378, 0.363, 0.377],
    'dist_mi':[
        0.062, 0.075, 0.037, 0.063, 0.056, 0.031, 0.040, 
        0.062, 0.058, 0.059, 0.073, 0.058, 0.058, 0.054, 0.059],
    'dist_freq':[
        .001, .001, .0, .001, .0, .001, .0, .001, .001, .0, 
        .001, .001, .001, .0, .001],
    'dist_mi_re':[
        0.006, 0.010, 0.014, 0.009, 0.011, 0.008, 0.012, 
        0.010, 0.012, 0.018, 0.009, 0.007, 0.021, 0.023, 0.006],
    'dist_freq_re':[
        0.069, 0.084, 0.036, 0.070, 0.059, 0.034, 0.048, 0.070, 0.066, 0.063, 
        0.081, 0.067, 0.065, 0.057, 0.065],
}

print('Economy')
print('Usage, Dist_mi', pearsonr(economy['usage'], economy['dist_mi']))
print('Context, Dist_mi', pearsonr(economy['ctt'], economy['dist_mi']))
print('Usage, Context', pearsonr(economy['usage'], economy['ctt']))
print('Usage, Dist_mi_re', pearsonr(economy['usage'], economy['dist_mi_re']))
print('Context, Dist_mi_re', pearsonr(economy['ctt'], economy['dist_mi_re']))
print()


vaccine = {
    'usage':[0.482, 0.454, 0.349, 0.493, 0.377, 0.378],
    'ctt':[0.326, 0.317, 0.207, 0.327, 0.209, 0.229],
    'dist_mi':[0.063, 0.049, 0.074, 0.075, 0.048, 0.067],
    'dist_freq':[0.064, 0.075, 0.074, 0.042, 0.027, 0.034],
    'dist_mi_re':[0.058, 0.048, 0.070, 0.063, 0.050, 0.059],
    'dist_freq_re':[0.064, 0.059, 0.080, 0.077, 0.054, 0.069],
}

print('Vaccine')
print('Usage, Dist_mi', pearsonr(vaccine['usage'], vaccine['dist_mi']))
print('Context, Dist_mi', pearsonr(vaccine['ctt'], vaccine['dist_mi']))
print('Usage, Context', pearsonr(vaccine['usage'], vaccine['ctt']))
print('Usage, Dist_mi_re', pearsonr(vaccine['usage'], vaccine['dist_mi_re']))
print('Context, Dist_mi_re', pearsonr(vaccine['ctt'], vaccine['dist_mi_re']))
print()


yelp_hotel = {
    'usage':[0.629, 0.585, 0.580, 0.826, 0.790, 0.880],
    'ctt':[0.215, 0.176, 0.177, 0.248, 0.243, 0.251],
    'dist_mi':[0.227, 0.326, 0.335, 0.177, 0.164, 0.071],
    'dist_freq':[.001, .0, .0, .001, .001, .0],
    'dist_mi_re':[0.360, 0.441, 0.461, 0.160, 0.151, 0.053],
    'dist_freq_re':[0.229, 0.321, 0.333, 0.183, 0.177, 0.074],
}

print('Yelp-hotel')
print('Usage, Dist_mi', pearsonr(yelp_hotel['usage'], yelp_hotel['dist_mi']))
print('Context, Dist_mi', pearsonr(yelp_hotel['ctt'], yelp_hotel['dist_mi']))
print('Usage, Context', pearsonr(yelp_hotel['usage'], yelp_hotel['ctt']))
print('Usage, Dist_mi_re', pearsonr(yelp_hotel['usage'], yelp_hotel['dist_mi_re']))
print('Context, Dist_mi_re', pearsonr(yelp_hotel['ctt'], yelp_hotel['dist_mi_re']))
print()


yelp_rest = {
    'usage':[0.767, 0.723, 0.686, 0.879, 0.818, 0.911],
    'ctt':[0.163, 0.124, 0.123, 0.211, 0.207, 0.219],
    'dist_mi':[0.412, 0.427, 0.456, 0.163, 0.137, 0.110],
    'dist_freq':[.0, .0, .0, .0, .0, .0],
    'dist_mi_re':[0.385, 0.411, 0.444, 0.148, 0.120, 0.106],
    'dist_freq_re':[0.443, 0.462, 0.481, 0.170, 0.137, 0.119],
}

print('Yelp-rest')
print('Usage, Dist_mi', pearsonr(yelp_rest['usage'], yelp_rest['dist_mi']))
print('Context, Dist_mi', pearsonr(yelp_rest['ctt'], yelp_rest['dist_mi']))
print('Usage, Context', pearsonr(yelp_rest['usage'], yelp_rest['ctt']))
print('Usage, Dist_mi_re', pearsonr(yelp_rest['usage'], yelp_rest['dist_mi_re']))
print('Context, Dist_mi_re', pearsonr(yelp_rest['ctt'], yelp_rest['dist_mi_re']))
print()

