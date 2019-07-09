#!/usr/bin/env python

# FProphecy
# Point formatting:
# QB:
# -  Pass yd: 1 pt / 25 yd
# -  Pass TD: 4 pt
# Rush:
# -  Rush yd: 1 pt / 10 yd
# -  Rush TD: 6 pt
# Pass:
# -  Catch:    1 pt
# -  Catch yd: 1 pt / 10 yd
# -  Catch TD: 6 pt
# MISC:
# -  Int: -2 pt
# -  Fuml: -2 pt


from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
import numpy as np

data1 = pd.read_csv("data/dlstats_2015_2017.csv")
data2 = pd.read_csv("data/dlstats_2012_2014.csv")
data3 = pd.read_csv("data/dlstats_2009_2011.csv")
data = pd.concat([data1, data2, data3], sort=False)

def total_ppg(row):

    passYd = row[10]
    passTD = row[11]
    rushYd = row[5]
    rushTD = row[6]
    catch = row[7]
    catchYd = row[8]
    catchTD = row[9]
    int = row[12]
    fuml = row[13]
    
    fppg = (passYd / 25 + passTD * 4 + rushYd / 10 + rushTD * 6 + catch + catchYd / 10 + catchTD + int * -2 + fuml * -2) / row[4]
    
    return fppg

def fix_names(row):

    name = row[0]

    if not ("," in name):
        name = row[0]
        name = "{}, {}".format(name[name.find(" ")+1::], name[:name.find(" "):])
        
    return name

# Dropping unnecessary columns and filling all null values with 0
data.drop('Rush', axis=1, inplace=True)
data.drop('Target', axis=1, inplace=True)
data.drop('Pass', axis=1, inplace=True)
data.drop('Complete', axis=1, inplace=True)
data.drop(data.columns[data.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
data.fillna(0, inplace=True)

# Creating column containing the total fantasy points that player made
data['Fantasy Points Per Game'] = data.apply(lambda row: total_ppg(row), axis=1)
data['Name'] = data.apply(lambda row: fix_names(row), axis=1)

data = data.sort_values(by=["Year"])

print(data.loc[data['Name'] == "Fitzgerald, Larry"])
dataLarry = [13.11, 15.525, 17.0925, 19.035, 15.50625, 17.64125, 14.49625, 21.46625, 14.25625]

# Fitting and checking an ARIMA time series on Larry Fitzgerald
model = ARIMA(dataLarry, order=(1,1,1))
model_fit = model.fit(disp=False)

yhat = model_fit.predict(len(dataLarry), len(dataLarry))
print(yhat)