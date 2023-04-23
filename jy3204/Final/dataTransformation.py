#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 09:02:35 2023

@author: jackyang
"""

import pandas as pd
from argparse import Namespace
import pickle
import numpy as np

modelParameters = Namespace(
    encodingDictFile = "data/encoding.pkl",
    )

def getProcessParameters():
    if "ProcessParameters" not in modelParameters:
        with open(modelParameters.encodingDictFile, 'rb') as handle:
            params = pickle.load(handle)
            modelParameters.ProcessParameters = params
    return modelParameters.ProcessParameters


def prepareDf(df):
    baseTransformation(df)
    crossTabEncoding(df)
    singleColumnEncoding(df)
    
def singleColumnEncoding(df):
    params = getProcessParameters()
    for singleColumn in params.singleColumnEncoded:
        df[singleColumn] = df.apply(lambda x: 0 if pd.isna(x[singleColumn]) or (x[singleColumn] not in params.dataDict[singleColumn]) else params.dataDict[singleColumn][x[singleColumn]], axis=1)

def baseTransformation(df):
    params = getProcessParameters()
    df["recency"] = df.apply(lambda x: float(x["overlapped_brain_pixel_selections"].split(":")[2]) if x["overlapped_brain_pixel_selections"]!="nulls" else -999 , axis=1)
    df["frequency"] = df.apply(lambda x: float(x["overlapped_brain_pixel_selections"].split(":")[3]) if x["overlapped_brain_pixel_selections"]!="nulls" else -999 , axis=1)
    df["qualifiedPixel"] = df.apply(lambda x: False if x["overlapped_brain_pixel_selections"]=="nulls" else True, axis=1) 
    df["Date"] = df.apply(lambda x:pd.to_datetime(x["imp_timestamp"].split(" ")[0]),axis=1) ##Only for Testing/Training Data usage
    for logCol in params.logColumns:
        df[f'{logCol}_log']=df.apply(lambda x: np.log(x[logCol]+1) if x[logCol]>-999 else -999, axis=1)

def encodeFunc(pairs,encodeDict):
    return lambda x: encodeDict["{}_{}".format(x[pairs[0]],x[pairs[1]])] if "{}_{}".format(x[pairs[0]],x[pairs[1]]) in encodeDict else 0
    
def crossTabEncoding(df,params):
    params = getProcessParameters()
    for pairs in (params.crossTabPairs):
        locLambda = encodeFunc(pairs,params.dataDict[f"{pairs[0]}_{pairs[1]}"])
        df["{}_{}".format(pairs[0],pairs[1])]= df.apply(locLambda, axis = 1)

def getOutputDf(df):
    params = getProcessParameters()
    prepareDf(df)
    print( [_ for _ in  params.numericalColumns+params.categorical+params.informationColumn+[params.targetColumn]  if _ not in df.columns])
    df = df[params.numericalColumns+params.categorical+params.informationColumn+[params.targetColumn]]
    df[params.categorical] = df[params.categorical].astype("category")
    return df

def main(inputCSVFile = None):
    params = getProcessParameters()
    df = pd.read_csv(params.modifiedCSV if inputCSVFile is None else inputCSVFile) 
    df = getOutputDf(df)
    df.to_csv(params.codedOutput, index=False)
    
if __name__ == "__main__":
    main()
