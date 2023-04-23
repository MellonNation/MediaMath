#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 09:02:35 2023

@author: jackyang
"""

import pandas as pd
import json
import re
from argparse import Namespace
from tqdm import tqdm
import pickle

modelParameters = Namespace(
    rawCSV = "data/roi_data.csv",
    contextual_data = "data/context.json",
    dataParameterCoverage = 50,
    includeColumns = [],
    encodingDictFile = "data/encoding.pkl",
    dataDict = {},
    logColumns = ['exchange_viewability_rate',"recency"],
    numericalColumns = ['exchange_viewability_rate_log', 'exchange_ctr','recency_log', 'frequency'],
    categorical = [  'qualifiedPixel', 'browser_version_device_model','hashed_app_id_device_model',
                     'deal_id','category_id','dma_id','browser_language_id','region_id',
                     'exchange_id','isp_id','impression_date','size','day_of_week','creative_id',
                     'id_vintage','device_id','day_part','conn_speed','fold_position','channel_type',
                     'num_device_ids','country_id','cross_device_flag','base_domain_device_model','site_id_device_model',
                     'publisher_id_device_model',"device_model", "os_version","browser_version","device_type" ],
    crossTabPairs = [['browser_version','device_model'],
                   ["hashed_app_id",'device_model'],
                   ["base_domain",'device_model'],
                   ["site_id",'device_model'],
                   ["publisher_id",'device_model']],
    informationColumn = ["Date"],
    singleColumnEncoded = ["device_model", "os_version","browser_version","device_type"],
    targetColumn = 'conversion_roi',
    codedOutput = "data/coded.csv",
    )


#df["Date"] = df.apply(lambda x:pd.to_datetime(x["imp_timestamp"].split(" ")[0]),axis=1) ##Only for Testing/Training Data usage

#We expect the raw data should have the same format as the roi_data.csv that was given for the training
def RawDataClean(rawCSVInputFile, showLog=False):
    badLines = []
    with open(rawCSVInputFile,"r") as f:
        lines = f.readlines()
        modifiedLines = []
        dicts = []
        for i in tqdm(range(len(lines)), desc='Load CSV Lines'):
            line = lines[i]
            if not i:
                modifiedLines.append(line.replace("contextual_data,",""))
            else:
                findit = re.findall("\"\{.*\}\"",line)
                if len(findit):
                    dictstr = findit[0]
                    dicts.append(json.loads(dictstr))
                    newline = line.replace(dictstr,"").replace(",,",",")
                    modifiedLines.append(newline)
                else:
                    badLines.append(i)
                    continue
    with open(modelParameters.contextual_data, 'w') as f:
        json.dump(dicts, f)
    
    modelParameters.modifiedCSV = re.sub(".csv$", "_new.csv", rawCSVInputFile)
    
    with open( modelParameters.modifiedCSV, 'w') as f:
        f.writelines(modifiedLines)
    
    if showLog:
        for i in badLines:
            print(lines[i])

            
def createCrossTabDict(df = None):
    df = pd.read_csv(modelParameters.modifiedCSV) if df is None else df
    for pairs in modelParameters.crossTabPairs:
        conf_matrix = pd.crosstab(df[pairs[0]],df[pairs[1]])
        data2d = conf_matrix.to_numpy()
        encodeDict={}
        codeVal = 1
        for i in range(data2d.shape[0]):
            for j in range(data2d.shape[1]):
                if data2d[i,j]>modelParameters.dataParameterCoverage:
                    encodeDict["{}_{}".format(conf_matrix.index[i],conf_matrix.columns[j])]=codeVal
                    codeVal += 1
        modelParameters.dataDict[f"{pairs[0]}_{pairs[1]}"] = encodeDict

def createSingleNameDict(df=None):    
    df = pd.read_csv(modelParameters.modifiedCSV) if df is None else df
    for singleColumn in modelParameters.singleColumnEncoded:
        validValues = list(df[df.apply(lambda x: not pd.isna(x[singleColumn]), axis=1)][singleColumn].unique())
        encodeDict = dict([[name,i+1] for i,name in enumerate(validValues)])
        modelParameters.dataDict[singleColumn] = encodeDict


def main(fileInput=None):
    fileInput = modelParameters.rawCSV if fileInput is None else fileInput
    RawDataClean(fileInput)
    createCrossTabDict()
    createSingleNameDict()
    with open(modelParameters.encodingDictFile, 'wb') as handle:
        pickle.dump(modelParameters, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()

