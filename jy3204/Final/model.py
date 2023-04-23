#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 11:38:16 2023

@author: jackyang
"""


from argparse import Namespace
import pickle
import pandas as pd
import re
from sklearn.utils import resample
#from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_poisson_deviance
from sklearn.linear_model import Ridge
from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder
#from sklearn.model_selection import cross_validate
from sklearn.utils import shuffle
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_selector
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
#from sklearn.model_selection import cross_val_score
#from sklearn.metrics import mean_poisson_deviance
import numpy as np
import matplotlib.pyplot as plt

modelParameters = Namespace(
    downSampleFactor = 3,
    encodingDictFile = "data/encoding.pkl",
    baggingSize = 100,
    low = 0.3,
    high = 0.7,
    # low = 0.5,
    # high = 0.7,
)


def binF1(y_true, y_pred,**kargs):
    bins = [-100,0.05,2,4,100]
    labels = [-100,0,2,4,6]
    # bins = [-100,0.05,4,100]
    # labels = [-100,0,4,6]
    # bins = [-100,0.02,3,5,100]
    # labels = [-100,0.02,3,5,6]

    y1_true = np.copy(y_true)
    y1_pred = np.copy(y_pred)
    for i in range(1,len(bins)):
        y1_true[(y1_true<=bins[i])&(y1_true>bins[i-1])]= labels[i]
        y1_pred[(y1_pred<=bins[i])&(y1_pred>bins[i-1])]= labels[i]
    if "sample_weight" in kargs:
        return f1_score(y1_true, y1_pred, average='macro', sample_weight=kargs["sample_weight"])
    return f1_score(y1_true, y1_pred, average='macro')

modelParameters.score = make_scorer(binF1, greater_is_better=True)

def PoissonDeviance(y_true, y_pred,**kargs):
    y1_true = np.copy(y_true)
    y1_pred = np.copy(y_pred)
    y1_pred[y1_pred<=0] = 0
    y1_pred = y1_pred+1e-10
    return mean_poisson_deviance(y1_true, y1_pred)

modelParameters.PossionScore = make_scorer(PoissonDeviance, greater_is_better=False)

def getProcessParameters():
    if "ProcessParameters" not in modelParameters:
        with open(modelParameters.encodingDictFile, 'rb') as handle:
            params = pickle.load(handle)
            modelParameters.ProcessParameters = params
    return modelParameters.ProcessParameters

def modelDfPreparation():
    params = getProcessParameters()
    df = pd.read_csv(params.codedOutput)
    df_train = df[df.apply(lambda x: x["Date"] not in ['2022-12-05','2022-12-04'], axis=1)].copy(deep=True)
    df_test = df[df.apply(lambda x: x["Date"] in ['2022-12-05','2022-12-04'], axis=1)].copy(deep=True)
    dtypesDict = dict([[_,df_test[_].dtype] for _ in df_test.columns])
    
    df_train[params.categorical] = df_train[params.categorical].astype("category")
    df_test[params.categorical] = df_test[params.categorical].astype("category")
    
    X_train = df_train[params.categorical+params.numericalColumns].copy(deep=True)
    X_test = df_test[params.categorical+params.numericalColumns].copy(deep=True)
    X_train[params.categorical] = X_train[params.categorical].astype("category")
    X_test[params.categorical] = X_test[params.categorical].astype("category")
    
    zeroROI = df_train[df_train["conversion_roi"]<1e-20].to_numpy()
    nonZeroROI = df_train[df_train["conversion_roi"]>1e-20].to_numpy()

    df_trains=[]
    X_trains = []
    for i in range(modelParameters.baggingSize):
        nonzeroSet = resample(nonZeroROI, n_samples=nonZeroROI.shape[0])
        df_trainNew = pd.DataFrame(nonzeroSet, columns=df_train.columns)
        zeroSet = resample(zeroROI, n_samples=int(zeroROI.shape[0]/modelParameters.downSampleFactor))
        df_trainNew = pd.concat([df_trainNew,pd.DataFrame(zeroSet,columns=df_train.columns)])
        df_trainNew = df_trainNew.astype(dtypesDict)
        weights    = np.ones(df_trainNew.shape[0])*1.
        weights[nonZeroROI.shape[0]:]=weights[nonZeroROI.shape[0]:]*0+modelParameters.downSampleFactor
        df_trainNew["weights"] = weights
        df_trainNew = shuffle(df_trainNew)
        X_trainNew = df_trainNew[params.categorical+params.numericalColumns].copy(deep=True)
        X_trainNew[params.categorical] = X_trainNew[params.categorical].astype("category")
        df_trains.append(df_trainNew)
        X_trains.append(X_trainNew)
    
    modelParameters.X_train = X_train
    modelParameters.X_test = X_test
    modelParameters.X_trains = X_trains
    modelParameters.df_train = df_train
    modelParameters.df_test = df_test
    modelParameters.df_trains = df_trains
    

def _linear_model_processor(unknown=np.nan):
    params = getProcessParameters()
    return ColumnTransformer(
            [
                ("passthrough_numeric", "passthrough", params.numericalColumns),
                (
                    "onehot_categorical",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=unknown),
                    params.categorical,
                ),
            ],
            remainder="drop",
        )

def _ordinal_encoder():
    return make_column_transformer(
        (
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan),
            make_column_selector(dtype_include="category"),
        ),
        remainder="passthrough",
        # Use short feature names to make it easier to specify the categorical
        # variables in the HistGradientBoostingRegressor in the next step
        # of the pipeline.
        verbose_feature_names_out=False )

def getDfInputs():
    if "df_train" not in modelParameters:
        modelDfPreparation()
    return [modelParameters.df_train,modelParameters.df_test,modelParameters.df_trains]

def getXInputs():
    if "X_train" not in modelParameters:
        modelDfPreparation()
    return [modelParameters.X_train,modelParameters.X_test,modelParameters.X_trains]

def predict(estimator, df_test):
    if isinstance(estimator, list):
        y_preds = np.stack([subModel.predict(df_test) for subModel in estimator], axis=-1)
        y_predx = np.argsort(y_preds,axis=-1)
        newVals = np.take_along_axis(y_preds,y_predx,axis=1)
        y_pred = np.average(newVals[:,int(modelParameters.low*len(estimator)):int(modelParameters.high*len(estimator))],axis=-1)
    else:
        y_pred = estimator.predict(df_test)
    return y_pred

def score_estimator(estimator, df_test, modelTitle):
    """Score an estimator on the test set."""
    params = getProcessParameters()
    y_pred = predict(estimator, df_test)
    y_predUsed = y_pred
    y_predUsed[y_predUsed<0]=0

    print( "Scoring Model {}".format(modelTitle))
    print( "RMSE:{:.4f}".format( np.sqrt(mean_squared_error( df_test[params.targetColumn], y_predUsed ))))
    print( "RMAE:{:.4f}".format( np.sqrt(mean_absolute_error( df_test[params.targetColumn], y_predUsed ))))

    # Ignore non-positive predictions, as they are invalid for
    # the Poisson deviance.
    mask = y_pred > 0
    if (~mask).any():
        n_masked, n_samples = (~mask).sum(), mask.shape[0]
        print( "WARNING: Estimator yields invalid, non-positive predictions  \
            for {} samples out of {}. These predictions \
            are ignored when computing the Poisson deviance.".format(n_masked,n_samples) )

    print( "mean Poisson deviance: {:.4f}".format(mean_poisson_deviance( df_test[params.targetColumn][mask], y_pred[mask]))) 
    print("Macro F1 Score: {:.4f}".format(binF1(df_test[params.targetColumn],y_predUsed)))

def zeroModel():
    df_train = getDfInputs()[0]
    params = getProcessParameters()
    zero = Pipeline(
    [
        ("preprocessor", _linear_model_processor()),
        ("regressor", DummyRegressor(strategy="constant", constant=1e-10)), #strict 0 is not allowed in deviance calculation
    ]).fit(df_train, df_train[params.targetColumn])
    return zero

def baselineModel():
    df_train = getDfInputs()[0]
    params = getProcessParameters()
    dummy = Pipeline(
    [
        ("preprocessor", _linear_model_processor()),
        ("regressor", DummyRegressor(strategy="mean")),
    ]).fit(df_train, df_train[params.targetColumn])
    return dummy

def linearRidgeModel():
    df_train = getDfInputs()[0]
    params = getProcessParameters()
    ridge_glm = Pipeline(
    [
        ("preprocessor", _linear_model_processor(-100)),
        ("regressor", Ridge(alpha=1e-4)),
    ]
    ).fit(df_train, df_train[params.targetColumn])
    return ridge_glm

def PoissonModel():
    df_train = getDfInputs()[0]
    params = getProcessParameters()
    poisson_glm = Pipeline(
        [
            ("preprocessor", _linear_model_processor(-100)),
            ("regressor", PoissonRegressor(alpha=1e-4, solver="newton-cholesky")),
        ]
    )
    poisson_glm.fit(df_train, df_train[params.targetColumn])  
    return poisson_glm

def plainGBRT():
    df_train = getDfInputs()[0]
    X_train = getXInputs()[0]
    params = getProcessParameters()
    hist_gbrt = make_pipeline(
        _ordinal_encoder(), HistGradientBoostingRegressor(
            random_state=42,
            max_leaf_nodes=128,
            scoring=modelParameters.score,
            validation_fraction=0.175))
    hist_gbrt.fit(X_train,df_train[params.targetColumn])
    return hist_gbrt

def modifiedGBRT_base():
    df_train = getDfInputs()[0]
    X_train = getXInputs()[0]
    params = getProcessParameters()
    hist_gbrt = make_pipeline(
        _ordinal_encoder(), HistGradientBoostingRegressor(
            loss="poisson",
            random_state=42,
            max_leaf_nodes=128,
            validation_fraction=0.175))
    hist_gbrt.fit(X_train,df_train[params.targetColumn])
    return hist_gbrt

def modifiedGBRT():
    df_train = getDfInputs()[0]
    X_train = getXInputs()[0]
    params = getProcessParameters()
    hist_gbrt = make_pipeline(
        _ordinal_encoder(), HistGradientBoostingRegressor(
            loss="poisson",
            random_state=42,
            max_leaf_nodes=128,
            scoring=modelParameters.score,
            validation_fraction=0.175))
    hist_gbrt.fit(X_train,df_train[params.targetColumn])
    return hist_gbrt

def downSampleGBRT():
    X_trains = getXInputs()[2]
    df_trains = getDfInputs()[2]
    params = getProcessParameters()
    tree_preprocessor = ColumnTransformer(
        [
            (
                "categorical",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan),
                params.categorical,
            ),
            ("numeric", "passthrough", params.numericalColumns),
        ],
        remainder="drop",
    )
    models =[]
    for i,(trainDf,X) in enumerate(zip(df_trains,X_trains)):
        poisson_gbrt = Pipeline(
            [
                ("preprocessor", tree_preprocessor),
                (
                    "regressor",
                    HistGradientBoostingRegressor(
                        loss="squared_error",
                        max_leaf_nodes=256,
                        scoring=modelParameters.score,
                        validation_fraction=0.2),
                ),
            ])

        poisson_gbrt.fit(X,trainDf[params.targetColumn],regressor__sample_weight=trainDf["weights"])
        models.append(poisson_gbrt)
    return models


def showError(models,modelTitles=None):
    df_train, df_test, _ = getDfInputs()
    params = getProcessParameters()
    fig, axes = plt.subplots(nrows=2, ncols=len(models)+1, figsize=(16, 6), sharey=True)
    fig.subplots_adjust(bottom=0.2)
    n_bins = 20
    bUpper = 6.
    for row_idx, label, df in zip(range(2), ["train", "test"], [df_train, df_test]):
        df[params.targetColumn].hist(bins=np.linspace(-1, bUpper, n_bins), ax=axes[row_idx, 0])
    
        axes[row_idx, 0].set_title("Data")
        axes[row_idx, 0].set_yscale("log")
        axes[row_idx, 0].set_xlabel("y (observed ROI)")
        axes[row_idx, 0].set_ylim([1e-10, 1.5e6])
        axes[row_idx, 0].set_ylabel(label + " samples")
    
        for idx, model in enumerate(models):
            y_pred = predict(model, df)
            if isinstance(model,list):
                y_pred = np.max(np.stack([subModel.predict(df) for subModel in model], axis=-1),axis=-1)
            else:
                y_pred = model.predict(df)
    
                
            pd.Series(y_pred).hist(
                bins=np.linspace(-1, bUpper, n_bins), ax=axes[row_idx, idx + 1]
            )
            
            if isinstance(model,list):
                submodel = model[0][-1]
            else:
                submodel = model[-1]
            
            if modelTitles is None:
                title = 'Light GBM' if re.search("Boost",submodel.__class__.__name__,re.IGNORECASE)  else submodel.__class__.__name__,
            else:
                title = modelTitles[idx]
                
            axes[row_idx, idx + 1].set(
                title= title,
                yscale="log",
                xlabel="y_pred (predicted expected ROI)",
            )
    plt.tight_layout()

def main():
    df_test = getDfInputs()[1]
    
    zero = zeroModel()
    baseLine = baselineModel()
    ridge_glm = linearRidgeModel()
    poisson_glm = PoissonModel()
    gbrt_plain = plainGBRT()
    gbrt_modified_base = modifiedGBRT_base()
    gbrt_modified = modifiedGBRT()
    dnSampleMdls = downSampleGBRT()
    
    score_estimator(zero, df_test, "zero")
    score_estimator(baseLine, df_test, "baseLine")
    score_estimator(ridge_glm, df_test,"ridge")
    score_estimator(poisson_glm, df_test,"Poisson")
    score_estimator(gbrt_plain, df_test,"gbrt base")
    score_estimator(gbrt_modified_base,df_test, "gbrt modified base")
    score_estimator(gbrt_modified,df_test, "gbrt modified")
    score_estimator(dnSampleMdls, df_test, "bagging gbrt base")
    score_estimator(dnSampleMdls+[gbrt_modified], df_test, "bagging gbrt and modified hybrid")
    
    showError([zero, baseLine, ridge_glm, poisson_glm, gbrt_plain,gbrt_modified,dnSampleMdls,dnSampleMdls+[gbrt_modified_base],dnSampleMdls+[gbrt_modified]],
              ["zero","baseLine","ridge","Poisson","grbt base","grbt modified","bgging gbrt","hybrid base","hybrid new"]
              )
    
if __name__ == "__main__":
    main()
