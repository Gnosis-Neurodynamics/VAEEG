from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from sklearn.metrics import  mean_absolute_error, r2_score
import scipy.stats as stat
from eegaux.montage import STANDARD_1020

pick_chs = ["T3", "T4", "T5", "T6", "O1", "O2", "FZ"]

def load_zims(info, mode, pick_channels=STANDARD_1020):
    res_X, res_Y, IDs = [], [], []
    ch_idx = [STANDARD_1020.index(ch) for ch in pick_channels]
    for idx, row in info.iterrows():
        zims = np.load(f"./SubAnalyze_BrainAge/data_zims/{mode}_zims/{row.SID + '.npy'}")
        zims = zims[ch_idx, :, :]
        zims_len = zims.shape[-2]
        Age = np.zeros(shape=(zims.shape[1])) + row.Age
        ID = [row.SID] * zims.shape[1]
        res_X.append(zims.transpose(1,0,2).reshape(zims_len, -1))
        res_Y.append(Age)
        IDs.extend(ID)
    res_X = np.concatenate(res_X, axis=0)
    res_Y = np.concatenate(res_Y, axis=0)
    
    res_Y = pd.DataFrame({"SID":IDs, "Age":res_Y})
    return res_X, res_Y
    

def cross_val_Regression(mode):
    info = pd.read_csv("./SubAnalyze_BrainAge/info/wake_nomal_segments.csv")
    info["quantile"] = pd.qcut(info.Age, q=5, labels=False)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)
    res = []
    for idx, (train_index, test_index) in enumerate(skf.split(info[["SID"]], info[["quantile"]])):
        train_info, test_info = info.loc[train_index], info.loc[test_index]
        X, Y = load_zims(train_info.reset_index(drop=True), mode=mode)
      
        model = LinearRegression()
        #train 
        model.fit(X, Y["Age"].values)
        Y_pred = model.predict(X)
        Y["Predict"] = Y_pred
        Y = Y.groupby("SID", as_index=False).agg("mean")
        mae = mean_absolute_error(Y["Age"].values, Y["Predict"].values)
        Rs = r2_score(Y["Age"].values, Y["Predict"].values)
        r, p = stat.pearsonr(Y["Age"].values, Y["Predict"].values)
        print(mode, "train", mae)
        Y["MAE"] = mae
        Y["RS"] = Rs
        Y["pearson_r"] = r
        Y["pearson_p"] = p
        Y["Batch_id"] = "%d"%idx
        Y["Batch"] = 'train'
        #test
        X_test, Y_test = load_zims(test_info.reset_index(drop=True), mode=mode)
        Y_test_pred = model.predict(X_test)
        Y_test["Predict"] = Y_test_pred
        Y_test = Y_test.groupby("SID", as_index=False).agg("mean")
        mae = mean_absolute_error(Y_test["Age"].values, Y_test["Predict"].values)
        Rs = r2_score(Y_test["Age"].values, Y_test["Predict"].values)
        r, p = stat.pearsonr(Y_test["Age"].values, Y_test["Predict"].values)
        print(mode, "test", mae)
        Y_test["MAE"] = mae
        Y_test["RS"] = Rs
        Y_test["pearson_r"] = r
        Y_test["pearson_p"] = p
        Y_test["Batch_id"] = "%d"%idx
        Y_test["Batch"] = "test"

        res.append(pd.concat([Y, Y_test], axis=0))
    res = pd.concat(res, axis=0)
    res["mode"] = mode
    return res

if __name__ == "__main__":
    res = []
    for m in ["vae", "transformer", "pca", "fastica"]:
        res_m = cross_val_Regression(m)
        res_m.to_csv("./SubAnalyze_BrainAge/results/LassoModelReg/%s_train_test_LR-spply.csv"%m, index=False)
        res.append(res_m)
    res = pd.concat(res, axis=0)
    res.to_csv("./SubAnalyze_BrainAge/results/LassoModelReg/RecModels_train_test_LR-spply.csv", index=False)








