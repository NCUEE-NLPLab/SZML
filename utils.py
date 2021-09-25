import pandas as pd
import numpy as np
import scipy.io as sio
import glob
import os
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import warnings
from nilearn.connectome import sym_matrix_to_vec

warnings.filterwarnings('ignore')

def read_mat(path):

    def down_tri(matname):
        Lmat=sio.loadmat(matname)
        cor=Lmat['ROICorrelation_FisherZ']
        cor = sym_matrix_to_vec(cor,discard_diagonal=True)
        cor = np.expand_dims(cor,axis=0)
        return cor

    data=np.array([])
    if os.path.isdir(path):
        for names in tqdm(sorted(glob.glob(path+'/*.mat'))):
            tmp=down_tri(names)
            data=np.concatenate((data,tmp))
        return data
    else:
        return down_tri(path)

def read_mat_fromdflist(path,dflist):

    def down_tri(matname):
        Lmat=sio.loadmat(matname)
        cor=Lmat['ROICorrelation_FisherZ']
        cor = sym_matrix_to_vec(cor,discard_diagonal=True)
        cor = np.expand_dims(cor,axis=0)
        return cor

    data=np.array([])
    for names in tqdm(dflist):
        tmpnames=os.path.join(path,names)
        tmp=down_tri(tmpnames)
        data=np.concatenate((data,tmp)) 
    return data



def select_data(data1,data2,size,seed):
    index1=np.arange(len(data1))
    index2=np.arange(len(data2))
    np.random.seed(seed)
    np.random.shuffle(index1)
    np.random.shuffle(index2)
    class1=data1[index1]
    class2=data2[index2]

    label=np.concatenate((np.ones((size)),np.zeros((size))))
    selectdata=np.concatenate((class1[:size],class2[:size]))
    return selectdata,label

def sensitivity(y_true,y_pred):
    confmat=confusion_matrix(y_true,y_pred)
    TN,FP,FN,TP=confmat[0][0],confmat[0][1],confmat[1][0],confmat[1][1]
    Score_sensitivity=TP/(TP+FN)
    return Score_sensitivity

def specificity(y_true,y_pred):
    confmat=confusion_matrix(y_true,y_pred)
    TN,FP,FN,TP=confmat[0][0],confmat[0][1],confmat[1][0],confmat[1][1]
    Score_specificity=TN/(FP+TN)

    return Score_specificity

def mean_(scorelist):
    meanscore=sum(scorelist)/len(scorelist)
    return meanscore

def CVresult(clf):
    clf_result=clf.cv_results_
    CVdict=dict()
    CVdict["accuracy"]=clf_result['mean_test_accuracy'][clf.best_index_]
    CVdict["f1_score"]=clf_result['mean_test_f1_score'][clf.best_index_]
    CVdict["sensitivity"]=clf_result['mean_test_sensitivity'][clf.best_index_]
    CVdict["specificity"]=clf_result['mean_test_specificity'][clf.best_index_]

    return CVdict

def CVstdresult(clf,acc_list,f1_list,Sen_list,Spe_list):
    clf_result=clf.cv_results_
    acc_list.append(clf_result['std_test_accuracy'][clf.best_index_])
    f1_list.append(clf_result['std_test_f1_score'][clf.best_index_])
    Sen_list.append(clf_result['std_test_sensitivity'][clf.best_index_])
    Spe_list.append(clf_result['std_test_specificity'][clf.best_index_])
#     p_list.append(pvalue)
    resultdict={"acc":acc_list,"f1":f1_list,"sensitivity":Sen_list,"specificity":Spe_list}
    return acc_list,f1_list,Sen_list,Spe_list,resultdict

def skf_fuction(data,label,n_splits=10,Seed=0):

    skf=StratifiedKFold(n_splits=n_splits,random_state=Seed, shuffle=True)
    traindata_skf=[i for i in range(n_splits)]
    trainlabel_skf=[i for i in range(n_splits)]
    testdata_skf=[i for i in range(n_splits)]
    testlabel_skf=[i for i in range(n_splits)]

    for num, indexarray in enumerate(tqdm(skf.split(data,label))):
        traindata_skf[num]=data[indexarray[0]]
        trainlabel_skf[num]=label[indexarray[0]]
        testdata_skf[num]=data[indexarray[1]]
        testlabel_skf[num]=label[indexarray[1]]

    return traindata_skf, trainlabel_skf,testdata_skf,testlabel_skf


def clfresult(y_true, y_pred, y_pred_prob):
    """
    input:
    *scorename= ["accuracy","f1_score","sensitivity",specificity]
    y_true
    y_pred
    y_pred_prob

    output:
    return scoredict
    """
    scoredict=dict()
    scoredict["accuracy"]=balanced_accuracy_score(y_true,y_pred)
    scoredict["f1_score"]=f1_score(y_true,y_pred)
    scoredict["sensitivity"]=sensitivity(y_true,y_pred)
    scoredict["specificity"]=specificity(y_true,y_pred)
    scoredict["auc_score"]=roc_auc_score(y_true,y_pred_prob)

    return scoredict
