#scikit-learn

from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC,LinearSVC
from sklearn.metrics import confusion_matrix,make_scorer

from utils import  sensitivity, specificity, CVresult, skf_fuction, clfresult, select_data

#import Classifier model
import os
import warnings
import pandas as pd
import numpy as np
import glob
import copy

warnings.filterwarnings('ignore')

def sk_model(model_name="SVC"):

    assert model_name in ["SVC","linear_svc"]
    if model_name=="linear_svc":
        return LinearSVC(random_state=42, class_weight="balanced",max_iter=1000)

    elif model_name=="SVC":
        return SVC(kernel='linear',class_weight="balanced", random_state=42,max_iter=1000)
    

def cv_svm(logger,args):
    normalizer=Normalizer()

    logger.info("Load data")
    hcdata=np.load(f"data/fake_hcdata.npy")
    szdata=np.load(f"data/fake_szdata.npy")

    alldata=np.concatenate((hcdata,szdata))

    label=np.concatenate((np.zeros(hcdata.shape[0]),np.ones(hcdata.shape[0])))

    #scale_samplewise
    
    alldata=normalizer.transform(alldata)
    model=sk_model(args.model)
    seedlist=[i for i in range(args.times)]
    n_splits=10

    scoring={'accuracy':'balanced_accuracy',
             'f1_score':'f1',
             'sensitivity':make_scorer(sensitivity),
             'specificity':make_scorer(specificity)}

    rankscore=np.zeros(alldata.shape[1],dtype=int)

    cm_all={"TP":0,"FP":0,"TN":0,"FN":0,"count":0}

    seed_dict={f"{mode}_{score}":[] for score in scoring.keys() for mode in ["train","test"]}
    seed_dict["test_auc_score"]=[]

    tuned_parameters = [{
            'C': [1,10,100,1000],
            'tol':[0.001,0.01,0.1,1],
            }]

    logger.info(f"tuned_parameters:{tuned_parameters}")
    logger.info(f"training seed:{seedlist}")
    logger.info(f"model:{model}")





    for Seed in seedlist:
        # train_test_split
        cv_train_df, cv_test_df= None,None

        traindata_skf, trainlabel_skf,testdata_skf,testlabel_skf=skf_fuction(alldata,label,n_splits=n_splits,Seed=Seed)

        for num_split in range(n_splits):

            logger.info(f"shuffle_seed:{Seed}")
            logger.info(f"num_split:{num_split}")
            logger.info(f"traindata_shape:{traindata_skf[0].shape}")
            logger.info(f"testdata_shape:{testdata_skf[0].shape}")

            # inner loop CV
            clf = GridSearchCV(model, tuned_parameters , cv=10 ,scoring=scoring , refit='f1_score',verbose=0,n_jobs=args.n_jobs)
            clf.fit(traindata_skf[num_split], trainlabel_skf[num_split])

            cv_train_dict=CVresult(clf)
            logger.info(f"cv_results")
            logger.info(f"{clf.best_params_}")
            logger.info(f"|accuracy:{cv_train_dict['accuracy']:.4f}|f1_score:{cv_train_dict['f1_score']:.4f}|sensitivity:{cv_train_dict['sensitivity']:.4f}|specificity:{cv_train_dict['specificity']:.4f}|")

            # outer loop CV
            val_pred=clf.predict(testdata_skf[num_split])
            val_pred_prob=clf.decision_function(testdata_skf[num_split])

            y_val=testlabel_skf[num_split]

            cv_test_dict=clfresult(y_val,val_pred,val_pred_prob)
            logger.info(f"test_results")
            logger.info(f"|accuracy:{cv_test_dict['accuracy']:.4f}|f1_score:{cv_test_dict['f1_score']:.4f}|sensitivity:{cv_test_dict['sensitivity']:.4f}|specificity:{cv_test_dict['specificity']:.4f}|auc_score:{cv_test_dict['auc_score']:.4f}")

            #20210823
            weight=clf.best_estimator_.coef_
            pos=np.argsort(abs(weight[0]))
            rankscore[pos]+=np.arange(len(pos))

            #20210823 confusion

            tn, fp, fn, tp =confusion_matrix(y_val,val_pred).ravel()
            cm_all["TN"]+=tn
            cm_all["FP"]+=fp
            cm_all["FN"]+=fn
            cm_all["TP"]+=tp
            cm_all["count"]+=1

            #score output
            if not isinstance(cv_train_df,pd.DataFrame):
                cv_train_df=pd.DataFrame(cv_train_dict,index=["split_0"])
            else:
                cv_train_df.loc[f'split_{num_split}']=cv_train_dict

            if not isinstance(cv_test_df,pd.DataFrame):
                cv_test_df=pd.DataFrame(cv_test_dict,index=["split_0"])
            else:
                cv_test_df.loc[f'split_{num_split}']=cv_test_dict


        for score in seed_dict.keys():
            if score.startswith("train"):
                seed_dict[score].append(cv_train_df[score.replace("train_","")].mean())

            elif score.startswith("test"):
                seed_dict[score].append(cv_test_df[score.replace("test_","")].mean())
        # print(seed_dict)
        cv_train_df.loc['average']=cv_train_df.mean()
        cv_test_df.loc['average']=cv_test_df.mean()
        cv_train_df.round(4).to_csv(f"{args.result_path}/{Seed}_cvresult.csv")
        cv_test_df.round(4).to_csv(f"{args.result_path}/{Seed}_testresult.csv")

    #20210823
    rankdf = pd.DataFrame(rankscore,columns=["rankscore"])
    rankdf["rank"]=rankdf.rank()
    rankdf.to_csv(f"{args.result_path}/rank.csv")

    rank200df=rankdf[rankdf["rank"]>=(alldata.shape[1]-200)]
    rank200df.to_csv(f"{args.result_path}/rank200.csv")

    cmdf=pd.DataFrame(cm_all,index=["cm"])
    cmdf.to_csv(f"{args.result_path}/cm.csv")


    result=pd.DataFrame(seed_dict,index=seedlist)
    # print(result)
    result_mean=result.mean()
    result_std=result.std()
    result.loc['average']=result_mean
    result.loc['std']=result_std
    result.round(4).to_csv(f"{args.result_path}/result.csv")
    return result