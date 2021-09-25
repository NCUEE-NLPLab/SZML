from model import cv_svm
import numpy as np
import os
import argparse
import pandas as pd
from mklog import get_logger
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--hcdata_path", type=str, default="data/fake_hcdata.npy", help="hc data path")
    parser.add_argument("--szdata_path", type=str, default="data/fake_szdata.npy", help="sz data path")
    parser.add_argument("--result_path", type=str, default="result", help="output folder")
    parser.add_argument("--log", type=str, default="train_log", help="logger filename")
    parser.add_argument("--model",type=str, default="SVC", help="select scikit-learn model")
    parser.add_argument("--times",type=int, default=2, help="repeat the classification with different random seed")
    parser.add_argument("--C", type=list, default=[1,10,100,1000], help="SVM parameters")
    parser.add_argument("--tol", type=list, default=[0.001,0.01,0.1,1], help="SVM parameters")
    parser.add_argument("--n_jobs", type=int, default=2, help="Number of jobs to run in parallel.")
    args=parser.parse_args()


    os.makedirs(args.result_path,exist_ok=True)

    logger=get_logger(outputdir=args.result_path,logname=args.log)

    result=cv_svm(logger=logger, args=args)

    logger.info(f"finish")