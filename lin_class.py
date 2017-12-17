from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from numpy import genfromtxt
#from matplotlib import pyplot
from sklearn import metrics
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sklearn
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import pickle
i=0
macro_av=0
micro_av=0
c_=[50, 100, 500]
macro_values=np.zeros((9,len(c_)))
micro_values=np.zeros((9,len(c_)))
i_idx=0

Dataset_X = np.genfromtxt('embedz', dtype=np.float32, delimiter=' ') #loading datasets
Dataset_X =sklearn.preprocessing.normalize(Dataset_X, norm='l2', axis=1, copy=True)
Dataset_Y=  np.reshape(np.argmax(np.load('labels.npy'), axis=1), (2708, 1)) 


for percentage in range(90,0,-10):
    j_idx=0
    for cc in c_:
        macro_av=0
        micro_av=0
        for i in range(10):
            Dataset_X = np.genfromtxt('embedz', dtype=np.float32, delimiter=' ') #loading datasets
            Dataset_X =sklearn.preprocessing.normalize(Dataset_X, norm='l2', axis=1, copy=True)
            Dataset_Y=  np.reshape(np.argmax(np.load('labels.npy'), axis=1), (2708, 1)) #loading datasets
            #pca=PCA(n_components=2, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None)
            lda=LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=None, n_components=2, store_covariance=False, tol=0.0001)
            pca=lda.fit(Dataset_X,Dataset_Y)
            Dataset_X_1=pca.transform(Dataset_X)
            #print(Dataset_X.shape)
            ##pyplot.figure()
            ##pyplot.scatter(Dataset_X_1[:,0],Dataset_X_1[:,1],c=Dataset_Y)
            #pyplot.show()
            Dataset=np.hstack((Dataset_X,Dataset_Y))
            np.random.shuffle(Dataset)
            Dataset_X=Dataset[:,:-1]
            Dataset_Y=Dataset[:,-1]
            train_data_X=Dataset_X[:int((percentage/100.0)*Dataset_X.shape[0]),:]
            test_data_X=Dataset_X[int((percentage/100.0)*Dataset_X.shape[0]):,:]
            test_data_Y=Dataset_Y[int((percentage/100.0)*Dataset_Y.shape[0]):]
            train_data_Y=Dataset_Y[:int((percentage/100.0)*Dataset_Y.shape[0])]

            #print("abc")
            model=sklearn.linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=cc, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
            model.fit(train_data_X,train_data_Y)
            #print("bcd")
            pred_y=model.predict(test_data_X)
            #print(metrics.classification_report(test_data_Y, pred_y))
            macro_av+=metrics.precision_recall_fscore_support(test_data_Y, pred_y,average='macro')[2]
            #print(macro_av)
            micro_av+=metrics.precision_recall_fscore_support(test_data_Y, pred_y,average='micro')[2]
            #print(micro_av)
    #python example_graphs/scoring.py --emb ../lap/embedz_linear_2000  --network lel.mat --adj-matrix-name adjmat --label-matrix-name labels  --num-shuffle 10 --all
        print("C=,"+str(cc)+"  percentage labeled data="+str(percentage))
        print("MACRO_AV"+str(macro_av/10))
        print("MICRO_AV"+str(micro_av/10))
        macro_values[i_idx,j_idx]=macro_av/10
        micro_values[i_idx,j_idx]=micro_av/10
        j_idx+=1
    i_idx+=1
print("==========================REPORT==============================")
np.set_printoptions(threshold=np.inf)
print("macro_grid:")
print(macro_values)
print("micro_grid:")
print(micro_values)        
