import utils
import etl
import models
import cross
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from collections import defaultdict
from sklearn.cross_validation import KFold, ShuffleSplit
from numpy import mean
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

'''
input:
output: X_train,Y_train,X_test
'''
def aggregate_test_data():
        filepath='../data/test/'
        deliverables_path='../deliverables/'
        events_df = pd.read_csv(filepath + 'events.csv', parse_dates=['timestamp'])
        feature_map_df = pd.read_csv(filepath + 'event_feature_map.csv')
        
        joined=pd.merge(events_df,feature_map_df,how='left',on='event_id')
        joined=joined[joined['value'].notnull()]
        split1,split2=joined[joined['idx']<2680],joined[joined['idx']>=2680]
        aggre1=split1[['patient_id','idx','value']].groupby(['patient_id','idx']).sum()
        aggre2=split2[['patient_id','idx','value']].groupby(['patient_id','idx']).count()
        result=pd.concat([aggre1,aggre2]).sort()
        aggregated_events = result.reset_index()
        listgroup=aggregated_events.groupby('idx')
        f=open(deliverables_path + 'test_features.txt','wb')
        f.write('patient_id'+',idx'+',value'+'\n')
        for name,group in listgroup:     
                normalized= (group['value'].as_matrix())/(group['value'].as_matrix().max())
                for i in range(len(normalized)):
                        f.write(str(name)+','+str(int(group.iloc[i]['patient_id']))+','+str(normalized[i])+'\n')
        f.close()
        aggregated_events=pd.read_csv(deliverables_path + 'test_features.txt')
        patient_features = defaultdict(list)
        for i in range(len(aggregated_events.index)):
                line=aggregated_events.iloc[i]
                patient_features[line['patient_id']].append((line['idx'],line['value']))
              
        deliverable1 = open(deliverables_path + 'test_features.txt', 'w')
        copy=patient_features.keys()[:]
        copy.sort()
        for key in copy:
                deliverable1.write(str(int(key))+' '+utils.bag_to_svmlight(sorted(patient_features[key]))+'\n');
        deliverable1.close()
        
def my_features():
	#TODO: complete this
        #aggregate_test_data()
        X_train, Y_train = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
        X_test, Y_test = utils.get_data_from_svmlight("../deliverables/test_features.txt")
        clf=ExtraTreesClassifier()
        clf = clf.fit(X_train, Y_train)
        model = SelectFromModel(clf, prefit=True)
        X_new = model.transform(X_train)
        X_test=model.transform(X_test)
        return X_new, Y_train, X_test
'''

input: X_train, Y_train, X_test
output: Y_pred
'''
def get_acc_auc_kfold(X,Y,model,k=5):
        #TODO:First get the train indices and test indices for each iteration
        #Then train the classifier accordingly
        #Report the mean accuracy and mean auc of all the folds
        cv=KFold(X.shape[0],k)
        Acc,Auc=[],[]
        for train, test in cv:
                Y_pred=model(X[train],Y[train],X[test])
                acc,auc,_,_,_=models.classification_metrics(Y_pred, Y[test])
                Acc.append(acc)
                Auc.append(auc)
        meanAcc=mean(Acc)
        meanAuc=mean(Auc)
        return meanAcc,meanAuc

def randomforest_pred(X_train, Y_train, X_test):
        #TODO:train a logistic regression classifier using X_train and Y_train. Use this to predict labels of X_test
        #IMPORTANT: use max_depth as 5. Else your test cases might fail.
        DT=RandomForestClassifier(n_estimators=100)
        DT.fit(X_train,Y_train)
        Y_pred=DT.predict(X_test)
        return Y_pred

def my_classifier_predictions(X_train,Y_train,X_test):
        #TODO: complete this
        lr = LogisticRegression()
        svc = LinearSVC(C=1.0)
        rfc = RandomForestClassifier(n_estimators=100)
        Y_pred,max_precision={},{}
        for clf, name in [(lr, models.logistic_regression_pred),
                  (svc, models.svm_pred),
                  (rfc, randomforest_pred)]:
                clf.fit(X_train, Y_train)
                Y_pred[name]=clf.predict(X_test)
                max_precision[name]=get_acc_auc_kfold(X_train,Y_train,name)[1]
        maximum=0     
        for key, value in max_precision.items():
                if value>maximum:
                        maximum=value
                        model=key
        #print model SVM             
        return Y_pred[model]
                


def main():
	X_train, Y_train, X_test = my_features()
	Y_pred = my_classifier_predictions(X_train,Y_train,X_test)
	utils.generate_submission("../deliverables/test_features.txt",Y_pred)
	#The above function will generate a csv file of (patient_id,predicted label) and will be saved as "my_predictions.csv" in the deliverables folder.

if __name__ == "__main__":
    main()

	
