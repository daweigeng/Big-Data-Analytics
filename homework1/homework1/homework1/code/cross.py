import models
from sklearn.cross_validation import KFold, ShuffleSplit
from numpy import mean

import utils

#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_kfold(X,Y,k=5):
        #TODO:First get the train indices and test indices for each iteration
        #Then train the classifier accordingly
        #Report the mean accuracy and mean auc of all the folds
        cv=KFold(X.shape[0],k)
        Acc,Auc=[],[]
        for train, test in cv:
                Y_pred=models.logistic_regression_pred(X[train],Y[train],X[test])
                acc,auc,_,_,_=models.classification_metrics(Y_pred, Y[test])
                Acc.append(acc)
                Auc.append(auc)
        meanAcc=mean(Acc)
        meanAuc=mean(Auc)
        return meanAcc,meanAuc



#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_randomisedCV(X,Y,n_iter=5,test_size=0.2):
        #TODO: First get the train indices and test indices for each iteration
        #Then train the classifier accordingly
        #Report the mean accuracy and mean auc of all the iterations
        cv=ShuffleSplit(X.shape[0],n_iter,test_size)
        Acc,Auc=[],[]
        for train, test in cv:
                Y_pred=models.logistic_regression_pred(X[train],Y[train],X[test])
                acc,auc,_,_,_=models.classification_metrics(Y_pred, Y[test])
                Acc.append(acc)
                Auc.append(auc)
        meanAcc=mean(Acc)
        meanAuc=mean(Auc)
        return meanAcc,meanAuc



def main():
	X,Y = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
	print "Classifier: Logistic Regression__________"
	acc_k,auc_k = get_acc_auc_kfold(X,Y)
	print "Average Accuracy in KFold CV: "+str(acc_k)
	print "Average AUC in KFold CV: "+str(auc_k)
	acc_r,auc_r = get_acc_auc_randomisedCV(X,Y)
	print "Average Accuracy in Randomised CV: "+str(acc_r)
	print "Average AUC in Randomised CV: "+str(auc_r)

if __name__ == "__main__":
	main()

