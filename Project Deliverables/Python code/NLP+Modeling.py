
# coding: utf-8

# # Improved Mortality of Patients Prediction using Clinical Notes

# In[3]:

import csv
import string
import re
import sys
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import NMF
import lda
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt


# In[4]:

csv.field_size_limit(2147483647)
stop=stopwords.words('english')
punc=string.punctuation+string.digits
regex = re.compile('[%s]' % re.escape(punc))


# ## Utility Function

# In[139]:

def getKey(item):
    return item[1]

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

def plot_roc(test_y,predict):
    fpr, tpr, _ = metrics.roc_curve(test_y, predict)
    roc_auc = metrics.auc(fpr, tpr)

    # Plot of a ROC curve for a specific class
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


# ## Parse Clinical Notes and Shuffle to selected words

# In[6]:

with open('E:/A+GTCLASS/CSE8803/paper/code/data/note_final.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    i=0
    doc_all=[]
    row_num=[]
    for row in spamreader:
        document=[]
        doc=row[1].lower().replace('\n','').split('&&&&&&&&&&')
        if i!=0:
            str1=''
            for j in doc:
                j=regex.sub('',j).replace('  ','')                
                j=' '.join(h for h in j.split(' ') if h not in stop)
                document.append(j)
                if str1!='':
                    str1=str1+' '+j
                else:
                    str1=j
            if len(str1.split(' '))>100:
                vectorizer = CountVectorizer()
                x=vectorizer.fit_transform(document).toarray()
                word=np.sum(x,axis=0)
                feature=vectorizer.get_feature_names()
                rank=sorted(zip(feature,word),key=getKey,reverse=True)
                if len(rank)>=100:
                    selected=[j[0] for j in rank[:100]]
                else:
                    selected=[j[0] for j in rank]
                s=' '.join(h for h in str1.split(' ') if h in selected)
                doc_all.append(s)
                row_num.append(int(row[0]))
        i=i+1


# ## TF-IDF

# In[7]:

count_vect = TfidfVectorizer(min_df=1)
X_train_counts = count_vect.fit_transform(doc_all)
feature_all=count_vect.get_feature_names()


# In[97]:

from scipy.io import mmwrite
mmwrite('E:/A+GTCLASS/CSE8803/paper/code/data/csr_matrix.mtx', X_train_counts)


# ## Join dataset together

# In[8]:

df_tfidf=pd.DataFrame(zip(row_num,range(X_train_counts.shape[0])),columns=["subject_id","vocab"])
df_tfidf['subject_id']=df_tfidf['subject_id'].astype(int)
df_tfidf.head()


# In[102]:

df_fea=pd.DataFrame(zip(range(len(feature_all)),feature_all),columns=["word_id","word"])
df_fea.to_csv('E:/A+GTCLASS/CSE8803/paper/code/data/lookupword.csv',index=False)


# In[99]:

df_tfidf.to_csv('E:/A+GTCLASS/CSE8803/paper/code/data/lookupdoc.csv',index=False)


# In[47]:

df_heart=pd.read_csv('E:/A+GTCLASS/CSE8803/paper/code/data/patients_tab.csv')#patient table
df_heart['subject_id']=df_heart['subject_id'].astype(int)
df_heart.set_value(pd.isnull(df_heart["dod_hosp"]), 'dead_in_hospital', 0)
df_heart.set_value(pd.notnull(df_heart["dod_hosp"]), 'dead_in_hospital', 1)
df_heart.head()


# In[48]:

df_merge=df_tfidf.merge(df_heart,on=['subject_id'],how='inner')
df_merge.head()


# In[11]:

from scipy.sparse import csr_matrix
from sklearn import linear_model
from sklearn.cross_validation import KFold


# ## dead after discharging from hospital using only text data, Naive Bayes

# In[20]:

auc,sco=[],[]
kf = KFold(df_merge.shape[0], n_folds=10,shuffle=True)
for train, test in kf:
    training_set,testing_set=df_merge.iloc[train],df_merge.iloc[test]
    training_x,training_y=X_train_counts[list(training_set['vocab']),],training_set['expire_flag'].as_matrix()
    testing_x,testing_y=X_train_counts[list(testing_set['vocab']),],testing_set['expire_flag'].as_matrix()
    clf = MultinomialNB()
    clf.fit(training_x,training_y)
    pre = clf.predict(testing_x)
    score=clf.score(testing_x,testing_y)
    sco.append(score)
    auc_sco=metrics.roc_auc_score(testing_y,pre)
    auc.append(auc_sco)
print np.mean(auc),np.mean(sco)


# ## dead in hospital using only text data, Naive Bayes

# In[49]:

auc,sco=[],[]
kf = KFold(df_merge.shape[0], n_folds=10,shuffle=True)
for train, test in kf:
    training_set,testing_set=df_merge.iloc[train],df_merge.iloc[test]
    training_x,training_y=X_train_counts[list(training_set['vocab']),],training_set['dead_in_hospital'].as_matrix()
    testing_x,testing_y=X_train_counts[list(testing_set['vocab']),],testing_set['dead_in_hospital'].as_matrix()
    clf = MultinomialNB()
    clf.fit(training_x,training_y)
    pre = clf.predict(testing_x)
    score=clf.score(testing_x,testing_y)
    sco.append(score)
    auc_sco=metrics.roc_auc_score(testing_y,pre)
    auc.append(auc_sco)
print np.mean(auc),np.mean(sco)


# In[21]:

print str1


# In[25]:

get_ipython().magic(u'matplotlib inline')
plot_roc(testing_y,pre)


# ## Dead eventually Using text data only, Random Forest

# In[95]:

from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import GradientBoostingClassifier
auc,sco=[],[]
kf = KFold(df_merge.shape[0], n_folds=10,shuffle=True)
for train, test in kf:
    training_set,testing_set=df_merge.iloc[train],df_merge.iloc[test]
    training_x,training_y=X_train_counts[list(training_set['vocab']),],training_set['expire_flag'].as_matrix()
    testing_x,testing_y=X_train_counts[list(testing_set['vocab']),],testing_set['expire_flag'].as_matrix()
    clf=RandomForestClassifier(max_depth=30, n_estimators=10)
    clf.fit(training_x,training_y)
    pre = clf.predict(testing_x)
    score=clf.score(testing_x,testing_y)
    sco.append(score)
    auc_sco=metrics.roc_auc_score(testing_y,pre)
    auc.append(auc_sco)
print np.mean(auc),np.mean(sco)


# In[96]:

plot_roc(testing_y,pre)


# ## dead in hopital with only text data, using Random Forest and Gradient Boosting

# In[50]:

auc,auc1=[],[]
kf = KFold(df_merge.shape[0], n_folds=10,shuffle=True)
for train, test in kf:
    training_set,testing_set=df_merge.iloc[train],df_merge.iloc[test]
    training_x,training_y=X_train_counts[list(training_set['vocab']),],training_set['dead_in_hospital'].as_matrix()
    testing_x,testing_y=X_train_counts[list(testing_set['vocab']),],testing_set['dead_in_hospital'].as_matrix()
    clf=RandomForestClassifier(max_depth=30, n_estimators=10)
    clf.fit(training_x,training_y)
    pre = clf.predict(testing_x)
    auc_sco=metrics.roc_auc_score(testing_y,pre)
    auc.append(auc_sco)
    clf1=GradientBoostingClassifier(n_estimators=10)
    clf1.fit(training_x,training_y)
    pre1 = clf1.predict(testing_x.toarray())
    auc_sco1=metrics.roc_auc_score(testing_y,pre1)
    auc1.append(auc_sco1)    
print np.mean(auc),np.mean(auc1)


# ## Classify using gradient boosting classifier with only text data

# In[99]:

from sklearn.ensemble import GradientBoostingClassifier
auc=[]
kf = KFold(df_merge.shape[0], n_folds=10,shuffle=True)
for train, test in kf:
    training_set,testing_set=df_merge.iloc[train],df_merge.iloc[test]
    training_x,training_y=X_train_counts[list(training_set['vocab']),],training_set['expire_flag'].as_matrix()
    testing_x,testing_y=X_train_counts[list(testing_set['vocab']),],testing_set['expire_flag'].as_matrix()
    clf=GradientBoostingClassifier(n_estimators=10)
    clf.fit(training_x,training_y)
    pre = clf.predict(testing_x.toarray())
    auc_sco=metrics.roc_auc_score(testing_y,pre)
    auc.append(auc_sco)
print np.mean(auc)


# In[100]:

plot_roc(testing_y,pre)


# ## Topic Modeling using NMF and LDA

# In[132]:

nmf_model = NMF(n_components=50, random_state=1, alpha=.1, l1_ratio=.5)
nmf_model.fit(X_train_counts)
print_top_words(nmf_model, feature_all, 10)


# In[35]:

nmf_model.components_.shape


# In[133]:

nmf_topic_vec=X_train_counts.dot(nmf_model.components_.T)
nmf_topic_vec.shape


# In[12]:

count_vect_real = CountVectorizer()
X_counts = count_vect_real.fit_transform(doc_all)
feature=count_vect_real.get_feature_names()


# In[128]:

X_counts[37747,43770]


# In[104]:

mmwrite('E:/A+GTCLASS/CSE8803/paper/code/data/csr_matrix_count.mtx', X_counts)


# In[105]:

df_fea1=pd.DataFrame(zip(range(len(feature)),feature),columns=["word_id","word"])
df_fea1.to_csv('E:/A+GTCLASS/CSE8803/paper/code/data/lookupwordcount.csv',index=False)


# In[13]:

lda = lda.LDA(n_topics=50, n_iter=150, random_state=1)
lda.fit(X_counts)
doc_topic=lda.doc_topic_
print_top_words(lda, feature, 10)


# In[156]:

doc_topic.shape


# In[14]:

df_topic=pd.DataFrame(zip(row_num,range(doc_topic.shape[0])),columns=["subject_id","topic"])
df_topic['subject_id']=df_topic['subject_id'].astype(int)
df_topic.head()


# ## Join data with topic features

# In[15]:

table_sapii=pd.read_csv('E:/A+GTCLASS/CSE8803/paper/code/data/saspii_edit.csv')#sapii table
table_sapii['subject_id']=table_sapii['subject_id'].astype(int)
df_topic_merge=df_topic.merge(table_sapii,on=['subject_id'],how='inner')
df_topic_merge.head()


# In[51]:

df_all=df_topic_merge.merge(df_heart,on=['subject_id'],how='inner')
df_all.head()


# In[112]:

df_count_admin=pd.read_csv('E:/A+GTCLASS/CSE8803/paper/code/data/count_admin.csv')#new var
df_count_admin['subject_id']=df_count_admin['subject_id'].astype(int)
df_count_admin.head()


# In[113]:

df_count_admin_stay=pd.read_csv('E:/A+GTCLASS/CSE8803/paper/code/data/count_admin_stay.csv')#new var 2
df_count_admin_stay['subject_id']=df_count_admin_stay['subject_id'].astype(int)
df_count_admin_stay.head()


# In[114]:

df_all_new1=df_all.merge(df_count_admin,on=['subject_id'],how='inner')
df_all_new2=df_all_new1.merge(df_count_admin_stay,on=['subject_id'],how='inner')
df_all_new2.head()


# In[135]:

df_all_var=df_all_new2[['subject_id','sapsii','gender','age_score','sapsii_prob','hr_score','sysbp_score','temp_score','uo_score','bun_score'                   ,'wbc_score','potassium_score','sodium_score','bicarbonate_score','gcs_score','comorbidity_score'                   ,'admissiontype_score','dead_in_hospital','expire_flag','count','avg','topic']]
df_all_var=df_all_var[pd.notnull(df_all_var).all(axis=1)]
df_all_var.head()


# In[131]:

df_all_var.to_csv('E:/A+GTCLASS/CSE8803/paper/code/data/scala/vartab1.csv',index=False)


# ## Feature selection using Logistic Regression

# In[150]:

from sklearn import linear_model
auc=[]
logit_cv = linear_model.LogisticRegressionCV(penalty='l1',solver='liblinear')
k_fold = cross_validation.KFold(df_all_var.shape[0], n_folds=10,shuffle=True)
for k, (train, test) in enumerate(k_fold):
    training_set,testing_set=df_all_var.iloc[train],df_all_var.iloc[test]
    training_x,training_y=training_set[['sapsii','gender','age_score','sapsii_prob','hr_score','sysbp_score','temp_score','uo_score','bun_score'                                        ,'wbc_score','potassium_score','sodium_score','bicarbonate_score','gcs_score','comorbidity_score'                                        ,'admissiontype_score','count','avg']].as_matrix(),training_set['dead_in_hospital'].as_matrix()
    testing_x,testing_y=testing_set[['sapsii','gender','age_score','sapsii_prob','hr_score','sysbp_score','temp_score','uo_score','bun_score'                                     ,'wbc_score','potassium_score','sodium_score','bicarbonate_score','gcs_score','comorbidity_score'                                     ,'admissiontype_score','count','avg']].as_matrix(),testing_set['dead_in_hospital'].as_matrix()
    logit_cv.fit(training_x,training_y)
    pre=logit_cv.predict(testing_x)
    auc_sco=metrics.roc_auc_score(testing_y,pre)
    auc.append(auc_sco)
print np.mean(auc)


# In[151]:

logit_cv.coef_


# In[86]:

plot_roc(testing_y,pre)


# In[152]:

auc=[]
logit_cv = linear_model.LogisticRegressionCV(penalty='l1',solver='liblinear')
k_fold = cross_validation.KFold(df_all_var.shape[0], n_folds=10,shuffle=True)
for k, (train, test) in enumerate(k_fold):
    training_set,testing_set=df_all_var.iloc[train],df_all_var.iloc[test]
    training_x,training_y=training_set[['sapsii','gender','age_score','sapsii_prob','hr_score','sysbp_score','temp_score','uo_score','bun_score'                                        ,'wbc_score','potassium_score','sodium_score','bicarbonate_score','gcs_score','comorbidity_score'                                        ,'admissiontype_score','count','avg']].as_matrix(),training_set['expire_flag'].as_matrix()
    testing_x,testing_y=testing_set[['sapsii','gender','age_score','sapsii_prob','hr_score','sysbp_score','temp_score','uo_score','bun_score'                                     ,'wbc_score','potassium_score','sodium_score','bicarbonate_score','gcs_score','comorbidity_score'                                     ,'admissiontype_score','count','avg']].as_matrix(),testing_set['expire_flag'].as_matrix()
    logit_cv.fit(training_x,training_y)
    pre=logit_cv.predict(testing_x)
    auc_sco=metrics.roc_auc_score(testing_y,pre)
    auc.append(auc_sco)
print np.mean(auc)


# In[153]:

logit_cv.coef_


# In[89]:

plot_roc(testing_y,pre)


# In[54]:

df_all_real=df_all[['subject_id','topic','sapsii','gender','age_score','expire_flag','dead_in_hospital']]
df_all_real.head()


# ## NMF topic modeling used

# In[148]:

from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import GradientBoostingClassifier

auc,auc1,auc2=[],[],[]
kf = KFold(df_all_var.shape[0], n_folds=10,shuffle=True)
for train, test in kf:
    training_set,testing_set=df_all_var.iloc[train],df_all_var.iloc[test]
    training_x,training_y=np.hstack((np.array(nmf_topic_vec[list(training_set['topic']),]),training_set[['sapsii','gender','age_score','sapsii_prob','hr_score','sysbp_score','temp_score','uo_score','bun_score'                                        ,'wbc_score','potassium_score','sodium_score','bicarbonate_score','gcs_score','comorbidity_score'                                        ,'admissiontype_score']].as_matrix())),training_set['expire_flag'].as_matrix()
    testing_x,testing_y=np.hstack((np.array(nmf_topic_vec[list(testing_set['topic']),]),testing_set[['sapsii','gender','age_score','sapsii_prob','hr_score','sysbp_score','temp_score','uo_score','bun_score'                                        ,'wbc_score','potassium_score','sodium_score','bicarbonate_score','gcs_score','comorbidity_score'                                        ,'admissiontype_score']].as_matrix())),testing_set['expire_flag'].as_matrix()
    clf = svm.SVC(C=1,gamma=0.1)
    clf.fit(training_x,training_y)
    pre = clf.predict(testing_x)
    clf1=GradientBoostingClassifier(n_estimators=10)
    clf1.fit(training_x,training_y)
    pre1 = clf1.predict(testing_x)
    clf2=RandomForestClassifier(max_depth=30, n_estimators=10)
    clf2.fit(training_x,training_y)
    pre2 = clf2.predict(testing_x)
    auc_sco=metrics.roc_auc_score(testing_y,pre)
    auc_sco1=metrics.roc_auc_score(testing_y,pre1)
    auc_sco2=metrics.roc_auc_score(testing_y,pre2)
    auc.append(auc_sco)
    auc1.append(auc_sco1)
    auc2.append(auc_sco2)
print np.mean(auc),np.mean(auc1),np.mean(auc2)


# In[149]:

plt.subplot(311)
fpr, tpr, _ = metrics.roc_curve(testing_y,pre)
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.subplot(312)
fpr, tpr, _ = metrics.roc_curve(testing_y,pre1)
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc="lower right")
plt.subplot(313)
fpr, tpr, _ = metrics.roc_curve(testing_y,pre2)
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc="lower right")
plt.show()


# In[146]:

auc,auc1,auc2=[],[],[]
for train, test in kf:
    training_set,testing_set=df_all_var.iloc[train],df_all_var.iloc[test]
    training_x,training_y=np.hstack((np.array(nmf_topic_vec[list(training_set['topic']),]),training_set[['sapsii','gender','age_score','sapsii_prob','hr_score','sysbp_score','temp_score','uo_score','bun_score'                                        ,'wbc_score','potassium_score','sodium_score','bicarbonate_score','gcs_score','comorbidity_score'                                        ,'admissiontype_score']].as_matrix())),training_set['dead_in_hospital'].as_matrix()
    testing_x,testing_y=np.hstack((np.array(nmf_topic_vec[list(testing_set['topic']),]),testing_set[['sapsii','gender','age_score','sapsii_prob','hr_score','sysbp_score','temp_score','uo_score','bun_score'                                        ,'wbc_score','potassium_score','sodium_score','bicarbonate_score','gcs_score','comorbidity_score'                                        ,'admissiontype_score']].as_matrix())),testing_set['dead_in_hospital'].as_matrix()
    clf = svm.SVC(C=1,gamma=0.1)
    clf.fit(training_x,training_y)
    pre = clf.predict(testing_x)
    clf1=GradientBoostingClassifier(n_estimators=10)
    clf1.fit(training_x,training_y)
    pre1 = clf1.predict(testing_x)
    clf2=RandomForestClassifier(max_depth=30, n_estimators=10)
    clf2.fit(training_x,training_y)
    pre2 = clf2.predict(testing_x)
    auc_sco=metrics.roc_auc_score(testing_y,pre)
    auc_sco1=metrics.roc_auc_score(testing_y,pre1)
    auc_sco2=metrics.roc_auc_score(testing_y,pre2)
    auc.append(auc_sco)
    auc1.append(auc_sco1)
    auc2.append(auc_sco2)
print np.mean(auc),np.mean(auc1),np.mean(auc2)


# In[147]:

plt.subplot(311)
fpr, tpr, _ = metrics.roc_curve(testing_y,pre)
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.subplot(312)
fpr, tpr, _ = metrics.roc_curve(testing_y,pre1)
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc="lower right")
plt.subplot(313)
fpr, tpr, _ = metrics.roc_curve(testing_y,pre2)
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc="lower right")
plt.show()


# In[137]:

auc,auc1,auc2=[],[],[]
kf = KFold(df_all_var.shape[0], n_folds=10,shuffle=True)
for train, test in kf:
    training_set,testing_set=df_all_var.iloc[train],df_all_var.iloc[test]
    training_x,training_y=training_set[['sapsii','gender','age_score','sapsii_prob','hr_score','sysbp_score','temp_score','uo_score','bun_score'                                        ,'wbc_score','potassium_score','sodium_score','bicarbonate_score','gcs_score','comorbidity_score'                                        ,'admissiontype_score','count','avg']].as_matrix(),training_set['dead_in_hospital'].as_matrix()
    testing_x,testing_y=testing_set[['sapsii','gender','age_score','sapsii_prob','hr_score','sysbp_score','temp_score','uo_score','bun_score'                                        ,'wbc_score','potassium_score','sodium_score','bicarbonate_score','gcs_score','comorbidity_score'                                        ,'admissiontype_score','count','avg']].as_matrix(),testing_set['dead_in_hospital'].as_matrix()
    clf = svm.SVC(C=1,gamma=0.1)
    clf.fit(training_x,training_y)
    pre = clf.predict(testing_x)
    clf1=GradientBoostingClassifier(n_estimators=10)
    clf1.fit(training_x,training_y)
    pre1 = clf1.predict(testing_x)
    clf2=RandomForestClassifier(max_depth=30, n_estimators=10)
    clf2.fit(training_x,training_y)
    pre2 = clf2.predict(testing_x)
    auc_sco=metrics.roc_auc_score(testing_y,pre)
    auc_sco1=metrics.roc_auc_score(testing_y,pre1)
    auc_sco2=metrics.roc_auc_score(testing_y,pre2)
    auc.append(auc_sco)
    auc1.append(auc_sco1)
    auc2.append(auc_sco2)
print np.mean(auc),np.mean(auc1),np.mean(auc2)


# In[145]:

import matplotlib.pyplot as plt
plt.subplot(311)
fpr, tpr, _ = metrics.roc_curve(testing_y,pre)
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.subplot(312)
fpr, tpr, _ = metrics.roc_curve(testing_y,pre1)
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc="lower right")
plt.subplot(313)
fpr, tpr, _ = metrics.roc_curve(testing_y,pre2)
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc="lower right")
plt.show()


# ## LDA topic features used

# In[21]:

auc,sco,auc1,auc2=[],[],[],[]
kf = KFold(df_all_real.shape[0], n_folds=10,shuffle=True)
for train, test in kf:
    training_set,testing_set=df_all_real.iloc[train],df_all_real.iloc[test]
    training_x,training_y=np.hstack((np.array(doc_topic[list(training_set['topic']),]),training_set[['sapsii','gender','age_score']]                                     .as_matrix())),training_set['expire_flag'].as_matrix()
    testing_x,testing_y=np.hstack((np.array(doc_topic[list(testing_set['topic']),]),testing_set[['sapsii','gender','age_score']]                                   .as_matrix())),testing_set['expire_flag'].as_matrix()
    clf = svm.SVC(C=1,gamma=0.1)
    clf.fit(training_x,training_y)
    pre = clf.predict(testing_x)
    clf1=GradientBoostingClassifier(n_estimators=10)
    clf1.fit(training_x,training_y)
    pre1 = clf1.predict(testing_x)
    clf2=RandomForestClassifier(max_depth=30, n_estimators=10)
    clf2.fit(training_x,training_y)
    pre2 = clf2.predict(testing_x)
    score=clf.score(testing_x,testing_y)
    sco.append(score)
    auc_sco=metrics.roc_auc_score(testing_y,pre)
    auc_sco1=metrics.roc_auc_score(testing_y,pre1)
    auc_sco2=metrics.roc_auc_score(testing_y,pre2)
    auc.append(auc_sco)
    auc1.append(auc_sco1)
    auc2.append(auc_sco2)
print np.mean(auc),np.mean(sco),np.mean(auc1),np.mean(auc2)


# In[91]:

plot_roc(testing_y,pre)


# In[23]:

get_ipython().magic(u'matplotlib inline')
plot_roc(testing_y,pre1)


# In[24]:

plot_roc(testing_y,pre2)


# In[92]:

auc,auc1,auc2=[],[],[]
kf = KFold(df_all_var.shape[0], n_folds=10,shuffle=True)
for train, test in kf:
    training_set,testing_set=df_all_var.iloc[train],df_all_var.iloc[test]
    training_x,training_y=np.hstack((np.array(doc_topic[list(training_set['topic']),]),training_set[['sapsii','gender','age_score','sapsii_prob','hr_score','sysbp_score','temp_score','uo_score','bun_score'                                        ,'wbc_score','potassium_score','sodium_score','bicarbonate_score','gcs_score','comorbidity_score'                                        ,'admissiontype_score']].as_matrix())),training_set['expire_flag'].as_matrix()
    testing_x,testing_y=np.hstack((np.array(doc_topic[list(testing_set['topic']),]),testing_set[['sapsii','gender','age_score','sapsii_prob','hr_score','sysbp_score','temp_score','uo_score','bun_score'                                        ,'wbc_score','potassium_score','sodium_score','bicarbonate_score','gcs_score','comorbidity_score'                                        ,'admissiontype_score']].as_matrix())),testing_set['expire_flag'].as_matrix()
    clf = svm.SVC(C=1,gamma=0.1)
    clf.fit(training_x,training_y)
    pre = clf.predict(testing_x)
    clf1=GradientBoostingClassifier(n_estimators=10)
    clf1.fit(training_x,training_y)
    pre1 = clf1.predict(testing_x)
    clf2=RandomForestClassifier(max_depth=30, n_estimators=10)
    clf2.fit(training_x,training_y)
    pre2 = clf2.predict(testing_x)
    auc_sco=metrics.roc_auc_score(testing_y,pre)
    auc_sco1=metrics.roc_auc_score(testing_y,pre1)
    auc_sco2=metrics.roc_auc_score(testing_y,pre2)
    auc.append(auc_sco)
    auc1.append(auc_sco1)
    auc2.append(auc_sco2)
print np.mean(auc),np.mean(auc1),np.mean(auc2)


# In[55]:

auc,auc1,auc2=[],[],[]
kf = KFold(df_all_real.shape[0], n_folds=10,shuffle=True)
for train, test in kf:
    training_set,testing_set=df_all_real.iloc[train],df_all_real.iloc[test]
    training_x,training_y=np.hstack((np.array(doc_topic[list(training_set['topic']),]),training_set[['sapsii','gender','age_score']]                                     .as_matrix())),training_set['dead_in_hospital'].as_matrix()
    testing_x,testing_y=np.hstack((np.array(doc_topic[list(testing_set['topic']),]),testing_set[['sapsii','gender','age_score']]                                   .as_matrix())),testing_set['dead_in_hospital'].as_matrix()
    clf = svm.SVC(C=1,gamma=0.1)
    clf.fit(training_x,training_y)
    pre = clf.predict(testing_x)
    clf1=GradientBoostingClassifier(n_estimators=10)
    clf1.fit(training_x,training_y)
    pre1 = clf1.predict(testing_x)
    clf2=RandomForestClassifier(max_depth=30, n_estimators=10)
    clf2.fit(training_x,training_y)
    pre2 = clf2.predict(testing_x)
    auc_sco=metrics.roc_auc_score(testing_y,pre)
    auc_sco1=metrics.roc_auc_score(testing_y,pre1)
    auc_sco2=metrics.roc_auc_score(testing_y,pre2)
    auc.append(auc_sco)
    auc1.append(auc_sco1)
    auc2.append(auc_sco2)
print np.mean(auc),np.mean(auc1),np.mean(auc2)


# In[88]:

auc,sco=[],[]
kf = KFold(df_all_real.shape[0], n_folds=10,shuffle=True)
for train, test in kf:
    training_set,testing_set=df_all_real.iloc[train],df_all_real.iloc[test]
    training_x,training_y=training_set[['sapsii','gender','age_score']].as_matrix(),training_set['expire_flag'].as_matrix()
    testing_x,testing_y=testing_set[['sapsii','gender','age_score']].as_matrix(),testing_set['expire_flag'].as_matrix()
    clf = svm.SVC(C=1,gamma=0.1)
    clf.fit(training_x,training_y)
    pre = clf.predict(testing_x)
    score=clf.score(testing_x,testing_y)
    sco.append(score)
    auc_sco=metrics.roc_auc_score(testing_y,pre)
    auc.append(auc_sco)
print np.mean(auc),np.mean(sco)


# In[89]:

plot_roc(testing_y,pre)


# In[20]:

auc,sco,auc1,auc2=[],[],[],[]
kf = KFold(df_all_real.shape[0], n_folds=10,shuffle=True)
for train, test in kf:
    training_set,testing_set=df_all_real.iloc[train],df_all_real.iloc[test]
    training_x,training_y=np.array(doc_topic[list(training_set['topic']),]),training_set['expire_flag'].as_matrix()
    testing_x,testing_y=np.array(doc_topic[list(testing_set['topic']),]),testing_set['expire_flag'].as_matrix()
    clf = svm.SVC(C=1,gamma=0.1)
    clf.fit(training_x,training_y)
    pre = clf.predict(testing_x)
    clf1=GradientBoostingClassifier(n_estimators=10)
    clf1.fit(training_x,training_y)
    pre1 = clf1.predict(testing_x)
    clf2=RandomForestClassifier(max_depth=30, n_estimators=10)
    clf2.fit(training_x,training_y)
    pre2 = clf2.predict(testing_x)
    score=clf.score(testing_x,testing_y)
    sco.append(score)
    auc_sco=metrics.roc_auc_score(testing_y,pre)
    auc_sco1=metrics.roc_auc_score(testing_y,pre1)
    auc_sco2=metrics.roc_auc_score(testing_y,pre2)
    auc.append(auc_sco)
    auc1.append(auc_sco1)
    auc2.append(auc_sco2)
print np.mean(auc),np.mean(sco),np.mean(auc1),np.mean(auc2)


# In[154]:

auc,auc1,auc2=[],[],[]
for train, test in kf:
    training_set,testing_set=df_all_real.iloc[train],df_all_real.iloc[test]
    training_x,training_y=np.array(doc_topic[list(training_set['topic']),]),training_set['dead_in_hospital'].as_matrix()
    testing_x,testing_y=np.array(doc_topic[list(testing_set['topic']),]),testing_set['dead_in_hospital'].as_matrix()
    clf = svm.SVC(C=1,gamma=0.1)
    clf.fit(training_x,training_y)
    pre = clf.predict(testing_x)
    clf1=GradientBoostingClassifier(n_estimators=10)
    clf1.fit(training_x,training_y)
    pre1 = clf1.predict(testing_x)
    clf2=RandomForestClassifier(max_depth=30, n_estimators=10)
    clf2.fit(training_x,training_y)
    pre2 = clf2.predict(testing_x)
    auc_sco=metrics.roc_auc_score(testing_y,pre)
    auc_sco1=metrics.roc_auc_score(testing_y,pre1)
    auc_sco2=metrics.roc_auc_score(testing_y,pre2)
    auc.append(auc_sco)
    auc1.append(auc_sco1)
    auc2.append(auc_sco2)
print np.mean(auc),np.mean(auc1),np.mean(auc2)


# In[82]:

x=np.hstack((np.array(doc_topic),df_all_real[['sapsii','gender','age_score']][:doc_topic.shape[0]].as_matrix()))
y=df_all_real['expire_flag'][:doc_topic.shape[0]].as_matrix()
y


# ## Hyper Parameter Tuning

# In[83]:

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV


C_range = [1e-2, 1, 1e2]
gamma_range = [1e-1, 1, 1e1]
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(y, n_iter=10, test_size=0.2, random_state=42)
grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
grid.fit(x, y)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))


# In[92]:

csvfile.close()

