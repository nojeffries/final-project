# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

manager_survey = pd.read_csv('C:/htoc/biof309/final-project/data/hr-analytics-case-study/manager_survey_data.csv')
employee_survey = pd.read_csv('C:/htoc/biof309/final-project/data/hr-analytics-case-study/employee_survey_data.csv')
general_data = pd.read_csv('C:/htoc/biof309/final-project/data/hr-analytics-case-study/general_data.csv')

pd.set_option('display.max_columns', None)

#print(employee_survey.shape)
#print(employee_survey.head())
#print(manager_survey.shape)
#print(manager_survey.head())
#print(general_data.shape)
#print(general_data.head())

attrition_data = pd.merge(general_data,pd.merge(manager_survey,employee_survey,on='EmployeeID',how='inner'),on='EmployeeID',how='inner')

attrition_data.columns = attrition_data.columns.str.lower()

(dim1,dim2) = attrition_data.shape

for i in range(dim2) :
    print(attrition_data.columns[i],'\t',round(attrition_data.iloc[                 :,i].count()/dim1,2))

attrition_data = attrition_data.dropna()
(dim1,dim2) = attrition_data.shape


#from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import classification_report, confusion_matrix
pd.crosstab(attrition_data['attrition'],columns="count",colnames=[''])

attrition_data['attrition'] = np.where(attrition_data['attrition']=="Yes",1,0) # reset attrition as a 0-1 variable

pd.crosstab(attrition_data['attrition'],columns="count",colnames=[''])


numvars = ['age', 'monthlyincome', 'numcompaniesworked', 'percentsalaryhike', 'totalworkingyears', 'yearsatcompany', 'yearssincelastpromotion', 'yearswithcurrmanager'] 


facvars=['businesstravel','department','education','educationfield','gender','joblevel','jobrole','maritalstatus','stockoptionlevel','jobinvolvement','performancerating','environmentsatisfaction','jobsatisfaction', 'worklifebalance']
print(facvars)

print(numvars)


pd.crosstab(attrition_data['businesstravel'],columns="count",colnames=[''])
attrition_data['travelalot'] = np.where(attrition_data['businesstravel']=="Travel_Frequently",1,0)
pd.crosstab(attrition_data['travelalot'],columns="count",colnames=[''])

pd.crosstab(attrition_data['department'],columns="count",colnames=[''])
attrition_data['randddepartment'] = np.where(attrition_data['department']=="Research & Development",1,0)
pd.crosstab(attrition_data['randddepartment'],columns="count",colnames=[''])

pd.crosstab(attrition_data['educationfield'],columns="count",colnames=[''])
attrition_data['sciencemedicaleduc'] = np.where(attrition_data['educationfield'].isin(['Life Sciences','Medical']),1,0)
pd.crosstab(attrition_data['sciencemedicaleduc'],columns="count",colnames=[''])

pd.crosstab(attrition_data['gender'],columns="count",colnames=[''])
attrition_data['male'] = np.where(attrition_data['gender']=="Male",1,0)
pd.crosstab(attrition_data['male'],columns="count",colnames=[''])

pd.crosstab(attrition_data['jobrole'],columns="count",colnames=[''])
attrition_data['researchjob'] = np.where(attrition_data['jobrole'].isin(['Research Director','Research Scientist']),1,0)
pd.crosstab(attrition_data['researchjob'],columns="count",colnames=[''])

pd.crosstab(attrition_data['maritalstatus'],columns="count",colnames=[''])
attrition_data['evermarried'] = np.where(attrition_data['maritalstatus']=="Single",0,1)
pd.crosstab(attrition_data['evermarried'],columns="count",colnames=[''])

pd.crosstab(attrition_data['environmentsatisfaction'],columns="count",colnames=[''])
attrition_data['highworkenvironment'] = np.where(attrition_data['environmentsatisfaction']>2,1,0)
pd.crosstab(attrition_data['highworkenvironment'],columns="count",colnames=[''])

pd.crosstab(attrition_data['jobsatisfaction'],columns="count",colnames=[''])
attrition_data['highjobsatisfaction'] = np.where(attrition_data['jobsatisfaction']>2,1,0)
pd.crosstab(attrition_data['highjobsatisfaction'],columns="count",colnames=[''])

pd.crosstab(attrition_data['worklifebalance'],columns="count",colnames=[''])
attrition_data['highworklifebalance'] = np.where(attrition_data['worklifebalance']>2,1,0)
pd.crosstab(attrition_data['highworklifebalance'],columns="count",colnames=[''])


pd.crosstab(attrition_data['male'],columns="count",colnames=[''])
attrition_data['male'] = np.where(attrition_data['gender']=="Male",1,0)
pd.crosstab(attrition_data['male'],columns="count",colnames=[''])



newfacvars = (
['travelalot','researchjob','evermarried','highworkenvironment','highjobsatisfaction','highworklifebalance']
)

jobs = attrition_data[['attrition']+numvars+newfacvars]

from sklearn.model_selection import train_test_split
X = jobs[numvars+newfacvars] # Features
y = jobs.attrition # Target variable
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


train = X_train
train = train.assign(attrition=list(y_train))
print("\nTraining data head")
print(train.head())
print(train.shape)
    
import matplotlib.pyplot as plt
from statsmodels.formula.api import logit
import statsmodels.api as sm

# show categorical variables
#for var in facvars: 
#    print('\n \n Univariate analysis of '+var)
#    train[var].hist() 
#    plt.show()
#    form1 = 'attrition ~ C('+var+')'    
#    fit = logit(form1,data=train).fit()
#    print(fit.summary())

# show continuous variables
  
for var in numvars: 
    print('\n \n Univariate analysis of '+var)
    print(train[var].describe()) 
    form1 = 'attrition ~ '+var    
    fit = logit(form1,data=train).fit()
    print(fit.summary())


for var in newfacvars: 
    print('\n \n Univariate analysis of '+var)
    print(train[var].describe()) 
    form1 = 'attrition ~ '+var    
    fit = logit(form1,data=train).fit()
    print(fit.summary())



#variables to keep for Lasso approach - those with p-value < .10
    
    
# recode categorical variables into 0-1 categories
# business travel


  
#numvars2 =   ['age']
#
#newfacvars2 = [
# 'highjobsatisfaction']

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression

from sklearn.metrics import classification_report, confusion_matrix

X_tr = train[numvars+newfacvars]
y_tr = y_train
Xstan_tr = ((X_tr-X_tr.mean())/X_tr.std())


clf = LogisticRegressionCV(Cs=30,cv=20,random_state=0,penalty='l1',solver='liblinear').fit(Xstan_tr,y_tr)
clf.coef_
clf.C_
predprobs = pd.DataFrame(clf.predict_proba(Xstan_tr)).iloc[:,1]

predprobs.describe()

pd.crosstab(y_tr,columns="count")/y_tr.count()
cutpoint = predprobs.quantile(1-.16062)
#zeroone = np.where(predprobs > predprobs.quantile(1-.16062),1,0)
#pd.DataFrame(zeroone).describe()


#
#y_pred = clf.predict(Xstan_tr)
#conf_m = confusion_matrix(y_tr,y_pred)
#print(conf_m)
#
#confusion_matrix(y_tr,y_pred)
#confusion_matrix(y_tr,zeroone)


test = X_test
test = test.assign(attrition=list(y_test))
print("\nTesting data head")
print(test.shape)

X_te = test[numvars+newfacvars]
y_te = y_test
Xstan_te = ((X_te-X_te.mean())/X_te.std())


#clf = LogisticRegressionCV(Cs=30,cv=20,random_state=0,penalty='l1',solver='liblinear').fit(Xstan_te,y_te)
#clf.coef_
#clf.C_
pd.DataFrame(clf.predict_proba(Xstan_te)).iloc[:,1].describe()

#predprobs = pd.DataFrame(clf.predict_proba(Xstan_te)).iloc[:,1]
pd.crosstab(y_te,columns="count")/y_te.count()
cutpoint = predprobs.quantile(1-.16)
zeroone = np.where(predprobs > cutpoint,1,0)
pd.DataFrame(zeroone).describe()



y_pred = clf.predict(Xstan_te)
conf_m = confusion_matrix(y_te,y_pred)
print(conf_m)

confusion_matrix(y_te,y_pred)
confusion_matrix(y_te,zeroone)



#clf = LogisticRegression(solver='lbfgs',penalty='none').fit(X,y)


X = jobs[numvars+newfacvars]
y = jobs.attrition 
Xstan = ((X-X.mean())/X.std())

Xstan2 = sm.add_constant(Xstan_tr)

model = sm.Logit(y_tr, Xstan2)
result = model.fit_regularized(method='l1',alpha=clf.C_)
result.summary()
#
#Xstan['y'] = y
#Xstan.to_csv('c:\\htoc\\biof309\\final-project\\data\\Xstan.csv')
#

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=50).fit(X_tr,y_tr)
y_pred_tr = clf.predict(X_tr)
confusion_matrix(y_tr,y_pred)

y_pred2 = clf.predict(X_te)
confusion_matrix(y_te,y_pred2)

print('Variable importance for Random Forest \n')
pd.DataFrame(clf.feature_importances_,columns = ['Importance'],index=X_te.columns).sort_values(by=['Importance'],ascending=False)
