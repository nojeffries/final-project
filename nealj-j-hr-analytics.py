# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

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

attrition_data = pd.merge(general_data,pd.merge(manager_survey,employee_survey,on='EmployeeID',how='inner'),
                         on='EmployeeID',how='inner')

attrition_data.columns = attrition_data.columns.str.lower()

(dim1,dim2) = attrition_data.shape

for i in range(dim2) :
        print(attrition_data.columns[i],'\t',round(attrition_data.iloc[:,i].count()/dim1,2))

attrition_data = attrition_data.dropna()
(dim1,dim2) = attrition_data.shape


#from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import classification_report, confusion_matrix

attrition_data['attrition'] = np.where(attrition_data['attrition']=="Yes",1,0) # reset attrition as a 0-1 variable

pd.crosstab(attrition_data['attrition'],columns="count")
#lmod = LogisticRegression().fit(attrition_data['performancerating'].reshape(-1, 1),attrition_data['attrition'])

#numvars = ['age','attrition','distancefromhome','monthlyincome','numcompaniesworked',
#           'percentsalaryhike','totalworkingyears','trainingtimeslastyear',
#           'yearssincelastpromotion','yearswithcurrentmanager']
#facvars = ['businesstravel','department','education','educationfield','joblevel','jobrole',
#           'maritalstatus','percentsalaryhike','stockoptionlevel','jobinvolvement',
#           'performancerating','environmentsatisfaction','jobsatisfaction',
#           'worklifebalance']

#statsmodels.formula.api.logit(formula, data, subset=None, drop_cols=None, *args, **kwargs)


numvars = []
facvars = []
for i in range(dim2):
    if len(attrition_data.iloc[:,i].unique()) < 7:
        facvars.append(attrition_data.columns[i])
    else:
        numvars.append(attrition_data.columns[i])

dropfac = ['employeecount','over18','standardhours'] #no variability in these measurements
dropnum = ['employeeid'] # matching id - not use in analysis

for var in dropfac:
    facvars.remove(var)

for var in dropnum:
    numvars.remove(var)

    
import matplotlib.pyplot as plt
from statsmodels.formula.api import logit

# show categorical variables
for var in facvars[1:]: 
    print('\n \n Anaysis of'+var)
    attrition_data[var].hist() 
    plt.show()
    form1 = 'attrition ~ C('+var+')'    
    fit = logit(form1,data=attrition_data).fit()
    print(fit.summary())

# show continuous variables
  
for var in numvars[1:]: 
    print('\n \n Anaysis of'+var)
    print(attrition_data[var].describe()) 
    form1 = 'attrition ~ '+var    
    fit = logit(form1,data=attrition_data).fit()
    print(fit.summary())


    

from statsmodels.formula.api import logit
