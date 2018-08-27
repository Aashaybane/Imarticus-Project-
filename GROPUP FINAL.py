
# Importing the libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Importing the dataset

dataset = pd.read_csv('XYZCorp_LendingData.txt', sep="\t",lineterminator='\r')
dataset.isnull().sum()                          #count of missing values 


#dropping the columns which has more than 60% missing values

dataset=dataset.drop(['desc','mths_since_last_record','mths_since_last_major_derog','annual_inc_joint','dti_joint',
                      'verification_status_joint','open_acc_6m','open_il_6m','open_il_12m','open_il_24m',
                      'mths_since_rcnt_il','total_bal_il','il_util','open_rv_12m','open_rv_24m','max_bal_bc',
                      'all_util','inq_fi','total_cu_tl','inq_last_12m'],axis=1)

#dropping the colloums with irrelavence

dataset=dataset.drop(["id","member_id","sub_grade","emp_title","member_id","pymnt_plan","title","zip_code",
                     "earliest_cr_line","out_prncp","total_rec_prncp","total_rec_int","recoveries",
                    "collection_recovery_fee","last_pymnt_d","last_pymnt_amnt","next_pymnt_d",
                     "last_credit_pull_d","collections_12_mths_ex_med","policy_code","application_type",
                     "acc_now_delinq","tot_coll_amt"],axis=1)
dataset.isnull().sum()                          #count of missing values 

#checking values present in default_ind variable

list(dataset['default_ind'].unique() ) 


#checking values present in emp_length variable

list(dataset['emp_length'].unique() ) 

#filling missing values

dataset['emp_length'].fillna(dataset['emp_length'].mode()[0],inplace=True)

#droping the irrelavent column

dataset=dataset.drop(['revol_util', ], axis = 1) 


#filling the missing values

dataset['tot_cur_bal'].fillna(dataset['tot_cur_bal'].mean(), inplace = True)

#filling the missing values

dataset['mths_since_last_delinq'].fillna(dataset['mths_since_last_delinq'].mean(), inplace = True)

#filling the missing values

dataset['total_rev_hi_lim'].fillna(dataset['total_rev_hi_lim'].median(), inplace = True)

#removing the row will all missing values 

dataset[dataset.isnull().any(axis=1)]               #TO fIND THE ROW WITH MISSING VALUES
print(dataset.iloc[[855969]])           #confirming the row with missing values 
dataset=dataset.drop(dataset.index[855969])

#applying label encoder to train set with selected columns 

from sklearn import preprocessing
colname= [ 'grade','home_ownership','term','verification_status','purpose','addr_state','initial_list_status',
           'total_rev_hi_lim','emp_length']

le={}

for x in colname:
    le[x]=preprocessing.LabelEncoder()
    
for x in colname:
    dataset[x]=le[x].fit_transform(dataset.__getattr__(x))
    
#spliting into train test split with regards to issue date
    
dataset['issue_d'] = pd.to_datetime(dataset['issue_d'])
split_date = pd.datetime(2015,6,1)
train = dataset.loc[dataset['issue_d'] < split_date]
test = dataset.loc[dataset['issue_d'] >= split_date]

#dropping the issue_d column 

train= train.drop(['issue_d'], axis = 1)
test = test.drop(['issue_d'], axis = 1)

#Know the ratio of '1' and '0' 

sns.countplot(x='default_ind',data=dataset, palette='hls')
plt.show()


#UPSAMPLING OF TRAINING DATA
from sklearn.utils import resample

# Separate majority and minority classes
train_majority = train[train.default_ind==0]
train_minority = train[train.default_ind==1]

# Upsample minority class
train_minority_upsampled = resample(train_minority,replace=True,
                                    n_samples=552529,
                                    random_state=123) # reproducible results

# Combine majority class with upsampled minority class
train_upsampled = pd.concat([train_majority, train_minority_upsampled])

# Display new class counts
train_upsampled.default_ind.value_counts()

#spliting the datase into train and test
X_train=train_upsampled.values[:,:-1]
Y_train=train_upsampled.values[:,-1]
X_test=test.values[:,:-1]
Y_test=test.values[:,-1]

#fitting the logistic regression model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression( random_state = 0)
classifier.fit(X_train, Y_train)

#predicting the test result
Y_pred = classifier.predict(X_test)

#checking the accuracy of model

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cfm = confusion_matrix(Y_test, Y_pred)
print(cfm)

print("calssification_report:")

print (classification_report(Y_test, Y_pred))

acc = accuracy_score(Y_test, Y_pred)
print ('acc', acc)

#Plotting ROC curve
from sklearn import metrics
fpr, tpr, threshold = metrics.roc_curve(Y_test, Y_pred)
auc = metrics.auc(fpr,tpr)
print(auc)

plt.title('ROC Curve')
plt.plot(fpr, tpr, 'b', label = auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#Tuning the model by changing the threshold value
y_pred_prob = classifier.predict_proba(X_test)
print(y_pred_prob)

for a in np.arange(0,1,0.05):
     predict_mine = np.where(y_pred_prob[:,0] < a,1,0)
     cfm = confusion_matrix(Y_test.tolist(),predict_mine)
     total_err = cfm[0,1] + cfm[1,0]
     print("Errors at threshold ",a,":",total_err,"type 2 error :",cfm[1,0])

y_pred_class=[]
for value in y_pred_prob[:,0]:
    if value < 0.10:
        y_pred_class.append(1)
    else:
        y_pred_class.append(0)
        
print(y_pred_class)

cfm=confusion_matrix(Y_test.tolist(),y_pred_class)
print("Confusion Matrix :")
print(cfm)
print("Classification report: ")

print(classification_report(Y_test.tolist(),y_pred_class))

acc=accuracy_score(Y_test.tolist(),y_pred_class)
print("Accuracy of the model: ",acc)

#ROC curve

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


#ploting the combination 
sns.barplot(x='default_ind',y='loan_amnt',data=dataset)
sns.barplot(x='default_ind',y='int_rate',data=dataset)
sns.barplot(x='default_ind',y='annual_inc',data=dataset)
