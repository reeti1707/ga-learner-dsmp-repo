# --------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Code starts here

df =pd.read_csv(path)
print(df.head())
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state =4)
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
# Cscalode ends here


# --------------
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
roc_score = roc_auc_score(y_pred, y_test)
print("roc score: %.2f " % roc_score)


# --------------
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=4)
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)
roc_score = roc_auc_score(y_pred, y_test)
print("roc scoreusing decision tree: %.2f " % roc_score)
#print(roc_score)


# --------------
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=4)

# Code strats here

rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)
roc_score = roc_auc_score(y_pred, y_test)
print("roc score with Random forest algorithm: %.2f " % roc_score)

# Code ends here


# --------------
# Import Bagging Classifier
from sklearn.ensemble import BaggingClassifier
bagging_clf = BaggingClassifier(base_estimator= DecisionTreeClassifier(), n_estimators=100 , max_samples=100,random_state=0)

bagging_clf.fit(X_train,y_train)

score_bagging = bagging_clf.score(X_test,y_test)
print('score_bagging is:%.2f'%score_bagging)



# Code ends here


# --------------
# Import libraries
from sklearn.ensemble import VotingClassifier

# Various models
clf_1 = LogisticRegression()
clf_2 = DecisionTreeClassifier(random_state=4)
clf_3 = RandomForestClassifier(random_state=4)

model_list = [('lr',clf_1),('DT',clf_2),('RF',clf_3)]


voting_clf_hard = VotingClassifier(estimators = model_list,voting = 'hard')
voting_clf_hard.fit(X_train,y_train)
hard_voting_score = voting_clf_hard.score(X_test,y_test)
print("Hard Voting Train Accuracy:%.2f"%hard_voting_score)

voting_clf_soft = VotingClassifier(estimators = model_list,voting = 'soft')
voting_clf_soft.fit(X_train,y_train)
soft_voting_score = voting_clf_soft.score(X_test,y_test)
print("soft Voting Train Accuracy:%.2f"%soft_voting_score)
# Code starts here



# Code ends here


