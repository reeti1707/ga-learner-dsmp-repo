# --------------
# import the libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv(path)
print(df.head())
df.columns
X = df.drop('insuranceclaim',axis = 1)
y = df['insuranceclaim']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 6)


# Code ends here


# --------------
import matplotlib.pyplot as plt

import seaborn as sns
# Code starts here
sns.boxplot(X_train['bmi'])

q_value = np.quantile(X_train['bmi'],0.95)
# Code ends here
y_train.value_counts()


# --------------
# Code starts here
relation = X_train.corr()

sns.pairplot(X_train)
# Code ends here


# --------------
import seaborn as sns
import matplotlib.pyplot as plt
#Predictor check! Let's check the count_plot for different features vs target variable insuranceclaim. This tells us which features are highly correlated with the target variable insuranceclaim and help us predict it better.
# Code starts here
cols = ['children','sex','region','smoker']
fig ,axes = plt.subplots(nrows = 2 , ncols = 2)
for i in range(0,2):
    for j in range(0,2):
        col = cols[i*2+j]
        sns.countplot(x=X_train[col], hue=y_train, ax=axes[i,j])


# Code ends here


# --------------
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# parameters for grid search
parameters = {'C':[0.1,0.5,1,5]}

# Code starts here
lr = LogisticRegression()
grid = GridSearchCV(estimator=lr,param_grid=parameters)
grid.fit(X_train,y_train)
y_pred = grid.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# Code ends here


# --------------
from sklearn.metrics import roc_auc_score
from sklearn import metrics
#Performance of a classifier !: Now let's visualize the performance of a binary classifier. Check the performance of the classifier using roc auc curve.
# Code starts here
print(df.columns)
score = roc_auc_score(y_test,y_pred)
y_pred_proba = [row[1] for row in grid.predict_proba(X_test)]
fpr,tpr,thresholds = metrics.roc_curve(y_test,y_pred)
roc_auc = roc_auc_score(y_test,y_pred_proba)
plt.plot(fpr,tpr,label="Logistic model, auc="+str(roc_auc))


# Code ends here


