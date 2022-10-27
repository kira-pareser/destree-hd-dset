#### HD-dataset 99% Test Simple
```
#loading dataset
import pandas as pd
import numpy as np
#EDA
from collections import Counter
# data preprocessing
from sklearn.preprocessing import StandardScaler
# data splitting
from sklearn.model_selection import train_test_split
# data modeling
from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier ##TREE
#ensembling
from sklearn.model_selection import cross_val_score
from sklearn import tree
```
```
data = pd.read_csv('heart.csv')
def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result
 ```
 ```
# data=normalize(data)
# data.isnull().sum()
# data.head()
```

DECISION TREE
```
y = data["target"]
X = data.drop('target',axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,shuffle=True,random_state=3)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
```
m6 = 'DecisionTreeClassifier'
dt = DecisionTreeClassifier(criterion = 'gini',random_state=0,max_depth = None, min_samples_split=2, min_samples_leaf = 1, max_leaf_nodes = None, min_impurity_decrease=0.0)
dt.fit(X_train,y_train)
dt_predicted = dt.predict(X_test)
```
```
dt_conf_matrix = confusion_matrix(y_test, dt_predicted)
dt_acc_score = accuracy_score(y_test, dt_predicted)
print("confussion matrix")
print(np.flip(np.array(dt_conf_matrix)))
print("\n")
print("Accuracy of DecisionTreeClassifier:",dt_acc_score*100,'\n')
print(classification_report(y_test,dt_predicted))
```

```
#plot tree
import graphviz
dot_data = tree.export_graphviz(dt, out_file=None) 
graph = graphviz.Source(dot_data) 
graph

```
