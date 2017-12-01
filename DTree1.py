
import pandas as pd
df10 = pd.read_csv("C:/Users/Abhishek/Desktop/SML Project/data_change.csv")

#df10.drop(['AM/PM'])
X = df10.ix[1:,2:]
y = df10.ix[1:,1]


print(y)

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

dtree_model = DecisionTreeClassifier(max_depth = 10).fit(X_train, y_train)
dtree_predictions = dtree_model.predict(X_test)
 
# creating a confusion matrix
cm = confusion_matrix(y_test, dtree_predictions)
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, dtree_predictions)
print(score*100)
#
model = ExtraTreesClassifier()
model.fit(X_train, y_train)
print(model.feature_importances_)

import matplotlib.pyplot as plt
important = model.feature_importances_
plt.hist(important)
plt.title("Important Features Histogram")
plt.xlabel("Feature Value")
plt.show()

#
#import numpy as np
#import seaborn as sb
#import matplotlib.pyplot as plt
#cols = ['Primary Type','Beat','District','Ward','Community Area','Latitude','Longitude', 'time_24_hour','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
#cor_matrix = np.corrcoef(df10[cols].values.T)
#sb.set(font_scale=1.5)
#cor_heat_map = sb.heatmap(cor_matrix, cbar=True, annot=True, fmt='.2f', annot_kws={'size':9}, yticklabels=cols, xticklabels=cols) #, 
#plt.show()
#
#model = ExtraTreesClassifier()
#model.fit(X_train, y_train)
#print(model.feature_importances_)




