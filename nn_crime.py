import pandas as pd
import numpy as np
import datetime
from sklearn import linear_model
from sklearn import model_selection
from sklearn import neural_network
from sklearn import decomposition
from sklearn import svm

table = pd.read_csv('data_change.csv',index_col=0,header=0)
df = pd.DataFrame(table)
print (df)

train, test = model_selection.train_test_split(df, test_size=0.3,random_state=42)
#train, test = model_selection.KFold(n_splits=5, shuffle=True, random_state=42)




y_train = train[['Primary Type']]
x_train = train.drop(['Primary Type'],1)

y_test = test[['Primary Type']]
x_test = test.drop(['Primary Type'],1)


#Y = df[['Primary Type']]
#X = df.drop(['Primary Type'],1)


#nn = neural_network.MLPClassifier(hidden_layer_sizes=(128,64,32),activation='logistic')

#svm = svm.SVC(decision_function_shape='ovo')

#print (model_selection.cross_val_score(svm,X,Y,scoring='accuracy',cv=5))

#print(linear_model.LogisticRegression(tol=0.0001,multi_class='multinomial',solver='newton-cg',).fit(x_train, y_train).score(x_test,y_test))

#print (neural_network.MLPClassifier(hidden_layer_sizes=(128,64,32),activation='logistic').fit(x_train, y_train).score(x_test,y_test))

print (svm.SVC(decision_function_shape='ovo').fit(x_train, y_train).score(x_test,y_test))