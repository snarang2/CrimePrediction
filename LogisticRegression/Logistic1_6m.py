import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

table = pd.read_csv('/Users/sanchitnarang94/Desktop/SML_Crime_Prediction/data_change.csv',header=0)

df = pd.DataFrame(table)

df6 = df.copy()

df6['Primary Type'] = pd.to_numeric(df6['Primary Type'], errors='coerce')

X = pd.DataFrame()

# Adding coluns to new data frame.
X['time'] = df6['time_24_hour']

X['latitude'] = df6['Latitude']
X['longitude'] = df6['Longitude']

X['community_area'] = df6['Community Area']
X['ward'] = df6['Ward']
X['district'] = df6['District']
X['beat'] = df6['Beat']

X['Monday'] = df6['Monday']
X['Tuesday'] = df6['Tuesday']
X['Wednesday'] = df6['Wednesday']
X['Thursday'] = df6['Thursday']
X['Friday'] = df6['Friday']
X['Saturday'] = df6['Saturday']
X['Sunday'] = df6['Sunday']

X['1'] = df6['1']
X['2'] = df6['2']
X['3'] = df6['3']
X['4'] = df6['4']
X['5'] = df6['5']
X['6'] = df6['6']
X['7'] = df6['7']
X['8'] = df6['8']
X['9'] = df6['9']
X['10'] = df6['10']
X['11'] = df6['11']
X['12'] = df6['12']
X['13'] = df6['13']
X['14'] = df6['14']
X['15'] = df6['15']
X['16'] = df6['16']
X['17'] = df6['17']
X['18'] = df6['18']
X['19'] = df6['19']
X['20'] = df6['20']
X['21'] = df6['21']
X['22'] = df6['22']
X['23'] = df6['23']
X['24'] = df6['24']
X['25'] = df6['25']
X['26'] = df6['26']
X['27'] = df6['27']
X['28'] = df6['28']
X['29'] = df6['29']
X['30'] = df6['30']
X['31'] = df6['31']

X['type'] = df6['Primary Type']

y = X['type']
y = y.astype('int')

# Drop target value from original data for prediction.
X = X.drop(['type'], axis = 1)

#Scale our features
scaler = StandardScaler()
X = scaler.fit_transform(X)

#build test and train data
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size = 0.2, random_state = 42)


model = LogisticRegression(multi_class = 'multinomial', penalty = 'l2', C=1, solver ='newton-cg')
model.fit(X_train, y_train)

print "Accuracy is %2.2f" %accuracy_score(y_test, model.predict(X_test))
