import pandas as pd
import numpy as np
import datetime



#Loading CSV File and converting it into pandas dataframe
table = pd.read_csv('/Users/sanchitnarang94/Desktop/SML_Crime_Prediction/Final_Clean_Data.csv',header=0)
df1 = pd.DataFrame(table)
df1.replace('', np.nan, inplace=True)
df1.dropna(axis = 0, how ='any')

df1['Latitude'].replace('', np.nan, inplace=True)
df1.dropna(subset=['Latitude'], inplace=True)

df1['Longitude'].replace('', np.nan, inplace=True)
df1.dropna(subset=['Longitude'], inplace=True)

df1['Primary Type'].replace('NON-CRIMINAL', np.nan, inplace=True)
df1.dropna(subset=['Primary Type'], inplace=True)

#Defining Mapping for Y_label Primary Type
type_mapping = {'BATTERY':1, 'OTHER OFFENSE':2, 'THEFT':3, 'ASSAULT':4, 'CRIMINAL TRESPASS':5, 'NARCOTICS':6,
			'CRIMINAL DAMAGE':7, 'BURGLARY':8, 'WEAPONS VIOLATION':9, 'MOTOR VEHICLE THEFT':10,
			'ROBBERY':11, 'INTERFERENCE WITH PUBLIC OFFICER':12, 'SEX OFFENSE':13, 'DECEPTIVE PRACTICE':14,
			'OFFENSE INVOLVING CHILDREN':15, 'ARSON':16, 'PUBLIC PEACE VIOLATION':17, 'CRIM SEXUAL ASSAULT':18,
			'STALKING':19, 'HOMICIDE':20, 'PROSTITUTION':21, 'LIQUOR LAW VIOLATION':22, 'KIDNAPPING':23,
			'INTIMIDATION':24, 'OBSCENITY':25, 'CONCEALED CARRY LICENSE VIOLATION':26, 'HUMAN TRAFFICKING':27,
			'OTHER NARCOTIC VIOLATION':28, 'NON-CRIMINAL':29, 'PUBLIC INDECENCY':30, 'GAMBLING':31,
			'NON-CRIMINAL (SUBJECT SPECIFIED)':32, 'RITUALISM':33}

#Mapping Primary type with integer value
df1 = df1.replace({'Primary Type':type_mapping})

#Type casting Boolean values to integer and adding 1 to avoid 0
#df.Arrest = df.Arrest.astype(int)+1
#df.Domestic = df.Domestic.astype(int)+1


#Dropping unimportant features
df1 = df1.drop(['ID','Case Number','Description','Location Description','FBI Code','Block','IUCR','Updated On','Location',
				'X Coordinate','Y Coordinate','SplitYear','Arrest','Domestic'],1)

#Loading Date in a matrix
date = df1.as_matrix(columns=['SplitDate'])

#Spliting date in month year and date and getting the weekday out of it
day_of_the_week = []
for i in range(len(date)):
	date_format = date[i][0]
	date_format = date_format.split('/')
	if len(date_format[2])==2:
		date_format[2] = '20'+date_format[2] #Converting YY fromat into YYYY
	day_of_the_week.append(datetime.datetime(int(date_format[2]),int(date_format[0]),int(date_format[1])).weekday() + 1) #Getting day of the week and adding 1 to avoid 0

#Loading Day of the week as column in pandas dataframe
df1['day_of_the_week'] = day_of_the_week


#Loading time and AM/PM in a matrix
time_Col = df1.as_matrix(columns=['Time'])
ampm_Col = df1.as_matrix(columns=['AM/PM'])

time_24_hour = []

#Converting 12 hour into 24 hour
for i in range(len(time_Col)):
	time = str(time_Col[i][0]).split(':') #Split Time in HH MM SS
	ampm = str(ampm_Col[i][0])
	if ampm=='PM':
		time[0] = int(time[0]) #Taking only HH value
		time[0]+=12 #Adding 12 if it is PM
		if(time[0]>=24): #If HH >=24, reset it to 0
			time[0]=00
	time_24_hour.append(time[0])


#Loading Array as column
df1['time_24_hour'] = time_24_hour

#Converting column as int type
df1.time_24_hour = df1.time_24_hour.astype(int)

#Loading Column in a matrix
day_of_the_week = df1.as_matrix(columns=['day_of_the_week'])

day = []

#For values in column day_of_the_week, append list as 1 for that day and 0 for other
for i in range(len(day_of_the_week)):
	if day_of_the_week[i][0]==1:
		day.append([1,0,0,0,0,0,0])
	elif day_of_the_week[i][0]==2:
		day.append([0,1,0,0,0,0,0])
	elif day_of_the_week[i][0]==3:
		day.append([0,0,1,0,0,0,0])
	elif day_of_the_week[i][0]==4:
		day.append([0,0,0,1,0,0,0])
	elif day_of_the_week[i][0]==5:
		day.append([0,0,0,0,1,0,0])
	elif day_of_the_week[i][0]==6:
		day.append([0,0,0,0,0,1,0])
	elif day_of_the_week[i][0]==7:
		day.append([0,0,0,0,0,0,1])



#Create another dataframe from array "day" and put it as column of the dataframe
df2 = pd.DataFrame(day,columns=list(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']))

#Concatenate both the dataframes
df = pd.concat([df1,df2],axis=1)


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

df6 = df.copy()

df6['Primary Type'] = pd.to_numeric(df6['Primary Type'], errors='coerce')
# df6['Primary Type'].replace('29', np.nan, inplace=True)
# df6.dropna(subset=['Primary Type'], inplace=True)

X = pd.DataFrame()
X['time'] = df6['time_24_hour']
X['latitude'] = df6['Latitude']
X['longitude'] = df6['Longitude']
X['community_area'] = df6['Community Area']
X['Monday'] = df6['Monday']
X['Tuesday'] = df6['Tuesday']
X['Wednesday'] = df6['Wednesday']
X['Thursday'] = df6['Thursday']
X['Friday'] = df6['Friday']
X['Saturday'] = df6['Saturday']
X['Sunday'] = df6['Sunday']
X['ward'] = df6['Ward']
X['district'] = df6['District']
X['beat'] = df6['Beat']
X['type'] = df6['Primary Type']

X.replace('', np.nan, inplace=True)
X = X.dropna(axis=0)
y = X['type']
y = y.astype('int')
X = X.drop(['type'], axis = 1)

#Scale our features
scaler = StandardScaler()
X = scaler.fit_transform(X)

#build test and train data
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size = 0.2, random_state = 42)


model = LogisticRegression(multi_class = 'multinomial', penalty = 'l2', C=1, solver ='newton-cg')
model.fit(X_train, y_train)

print "Accuracy is %2.2f" %accuracy_score(y_test, model.predict(X_test))
