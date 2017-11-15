import pandas as pd
import numpy as np
import datetime
from sklearn import linear_model
from sklearn import model_selection
from sklearn import neural_network



#Loading CSV File and converting it into pandas dataframe
table = pd.read_csv('/Users/sanchitnarang94/Desktop/SML_Crime_Prediction/Final_Clean_Data.csv',header=0)
df1 = pd.DataFrame(table)


#Defining Mapping for Y_label Primary Type
type_mapping = {'BATTERY':1, 'OTHER OFFENSE':3, 'THEFT':2, 'ASSAULT':1, 'CRIMINAL TRESPASS':2, 'NARCOTICS':2,
			'CRIMINAL DAMAGE':1, 'BURGLARY':2, 'WEAPONS VIOLATION':1, 'MOTOR VEHICLE THEFT':2,
			'ROBBERY':2, 'INTERFERENCE WITH PUBLIC OFFICER':3, 'SEX OFFENSE':2, 'DECEPTIVE PRACTICE':2,
			'OFFENSE INVOLVING CHILDREN':2, 'ARSON':1, 'PUBLIC PEACE VIOLATION':2, 'CRIM SEXUAL ASSAULT':1,
			'STALKING':3, 'HOMICIDE':1, 'PROSTITUTION':3, 'LIQUOR LAW VIOLATION':3, 'KIDNAPPING':1,
			'INTIMIDATION':2, 'OBSCENITY':3, 'CONCEALED CARRY LICENSE VIOLATION':1, 'HUMAN TRAFFICKING':1,
			'OTHER NARCOTIC VIOLATION':3, 'NON-CRIMINAL':3, 'PUBLIC INDECENCY':3, 'GAMBLING':3,
			'NON-CRIMINAL (SUBJECT SPECIFIED)':3, 'RITUALISM':3}

'''high threat = {battery,assault,criminal damage,weapon violation,arson,crim sexual assault,homicide,kidnapping,concealed carry,human trafficking,
				}
mid threat = {theft,narcotics,criminal trespas,burglary,motor vehicle theft,robbery,sex offence,deceptive practice,offence involving children,public peace
				violation,intimidation,}
low threat = {other offense,interference with public officer,stalking,prostitution,liquor law violation,obscenity,other narcotic
				violation,non criminal,public indecency,gambling, non criminal subject specified,ritualism}'''

#Mapping Primary type with integer value
df1 = df1.replace({'Primary Type':type_mapping})

#Type casting Boolean values to integer and adding 1 to avoid 0
#df.Arrest = df.Arrest.astype(int)+1
#df.Domestic = df.Domestic.astype(int)+1


#Dropping unimportant features
df1 = df1.drop(['Unnamed: 0','ID','Case Number','Description','Location Description','FBI Code','Block','IUCR','Updated On','Location',
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
day = np.zeros([df1.shape[0],7])
for i in range(df1.shape[0]):
	day[i][day_of_the_week[i][0]-1] = 1




month_matrix = df1.as_matrix(columns=['SplitMonth'])
month = np.zeros([df1.shape[0],12])
for i in range(df1.shape[0]):
	month[i][month_matrix[i][0]-1] = 1




day_matrix = df1.as_matrix(columns=['SplitDay'])

day_date = np.zeros([df1.shape[0],31])
for i in range(df1.shape[0]):
	day_date[i][day_matrix[i][0]-1] = 1




#Create another dataframe from array "day" and put it as column of the dataframe
df2 = pd.DataFrame(day,columns=list(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']))

df3 = pd.DataFrame(month,columns=list(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']))

df4 = pd.DataFrame(day_date,columns=list(['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'
										,'21','22','23','24','25','26','27','28','29','30','31']))


#Concatenate both the dataframes
df = pd.concat([df1,df2,df3,df4],axis=1)

df = df.drop(['day_of_the_week','SplitDate','Time','AM/PM'],1)

df['Latitude'].replace('', np.nan, inplace=True)
df.dropna(subset=['Latitude'], inplace=True)

df['Longitude'].replace('', np.nan, inplace=True)
df.dropna(subset=['Longitude'], inplace=True)

df['Beat'].replace('', np.nan, inplace=True)
df.dropna(subset=['Beat'], inplace=True)

df['District'].replace('', np.nan, inplace=True)
df.dropna(subset=['District'], inplace=True)


df['Ward'].replace('', np.nan, inplace=True)
df.dropna(subset=['Ward'], inplace=True)

df['Community Area'].replace('', np.nan, inplace=True)
df.dropna(subset=['Community Area'], inplace=True)

df['SplitMonth'].replace('', np.nan, inplace=True)
df.dropna(subset=['SplitMonth'], inplace=True)

df['SplitDay'].replace('', np.nan, inplace=True)
df.dropna(subset=['SplitDay'], inplace=True)

df['time_24_hour'].replace('', np.nan, inplace=True)
df.dropna(subset=['time_24_hour'], inplace=True)

df['Primary Type'].replace('NON - CRIMINAL', np.nan, inplace=True)
df.dropna(subset=['Primary Type'], inplace=True)

df['Monday'].replace('', np.nan, inplace=True)
df.dropna(subset=['Monday'], inplace=True)

df['Tuesday'].replace('', np.nan, inplace=True)
df.dropna(subset=['Tuesday'], inplace=True)

df['Wednesday'].replace('', np.nan, inplace=True)
df.dropna(subset=['Wednesday'], inplace=True)

df['Thursday'].replace('', np.nan, inplace=True)
df.dropna(subset=['Thursday'], inplace=True)

df['Friday'].replace('', np.nan, inplace=True)
df.dropna(subset=['Friday'], inplace=True)

df['Saturday'].replace('', np.nan, inplace=True)
df.dropna(subset=['Saturday'], inplace=True)

df['Sunday'].replace('', np.nan, inplace=True)
df.dropna(subset=['Sunday'], inplace=True)

df['Jan'].replace('', np.nan, inplace=True)
df.dropna(subset=['Jan'], inplace=True)

df['Feb'].replace('', np.nan, inplace=True)
df.dropna(subset=['Feb'], inplace=True)

df['Mar'].replace('', np.nan, inplace=True)
df.dropna(subset=['Mar'], inplace=True)

df['Apr'].replace('', np.nan, inplace=True)
df.dropna(subset=['Apr'], inplace=True)

df['May'].replace('', np.nan, inplace=True)
df.dropna(subset=['May'], inplace=True)

df['Jun'].replace('', np.nan, inplace=True)
df.dropna(subset=['Jun'], inplace=True)

df['Jul'].replace('', np.nan, inplace=True)
df.dropna(subset=['Jul'], inplace=True)

df['Aug'].replace('', np.nan, inplace=True)
df.dropna(subset=['Aug'], inplace=True)

df['Sep'].replace('', np.nan, inplace=True)
df.dropna(subset=['Sep'], inplace=True)

df['Oct'].replace('', np.nan, inplace=True)
df.dropna(subset=['Oct'], inplace=True)

df['Nov'].replace('', np.nan, inplace=True)
df.dropna(subset=['Nov'], inplace=True)

df['Dec'].replace('', np.nan, inplace=True)
df.dropna(subset=['Dec'], inplace=True)




df.dropna()

df = df.drop(['Year','SplitMonth','SplitDay'],1)

df.to_csv('/Users/sanchitnarang94/Desktop/SML_Crime_Prediction/data_change_allyears.csv.csv')

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

table = pd.read_csv('/Users/sanchitnarang94/Desktop/SML_Crime_Prediction/data_change_allyears.csv',header=0)

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
