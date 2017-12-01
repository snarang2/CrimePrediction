import pandas as pd
df = pd.read_csv("/Users/sanchitnarang94/Desktop/SML_Crime_Prediction/Crimes_-_2001_to_present.csv")
df.head()

df4 = df.copy()
df4[['SplitDate', 'Time', 'AM/PM']] = df4['Date'].str.split(' ', expand=True)
df4 = df4.drop('Date', axis=1)
df4.head()

df4[['SplitMonth', 'SplitDay', 'SplitYear']] = df4['SplitDate'].str.split('/', expand=True)
df4.head()

df4.dropna(axis=0, how='any')
df4.to_csv('/Users/sanchitnarang94/Desktop/SML_Crime_Prediction/Final_Clean_Data.csv', encoding='utf-8')
