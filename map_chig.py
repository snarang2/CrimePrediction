

import gmplot
import pandas as pd 
gmap = gmplot.GoogleMapPlotter.from_geocode("Chicago")
#gmap = gmplot.from_geocode("Chicago")
df =pd.read_csv('C:\Users\pd\Documents\sml_pro\pp.csv')

df1 = df[['Latitude','Longitude','Primary Type']] 
#classi = df1.as_matrix()
#print df1 
class1 = df['Primary Type'] == 1 
sf1= df1[class1]
class2 = df['Primary Type'] == 2 
sf2 = df1[class2]
gmap.scatter(sf1['Latitude'], sf2['Longitude'], '#000000', size=50, marker=False)
gmap.scatter(sf1['Latitude'], sf2['Longitude'], '#FFF0000', size=50, marker=False)
        




gmap.draw("mymap.html")
#gmap.heatmap_weighted(df1['Latitude'] , df1['Longitude'] ,df['Primary Type'])




# import plotly.graph_objs as go 
# from plotly.offline import init_notebook_mode,iplot
# init_notebook_mode(connected=True) 
# import pandas as pd
# df =pd.read_csv('C:\Users\pd\Documents\sml_pro\pp.csv')
# df = df[['Latitude','Longitude','Primary Type']] 
# df.head()

# data = dict(type='choropleth',
                # locations = df['Longitude' , 'Latitude'],
                # locationmode = 'country names',
                # z = df['    Primary Type'],
                # text = df['Country'],
                # colorbar = {'title':'Power Consumption KWH'},
                # colorscale = 'Viridis',
                # reversescale = True
                # )

# # Lets make a layout
# layout = dict(title='2014 World Power Consumption',
# geo = dict(showframe=False,projection={'type':'Mercator'}))
# # Pass the data and layout and plot using iplot
# choromap = go.Figure(data = [data],layout = layout)
# iplot(choromap,validate=False)





