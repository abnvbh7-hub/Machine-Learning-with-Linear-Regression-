import pandas as pd
import plotly.express as px
from sklearn.metrics import r2_score,root_mean_squared_error
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.linear_model import LinearRegression
from datetime import datetime
import numpy as  np
np.set_printoptions(suppress=True)

#data loading
df = pd.read_html('/content/Agmarknet_Price_And_Arrival_Report.xls')
df = pd.DataFrame(df[0])

#data cleaning nd processing
df.drop(columns=df.columns[[0,1,2,4]],inplace= True)
df.rename(columns ={'Variety':'type',
                    'Arrivals (Tonnes)':'arrival',
                     'Min Price (Rs./Quintal)':'min',
                    'Max Price (Rs./Quintal)':'max',
                    'Modal Price (Rs./Quintal)':'modal',
                    'Reported Date': 'date'
                    },inplace = True)
df['date'] = pd.to_datetime(df['date'])
df.sort_values(by=['date'],inplace=True)
df.reset_index(inplace = True,drop=True)
df.fillna(0,inplace=True)

# converting categorical columns
df['type_code']= df['type'].replace('Local',1,regex=True,inplace=True)
df['type_code'] = df['type'].replace('Other',0,regex=True,inplace=True)
df.drop(columns='type_code',inplace=True)
df['ordinal_date'] = df['date'].apply(lambda x : x.toordinal())
df['month'] = df['date'].dt.year

#plotting arrival
fig1 = px.line(df,x ='ordinal_date',y ='arrival')

#plotting correlation map 
cdf = df.corr()
corr_map = px.imshow(cdf,color_continuous_scale ='viridis',title ='correlation map',text_auto = True)

#scaling data for better uniformity
scaler = MinMaxScaler()
sf = df
sf['ordinal_date'] = scaler.fit_transform(df[['ordinal_date']])
sf['min'] = scaler.fit_transform(df[['min']])
sf['max'] = scaler.fit_transform(df[['max']])
sf['arrival'] = scaler.fit_transform(df[['arrival']])

#fittin the data with a LINEAR REGRESSION MODEL
model = LinearRegression()
input = sf[['ordinal_date','arrival','type']]
output= sf['modal']
model.fit(input,output)
pred = model.predict(input)
df['predictions'] = pred


#EVALUATION
res = df.drop(columns = df.columns[[0,1,3,2,6]])
res = res[['date','modal','predictions']]
res['modal']=res['modal']/100
res['predictions'] = res['predictions']/100
res[['predictions', 'modal']]=res[['predictions','modal']].astype(int)
res.tail(30)
rms = root_mean_squared_error(sf['modal'],sf['predictions'])
r2 = r2_score(sf['modal'],sf['predictions'])

#Plotting predicted and actual data
fig2 = px.line(res,x = 'date',y = 'modal',hover_data=['predictions','modal'])
fig2.update_traces(line={'color':'#986789'})
fig2.add_scatter(x = res['date'],y = res['predictions'],line={'color':'black','width':1})


fig2.update_layout(
    title = 'ACTUAL VS PREDICTION (Modal)',
    title_font={'size':20,'color':'#011f4b'},
    xaxis_title='Date',
    yaxis_title = 'Modal',
    plot_bgcolor='#ffffff',
    paper_bgcolor ='#ffffff')

fig2.update_yaxes(
    titlefont={'size':16,'color':'#011f4b'},
    showgrid=False ,
    gridwidth = 0.3,
    gridcolor = '#005b96')

fig2.update_xaxes(
    titlefont = {'size':16,'color':'#011f4b'},
    showgrid = False,
    gridwidth = 0.3,
    gridcolor = '#005b96')
