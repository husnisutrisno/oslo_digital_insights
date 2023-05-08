

import os, types
import pandas as pd
from botocore.client import Config
import ibm_boto3


#from jupyter_dash import JupyterDash
import dash
from dash import html, dcc
from dash.dependencies import Output, Input, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
#from datetime import datetime
pio.templates.default = "simple_white"

from wordcloud import WordCloud
from pytrends.request import TrendReq
import pytrends
# from nltk.corpus import stopwords
# import nltk
# nltk.download('stopwords')
# stop_words = stopwords.words('norwegian')

## Keywords for google trends
keyword = ['transport', 'tog', 'elbil', 'streaming', 'posten']

## setup color palette for WSP
lightblue = '#D8E6F0'
red = '#F9423A'
charcoal_dark = '#1E252B'
charcoal_blue = '#343E48'
light_gray = '#D9D9D6'
light_gray50 = '#EFECEA'


# In[94]:


def __iter__(self): return 0
endpoint_access = 'https://s3.eu.cloud-object-storage.appdomain.cloud'

client_access = ibm_boto3.client(
    service_name='s3',
    ibm_api_key_id='Ok05yixj-YW-zXdte-irTwTCu6UGQhpROV8kPd6SYaIH',
    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url=endpoint_access)

def download_data(file):
    body = client_access.get_object(Bucket='digitalinsightspublictransport-donotdelete-pr-zraerohsdy8fer',Key=file)['Body']
    if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )
    df = pd.read_csv(body)
    return df


############################### Oslo mobility data ##################################################
df_oslo = download_data('Oslo_processed_data_for_Dashboard.csv')
df_oslo_mob = download_data('Oslo_mobility_data_for_Dashboard.csv')
df_norway_mob = download_data('Norway_mobility_data_for_Dashboard.csv')

################################# E-commerce data ####################################################
df_ecommerce = download_data('online_payment_data_for_norway.csv')
# df_ecommerce['years']=pd.to_datetime(df_ecommerce['years'], format='%Y')
df_ecommerce = df_ecommerce[df_ecommerce.years>2014]

################################# Car sales data ##################################################
df_car = download_data('Registered_car_data_for_norway.csv')

############################################ Netflix data #######################################################
df_netflix = download_data('netflix_subsciption_data.csv')
# df_netflix['Quarter'] = pd.to_datetime(df_netflix['time1'])
df_netflix.rename(columns={"time1": "Quarter"}, inplace=True)

## Google trends data
# df_google = download_data('')


# ## Graph Functions

# In[99]:


def plot1(df1, df2):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df1['date'], y=df1['covid per10k'], name='Covid per 10K Population', fill='tozeroy',
                             line=dict(color='coral')), secondary_y=False)
    fig.add_trace(go.Scatter(x=df2['date'], y=df2['average transit stations(5d)'], name='Mobility Change at Transit Stations',
                             line=dict(color='green')), secondary_y=True)
    fig.add_trace(go.Scatter(x=df2['date'], y=df2['average workplaces(5d)'], name='Mobility Change at Workplaces',
                             line=dict(color='red')), secondary_y=True)
    fig.add_trace(go.Scatter(x=df2['date'], y=df2['average retail and recreation(5d)'], name='Mobility Change at Retail&Recreation',
                             line=dict(color='midnightblue')), secondary_y=True)

#     range1 = int(round(df1['covid per10k'].max(), -1))
#     range2 = int(round(df2['average transit stations(5d)'].min()-5, -1))
    range1 = 80 # set this one up to get fix y value for both y-axis
    range2 = -80 # set this one up to get fix y value for both y-axis
    fig.update_yaxes(title='Covid Cases', range=[range1 * (-1), range1], secondary_y=False)
    fig.update_yaxes(title='Mobility Change (%)', range=[range2, range2 * (-1)], secondary_y=True)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=50),
                      legend=dict(orientation='h', yanchor="bottom", y=-0.35, xanchor="center", x=0.5))
    
    fig.add_annotation(dict(font=dict(color="black",size=10),align='left',xref='paper',
                            yref='paper',x=0.95, y = -0.15, showarrow=False,
                            text='Source: Google Mobility, Folkehelseinstituttet'))
    return fig

def plot2():
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df_ecommerce.years, y=df_ecommerce['Posten Yearly Parcels(k)'], name='Total parcels delivered',
                             marker_color='indianred', opacity=0.2), secondary_y=True)

    fig.add_trace(go.Scatter(x=df_ecommerce.years, y=df_ecommerce['percentage of transaction'], name='% of internet payment transactions', mode='lines',
                             line=dict(color='MediumPurple')), secondary_y=False)
    fig.add_trace(go.Scatter(x=df_ecommerce.years, y=df_ecommerce['percentage of value'], name='% of internet payment values',mode='lines',
                              line=dict(color=charcoal_dark)), secondary_y=False)
    
    fig.update_yaxes(title='Internet payment percentage',  secondary_y=False)
    fig.update_yaxes(title='Parcels delivered (in thousand)', secondary_y=True)
    fig.update_xaxes(tickangle=0, )
    fig.update_layout(margin=dict(l=0, r=0, b=50, t=50),
                      legend=dict(orientation='v', yanchor="bottom", y=0.75, xanchor="center", x=0.4))
    
    fig.add_annotation(dict(font=dict(color="black",size=10),align='left',xref='paper',
                            yref='paper',x=0.9, y = -0.15, showarrow=False,
                            text='Source: Posten Norge, Norges Bank'))

    # annotations = []
    # text_world = (df_netflix.worldwide[16][:-9] + "B")
    # annotations.append(dict(xref='paper', x=0.95, y=df_netflix['worldwide'][16], xanchor='right', yanchor='middle', 
    #                         text=text_world, font=dict(family='Montserrat', size=16),
    #                         showarrow=False))

    # fig2.update_layout(annotations=annotations)

    return fig

def plot3():
    df = df_car.copy()
    hybrid = ['Diesel hybrid, non-chargeable','Diesel hybrid, chargeable','Petrol hybrid, chargeable',
                 'Petrol hybrid, non-chargeable']
    df.replace(hybrid, 'Hybrid', inplace=True)
    df.drop(df_car[df['type of fuel'].isin(['Paraffin','Other fuel','Gas','Hydrogen'])].index, inplace = True)
    df = df.groupby(['year','type of fuel','type of registration'], as_index=False).agg({'car_per_capita':'sum'})
    
    fig = px.bar(df, 'year', 'car_per_capita', text='car_per_capita', color='type of fuel', 
                 facet_row='type of registration', barmode='group', 
                 labels={'car_per_capita':'Total Registration per Capita','type of fuel':''},
                 # title='Yearly Total Car Registration in Norway',
                 color_discrete_sequence=["goldenrod", "green", "blue", "red"])

    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_xaxes(title=None)
    fig.update_yaxes(title=None)
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside', cliponaxis=False)
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide',
                      margin=dict(l=0, r=0, b=100, t=20),
                     legend=dict(orientation='h', yanchor="bottom", y=-0.3, xanchor="center", x=0.5))
    
    fig.add_annotation(dict(font=dict(color="black",size=10),align='left',xref='paper',
                            yref='paper',x=0.9, y = -0.15, showarrow=False,
                            text='Source: Statistik sentralbyrå'))
    
    return fig

def plot4():
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df_netflix['Quarter'], y=df_netflix['worldwide'], name='Worldwide', mode='lines',
                             line=dict(color=red)), secondary_y=False)
    fig.add_trace(go.Bar(x=df_netflix['Quarter'], y=df_netflix['Sweden'], name='Sweden',#mode='lines',
                              marker_color=charcoal_dark), secondary_y=True)
    fig.add_trace(go.Bar(x=df_netflix['Quarter'], y=df_netflix['Norway'], name='Norway',#mode='lines',
                              marker_color=lightblue), secondary_y=True)

    fig.update_yaxes(title='Subscription worldwide',  secondary_y=False)
    fig.update_yaxes(title='Subscription region', range=[1.5*1000000, 2.5*1000000], secondary_y=True)
    fig.update_xaxes(tickangle=45, range=['2018-01-01', '2022-04-01'])
    fig.update_layout(margin=dict(l=0, r=0, b=100, t=50),
                      legend=dict(orientation='v', yanchor="bottom", y=0.75, xanchor="center", x=0.1))
    
    fig.add_annotation(dict(font=dict(color="black",size=10),align='left',xref='paper',
                            yref='paper',x=0.9, y = -0.3, showarrow=False,
                            text='Source: Netflix, Infogram, Comparitech'))

    # annotations = []
    # text_world = (df_netflix.worldwide[16][:-9] + "B")
    # annotations.append(dict(xref='paper', x=0.95, y=df_netflix['worldwide'][16], xanchor='right', yanchor='middle', 
    #                         text=text_world, font=dict(family='Montserrat', size=16),
    #                         showarrow=False))

    # fig2.update_layout(annotations=annotations)

    return fig


def plot5(keyword):
#     stop_words.extend(keyword)
    pytrend = TrendReq()
    pytrend.build_payload(kw_list=keyword, cat = 0, timeframe='today 12-m', geo='NO', gprop='')
    
    rq = pytrend.related_queries()

    df_top=pd.DataFrame()
    df_rising=pd.DataFrame()

    for i in keyword:
        temp_top = rq[i]['top']
        df_top = pd.concat([df_top, temp_top])

        temp_rising = rq[i]['rising']
        df_rising = pd.concat([df_rising, temp_rising])
        
    text = {}
    for k,v in df_top.values:
        text[k] = int(v)

    wc = WordCloud(min_font_size=20, height=1200, width=2400).generate_from_frequencies(text)
    fig = px.imshow(wc)
    fig.update_xaxes(visible=False, title='Keywords')
    fig.update_yaxes(visible=False)
    title = ' '.join(keyword)
    fig.update_layout(title=f'Search keywords: {title}', title_x=0.5)
    
    return fig


# ## Layout

# In[122]:


def title():
    layout = dbc.Card([
        dbc.Row([
            dbc.Col([dbc.CardImg(src="assets/wsp_logo.png", className="img-fluid rounded-start")
            ], className="mr-2"),
            dbc.Col([dbc.CardBody([html.H1("Digital Insights WSP"),
                                 html.P(f'What is the trend in mobility post pandemic', style={'font-family':'Montserrat'})])
            ], className="col-md-10"),
        ],className="g-0 d-flex align-items-center"),
    ],color=lightblue, outline = False, className='mb-3 mt-3', style={'font-family':'Montserrat'})
    return layout

def graph1():
    '''This function for layout for mobility and covid data'''
    return dbc.Card([dbc.CardBody([
        dbc.Row([
            dbc.Col([html.H5("Mobility data vs Covid Cases"),
                     html.P(t1, className="card-text")],width=10),
            dbc.Col(dbc.RadioItems(id='radio', options=[{"label": "Norway", "value": 'Norway'},
                    {"label": "Oslo", "value": 'Oslo'}], value='Norway',
                                  labelCheckedClassName="text-danger",
                                  ))]),
        dbc.Row(dbc.Col(dcc.Graph(id='covid_mob_oslo',figure={},config = {'staticPlot': True}))),])])

def graph2():
    '''This function for layout for e-commerce and parcels'''
    return dbc.Card([dbc.CardBody([
        html.H5("E-Commerce payments and packages delivered"),
        html.P(t2, className="card-text"),
        dcc.Graph(id='ecom',figure={},config = {'staticPlot': True}),])])
    
def graph3():
    '''This function for layout for graph car sales'''
    return dbc.Card([dbc.CardBody([
        html.H5("Car register per capita in Norway"),
        html.P(t3, className="card-text"),
        dcc.Graph(id='car-sales',figure={},config = {'staticPlot': True}),])])

def graph4():
    '''This function for layout for netflix'''
    return dbc.Card([dbc.CardBody([
        html.H5("Streaming service subscriptions"),
        html.P(t4, className="card-text"),
        dcc.Graph(id='netflix',figure={},config = {'staticPlot': True}),])])

def graph5():
    '''This function for layout for google trends'''
    return dbc.Card([dbc.CardBody([
        html.H5("Google Trend Search in Norway"),
        html.P(t5, className="card-text"),
        dcc.Graph(id='google',figure={},config = {'staticPlot': True}),
        
        dcc.Input(id='input', type="search", size='50', placeholder="Search keywords", value='transport,tog,elbil,streaming,posten'),
        dbc.Button("Search", id='button', n_clicks=0, color="dark", outline=True, className='ml-4'),
        dbc.FormText("Input keywords separated by comma and without whitespace between keyword (max 5 item)")#("Input keywords separated by comma and without whitespace (max 5 item)", width='auto'),
            ])
        ]#, style={'height':'75vh'}
                   )

def summary():
    return dbc.Card([
        dbc.CardBody([
            dcc.Markdown(summary_text),
        ])
    ], color=lightblue, inverse=False, outline=True, #style={'height':'75vh'}
    ),

## graph small text
t1 = "How mobility behaviour changes in response to number of covid cases"
t2 = "Online payment and number of parcels delivered by Posten in Norway"
t3 = "New and second-hand imported cars register in Norway"
t4 = "How streaming service subscription changes over time"
t5 = "What is trending since last week related to mobility"

summary_text = '''
### Key Takeaway

---
\n

##### 2020 - 2021: Mobility was influenced by covid pandemic


* Travel with public transport has been dampened, which can relate to the work from home recommendation
* Retail & recreation mobility has been decreased


\n---\n


##### 2021 - 2022: Attitude is shifting toward hybrid mode


* Car sales are increasing since people are more flexible in term of mobility with hybrid attitude.
* Workplaces mobility shows stable trend which means people still tend to work from home


\n---\n



##### 2022 - Onward: How will post-pandemic future look like in term of mobility?


* Parcels delivery has been increased along with a raise of online payment, does People-to-goods gradually reverse to Goods-to-people? 
* Streaming industry's subsription stops spiking, so are population ready to mingle outside? 



\n---\n


'''


# In[123]:


layout = [    
    title(),
    ### First row for mobility and E-commerce graphs  
    dbc.Row([
        dbc.Col(graph1(),xl=6, className = 'mt-3 mb-3'),
        dbc.Col(graph2(),xl=6, className = 'mt-3 mb-3')], className = 'mt-3 mb-3'),
    
    dbc.Row([
        dbc.Col(graph3(),xl=6, className = 'mt-3 mb-3'),
        dbc.Col(graph4(),xl=6, className = 'mt-3 mb-3')], className = 'mt-3 mb-3'),
    
    dbc.Row([
        dbc.Col(graph5(),xl=6, className = 'mt-3 mb-3'),
        dbc.Col(summary(),xl=6, className = 'mt-3 mb-3')], style={'height':'65vh'}, className = 'mt-3 mb-3'),
]


# In[124]:


## App run
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LITERA], title='Digital Insights WSP', meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=0.6 maximum-scale=1"}]) 
server = app.server
app.layout = dbc.Container(layout,fluid=True)

@app.callback(
    Output('netflix','figure'),
    [Input('netflix','figure')],)
def netflix_fig(figure):
    fig = plot4()
    return fig

@app.callback(
    Output('covid_mob_oslo','figure'),
    [Input('radio','value')],)
def covid_mob_fig(value):
    if value == 'Norway':
        df2 = df_norway_mob
    else:
        df2 = df_oslo_mob
        
    fig = plot1(df_oslo, df2)
    return fig

@app.callback(
    Output('ecom','figure'),
    [Input('ecom','figure')],)
def ecom_fig(figure):
    fig = plot2()
    return fig

@app.callback(
    Output('car-sales','figure'),
    [Input('car-sales','figure')],)
def car_fig(figure):
    fig = plot3()
    return fig

@app.callback(
    Output('google','figure'),
    [Input('button','n_clicks')],
    [State('input', "value")],
)
def google_fig(n_clicks,value):
    keyword = value.split(',')
    fig = plot5(keyword)
    return fig

#port = 8051
#app.run_server(mode='jupyterlab', port=port)
# app.run_server(mode='inline')
#print('App is running now on http://127.0.0.1:{}/'.format(port))

if __name__ == '__main__':
    app.run_server()
# In[ ]:




