# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import joblib
import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from plotly.graph_objs import Bar, Pie

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

with open('big_model.pkl', 'rb') as fp:
     model = joblib.load(fp)

engine = create_engine(os.path.join('sqlite:///', './data/categorized_messages.db'))

df = pd.read_sql_table('messages', engine)
genre_counts = df.groupby('genre').count()['message']
genre_names = list(genre_counts.index)
cat_columns = list(model.cat_columns)
df_positive = df[cat_columns].apply(np.sum)
df_positive = df_positive/df.shape[0]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server

navbar = dbc.NavbarSimple(
    children=[
    dbc.NavItem(dbc.NavLink("Source", href="https://github.com/neshitov/response")),
    ],
    brand="Disaster Response Project",
    brand_href="#",
    sticky="top",
)

app.layout = html.Div(children=[
    navbar,

    dcc.Textarea(placeholder='Enter a message to classify',
    style={'width': '100%'},
    value='', id='input-field'),

    html.Button('Submit', id='button'),

    html.Div( id = 'list',
    children = []
    ),

    html.H3(children='Dataset visualization:', style={'text-align':'center'}),
    dbc.Container([
        dbc.Row([
            dbc.Col(html.Img(src=app.get_asset_url('white_cloud.png'))),
            dbc.Col(
                dcc.Graph(
                    id='example-graph',
                    figure={
                        'data': [Pie(labels=genre_names, values=genre_counts.tolist())],
                        'layout': {'title': 'Message sources'}
                           }
                         )
                   )
                ])
            ]),

    dcc.Graph(
        id='pos_ratio',
        figure={
            'data': [Bar(x=cat_columns,y=df_positive.values)],
            'layout': {'title': 'Positive ratio per category'}
               }
             )
])

@app.callback(Output('list', 'children'),
            [Input('button', 'n_clicks')],
            [State('input-field', 'value')])

def update_output(n, message_text):
    if len(message_text)>0:
        message = pd.Series(message_text)
        # use model to predict classification for query
        df = model.predict(message)
        colors = {}
        for col in cat_columns:
            if df.loc[0,col]==0:
                colors[col]='gray'
            else:
                colors[col]='green'
        ext_cols = cat_columns + ['' for i in range(-len(cat_columns)%6)]
        twod_cols = np.array(ext_cols).reshape((-1,6))
        n_rows = twod_cols.shape[0]
        cat_table = [dbc.Row([dbc.Col(html.H4(children=twod_cols[i][j], style={'color':colors[twod_cols[i][j]]}))for j in range(6)]) for i in range(n_rows)]
        nice_list = dbc.Container(cat_table)
        return nice_list

    else:
        return []

if __name__ == '__main__':
    app.run_server(debug=True)
