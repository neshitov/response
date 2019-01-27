# -*- coding: utf-8 -*-
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output, State
import joblib
import pandas as pd
from flask import Flask
from flask import render_template, request, jsonify
from sqlalchemy import create_engine
import os
import sys
import json
import plotly
from plotly.graph_objs import Bar, Pie

with open('small_model.pkl', 'rb') as fp:
     model = joblib.load(fp)


engine = create_engine(os.path.join('sqlite:///', './data/categorized_messages.db'))
df = pd.read_sql_table('messages', engine)
cat_columns = list(model.cat_columns)
message_text = input('message: ')
message = pd.Series(message_text)
df = model.predict(message)
print(df)

sys.exit()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),
    dcc.Input(placeholder='Enter a value...',
    type='text',
    value='', id='input-field'
                      ),

    html.Div( id = 'list',
    children = [html.H4(children=x) for x in cat_columns]
    ),

    dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
            ],
            'layout': {
                'title': 'Dash Data Visualization'
            }
        }
    )
])


@app.callback(Output('list', 'children'),
              [Input('input-field', 'n_submit')],
              [State('input-field', 'value')])
def update_output(n, message_text):
    message = pd.Series(message_text)
    # use model to predict classification for query
    df = model.predict(message)
    colors = {}
    for col in cat_columns:
        if df.loc[0,col]==0:
            colors[col]='gray'
        else:
            colors[col]='green'
    return [html.H4(children=x, style={'color':colors[x]}) for x in cat_columns]

if __name__ == '__main__':
    app.run_server(debug=True)
