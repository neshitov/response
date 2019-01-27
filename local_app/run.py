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

print(df.columns)


sys.exit()
app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals

    labels = genre_names
    values = genre_counts.tolist()

    graphs = [
        {
            'data': [
                Pie(labels=labels, values=values)

            ],

            'layout': {
                'title': 'Message sources'

            }
        }
    ]


        # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


@app.route('/go')
def go():
    # save user input in query
    message_text = request.args.get('query', '')
    message = pd.Series(message_text)
    # use model to predict classification for query
    df = model.predict(message)
    labels = df.values[0]
    classification_results = dict(zip(df.columns, labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=message_text,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
'''
message_text = input('Message: ')
message = pd.Series(message_text)
df = model.predict(message)
print(df)
g = {'cats': list(df.columns), 'labels': df.values.squeeze().tolist()}
print(g)
'''
