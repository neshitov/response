import pickle
import pandas as pd
from flask import Flask
from flask import render_template, request, jsonify
from response_model import ResponseModel

with open('small_model.pkl', 'rb') as fp:
     model = pickle.load(fp)

app = Flask(__name__)

@app.route('/')
def show():
    return render_template('main.html', data={})

@app.route('/', methods=['POST'])
def analyze_message():
    message_text = request.form['message']
    message = pd.Series(message_text)
    df = model.predict(message)
    g = {'cats': list(df.columns), 'labels': df.values.squeeze().tolist()}
    return render_template('main.html', data=g)

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
