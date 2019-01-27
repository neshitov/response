from metersapp import app
from flask import Flask, render_template, request
from metersapp.display_on_map import get_meter_coordinates

#app = Flask(__name__)

coords = get_meter_coordinates('walt disney concert hall')

@app.route('/')
def show_meters():
    return render_template('main.html', data=coords)

@app.route('/', methods=['POST'])
def classify_message():
    message = request.form['address']
    '''....... all the work ......'''
    ''' classification = prediction list '''
    return render_template('main.html', data=classification)
