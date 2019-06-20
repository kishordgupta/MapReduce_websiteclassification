import flask
from flask import Flask
from flask import request
import random
from classifier import mlib_classifier
app = Flask(__name__)


@app.route('/')
def hello():
    return "Hello World!"

@app.route('/NN')
def NN():
    ratio = float(request.args.get('ratio'))
    ratio = (ratio*.1) + .2
    j = int(request.args.get('j'))
    cf = mlib_classifier()
    score = cf.GetFScore(j,1 - ratio)
    response = flask.jsonify({'fscore': float(score)*100})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == '__main__':
    app.run()