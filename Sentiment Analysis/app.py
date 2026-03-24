import pandas as pd
from flask import Flask, render_template,request, redirect
from helper import preprocessing, vectorizer, get_prediction


app = Flask(__name__)

data = dict()
reviews = []
positive = 0
negative = 0

@app.route("/")
def index():
    data['reviews'] = reviews
    data['positive'] = positive
    data['negative'] = negative

    return render_template('index.html', data=data)

vocab = pd.read_csv('static/model/vocabulary.txt', header=None)
tokens = vocab[0].tolist()

@app.route("/", methods = ['post'])
def my_post():
    text = request.form['text']
    prediction = get_prediction(text, tokens)

    global positive, negative
    if prediction == 'positive':
      positive += 1
    else:
      negative += 1
    
    reviews.insert(0, text)
    return redirect(request.url)

if __name__ == "__main__":
    app.run()