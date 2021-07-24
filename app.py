from flask import Flask, render_template, request
import tensorflow
from tensorflow import keras
from keras.preprocessing import text, sequence
from keras.models import Model, load_model

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/",methods=['POST'])
def detect_article():
    detect = None

    max_features = 10000
    maxlen = 300
    
    if request.method == 'POST':
        sample_text = [x for x in request.form.values()][0]

        tokenized_text = tokenizer.texts_to_sequences(sample_text)
        
        padded_text = sequence.pad_sequences(tokenized_text, maxlen=maxlen)

        predictions = model.predict(padded_text)
        index = int(predictions[0])
        
        if index == 0:
            detect = 'This article is fake news'
        else:
            detect = 'This article is not a fake news'

        return render_template('index.html', prediction_text=detect, article=sample_text)

if __name__ == "__main__":
    model = load_model('model.h5')

    max_features = 10000
    maxlen = 300
    tokenizer = text.Tokenizer(num_words=max_features)
    
    app.run()
