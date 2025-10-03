from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# Load model
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load Vectorizer
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    sentiment_label = ""
    
    
    if request.method == "POST":
        review = request.form["review"]
        vector = vectorizer.transform([review])
        prediction = model.predict(vector)[0]

        if prediction == 1:
            sentiment = "Positive"
            sentiment_label = "positive"
            
        else:
            sentiment = "Negative"
            sentiment_label = "negative"
            

    return render_template("index.html", sentiment=sentiment, sentiment_label=sentiment_label)

if __name__ == "__main__":
    app.run(debug=True)