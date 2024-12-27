from flask import Flask, render_template, request
import pickle

spam_model = pickle.load(open("finalized_model.sav", 'rb'))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", 'rb'))

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ''
    if request.method == 'POST':
        text = request.form['spam_text']
        
        text_vectorized = tfidf.transform([text])
        is_spam = spam_model.predict(text_vectorized)
        
        if is_spam == 1:
            result = "Result: Text is Spam"
        else:
            result = "Result: Text is Not Spam"
    
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
