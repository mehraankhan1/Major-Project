from flask import Flask, render_template, url_for, request, redirect, session
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

USER_CREDENTIALS = {'username': 'password'}

# Load English and Hindi models and transformers
english_model = 'englishmodel.pkl'
english_cv = 'englishtranform.pkl'

hindi_model = 'hindimodel.pkl'
hindi_cv = 'hinditranform.pkl'

app = Flask(__name__)

# Load the models and transformers
english_clf = pickle.load(open(english_model, 'rb'))
english_cv = pickle.load(open(english_cv, 'rb'))

hindi_clf = pickle.load(open(hindi_model, 'rb'))
hindi_cv = pickle.load(open(hindi_cv, 'rb'))

analyzer = SentimentIntensityAnalyzer()

@app.route('/')
def home():
    if 'logged_in' in session:
        return render_template('homepage.html')
    else:
        return redirect(url_for('login'))

@app.route('/about')
def about():
    if 'logged_in' in session:
        return render_template('about.html')
    else:
        return redirect(url_for('login'))

@app.route('/predict_english', methods=['GET', 'POST'])
def predict_english():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        message = request.form['message']
        # Perform Sentiment Analysis
        sentiment_score = analyzer.polarity_scores(message)
        # Get the compound score which indicates overall sentiment
        sentiment = 'Positive' if sentiment_score['compound'] >= 0 else 'Negative'
        
        data = [message]
        vect = english_cv.transform(data).toarray()
        my_prediction = english_clf.predict(vect)
        return render_template('english.html', prediction=my_prediction[0], sentiment=sentiment)
    return render_template('english.html', prediction='', sentiment='')

@app.route('/predict_hindi', methods=['GET', 'POST'])
def predict_hindi():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        message = request.form['message']
        # Perform Sentiment Analysis
        sentiment_score = analyzer.polarity_scores(message)
        # Get the compound score which indicates overall sentiment
        sentiment = 'Positive' if sentiment_score['compound'] >= 0 else 'Negative'
        
        data = [message]
        vect = hindi_cv.transform(data).toarray()
        my_prediction = hindi_clf.predict(vect)
        return render_template('hindi.html', prediction=my_prediction[0], sentiment=sentiment)
    return render_template('hindi.html', prediction='', sentiment='')

@app.route('/machine_learning_models')
def machine_learning_models():
    if 'logged_in' in session:
        return render_template('machine_learning_models.html')
    else:
        return redirect(url_for('login'))

@app.route('/developers')
def developers():
    if 'logged_in' in session:
        return render_template('developers.html')
    else:
        return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            session['logged_in'] = True
            return redirect(url_for('home'))
        else:
            return render_template('login.html', message='Invalid username or password.')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.secret_key = 'supersecretkey'  # Required for session management
    app.run(debug=True)
