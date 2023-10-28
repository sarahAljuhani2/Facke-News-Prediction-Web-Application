# Import the necessary libraries
import warnings
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from flask import Flask, render_template, request
# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Read the data
data = pd.read_csv(r"C:\Users\moner\Downloads\datamlproject.csv")

# Ignore warnings
warnings.filterwarnings("ignore")

# Remove duplicates from the DataFrame
data = data.drop_duplicates()

# Combine the text columns
data['combined'] = data['title'] + ' ' + data['text'] + ' ' + data['subject'] + ' ' + data['date']

class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english'))
        self.sid = SentimentIntensityAnalyzer()

    def __call__(self, doc):
        tokens = word_tokenize(doc, preserve_line=True)  # Preserve the entire document as a single sentence
        tokens = [self.wnl.lemmatize(t) for t in tokens]  # Lemmatization
        tokens = [t for t in tokens if t.lower() not in self.stopwords]  # Stop word removal

        # Sentiment analysis
        sent_scores = self.sid.polarity_scores(doc)
        tokens.extend(["sent_{}".format(k) for k, v in sent_scores.items() if v != 0])

        return tokens

class TextPreprocessor:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english'))
        self.sid = SentimentIntensityAnalyzer()
        self.lda = LatentDirichletAllocation(n_components=5, random_state=42)

    def preprocess(self, doc):
        # Tokenization and Lemmatization
        tokens = word_tokenize(doc)
        tokens = [self.wnl.lemmatize(t) for t in tokens]

        # Stop word removal
        tokens = [t for t in tokens if t.lower() not in self.stopwords]

        # Sentiment analysis
        sent_scores = self.sid.polarity_scores(doc)
        tokens.extend(["sent_{}".format(k) for k, v in sent_scores.items() if v != 0])

        return ' '.join(tokens)

# Apply the preprocessing on the combined text
preprocessor = TextPreprocessor()
data['preprocessed'] = data['combined'].apply(preprocessor.preprocess)

# Split the data into features and target
X = data['preprocessed']
y = data['class']

# Initialize the vectorizer and scaler
vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer())
standard_scaler = StandardScaler(with_mean=False)

# Transform the features using the vectorizer and scaler
X_tfidf = vectorizer.fit_transform(X)
X_scaled = standard_scaler.fit_transform(X_tfidf)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.40, random_state=42)

lr_model = LogisticRegression()

# Train and evaluate Logistic Regression model with cross-validation
lr_cv_scores = cross_val_score(lr_model, X_train, y_train, cv=5)
lr_cv_scores_mean = lr_cv_scores.mean()

# Train the Logistic Regression model on the entire training set
lr_model.fit(X_train, y_train)

# Make predictions on the test set
lr_y_pred = lr_model.predict(X_test)

# Calculate the test accuracy for Logistic Regression
lr_test_accuracy = accuracy_score(y_test, lr_y_pred)

# Initialize the Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        user_input_news = request.form['news_input']
        preprocessed_news = preprocessor.preprocess(user_input_news)
        vectorized_news = vectorizer.transform([preprocessed_news])
        scaled_news = standard_scaler.transform(vectorized_news)
        prediction = lr_model.predict(scaled_news)
        prediction_text = "Fake News" if prediction[0] == 1 else "Not Fake News"  
        return render_template('results.html',
                               prediction_text=prediction_text)
    return render_template('input_news.html')

if __name__ == '__main__':
    app.run()