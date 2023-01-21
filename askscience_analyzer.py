import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

# To collapse across different forms of same root
from nltk.stem import WordNetLemmatizer

# For determining important words to each document
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

# To use R syntax for specifying formula for linear regression model
from statsmodels.formula.api import ols

# Download nltk models for word frequencies, sentiment, stopwords
nltk_models = ["vader_lexicon", "punkt", "wordnet", "averaged_perceptron_tagger"]
nltk.download(nltk_models)
stopwords = nltk.corpus.stopwords.words("english")

# Display option for troubleshooting
pd.set_option('display.max_columns', None)

# Read data into pd dataframe
df = pd.read_csv("askscience_data.csv")

# Instantiate sia and lemmatizer
sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()

# Replace na body or title with empty string for simplicity
df['title'] = df.apply(lambda row: str("" if pd.isna(row.title) else row.title), axis=1)
df['body'] = df.apply(lambda row: str("" if pd.isna(row.body) else row.body), axis=1)

# Cast datetime string to datetime type
df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')

# Extract time and date details and cast as integers
df['year'] = df.apply(lambda row: int(row.datetime.year), axis=1)
df['month'] = df.apply(lambda row: int(row.datetime.month), axis=1)
df['day'] = df.apply(lambda row: int(row.datetime.day), axis=1)
df['hour'] = df.apply(lambda row: int(row.datetime.hour), axis=1)
df['weekday'] = df.apply(lambda row: int(row.datetime.weekday()), axis=1)

# Collapse across capitalization and leading whitespace on tag and author
df['tag'] = df.apply(lambda row: str(row.tag).lower().strip(), axis=1)
df['author'] = df.apply(lambda row: str(row.author).lower().strip(), axis=1)

# Generate a text column that is the concatenation of title and body 
df['text'] = df.apply(lambda row: str(row.title + " " + row.body), axis=1)

# Lemmatize each 'word' in 'text' column and replace text value with lemmatized version of nouns
for idx in range(df.shape[0]):
    word_list = [lemmatizer.lemmatize(x) for x in nltk.word_tokenize(df.at[idx, 'text']) if x.isalpha()]
    tag_list = nltk.pos_tag(word_list)
    df.at[idx, 'text'] = ' '.join([w for w, t in tag_list if w not in stopwords and t in ["NN", "NNP", "NNS", "NNPS"]])

# Create a corpus of lemmatized noun 'texts' for tf-idf analysis
docs = df.text.values.tolist()

cv = CountVectorizer(stop_words='english')
word_count_vector=cv.fit_transform(docs)

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True) 
tfidf_transformer.fit(word_count_vector)
 
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names_out(),columns=["idf_weights"]) 
df_idf.sort_values(by=['idf_weights'])

count_vector=cv.transform(docs) 
tf_idf_vector=tfidf_transformer.transform(count_vector)
feature_names = cv.get_feature_names_out() 

# For each document, find its highest value tf-idf word and assign it to df['top_word_by_tfidf']
df['top_word_by_tfidf'] = ""
for idx in range(df.shape[0]):
    document_vector=tf_idf_vector[idx]
    temp_vec = (pd.DataFrame(document_vector.T.todense(), index=feature_names, columns=["tfidf"]).sort_values(by=["tfidf"],ascending=False))
    df.at[idx, 'top_word_by_tfidf'] = temp_vec.iloc[0].name


# def show_doc(index):
#     print (df.at[index, 'title'])
#     print (df.at[index, 'body'])
#     print (df.at[index, 'text'])
#     document_vector=tf_idf_vector[index]
#     stuff = (pd.DataFrame(document_vector.T.todense(), index=feature_names, columns=["tfidf"]).sort_values(by=["tfidf"],ascending=False))
#     print (stuff)
#     print (stuff.iloc[0])     
#     print (stuff.iloc[0].name)    
#     print (type(stuff.iloc[0]))

# show_doc(index)


# Get sentiment scores for title and body
df['title_sentiment'] = df.apply(lambda row: float(sia.polarity_scores(row.title)['compound']), axis=1)
df['body_sentiment'] = df.apply(lambda row: float(sia.polarity_scores(row.body)['compound']), axis=1)
df['overall_sentiment'] = df.apply(lambda row: float(sia.polarity_scores(row.title + row.body)['compound']), axis=1)

file = df.to_csv("processed_file.csv")

#model = ols('score ~ C(top_word_by_tfidf) + title_sentiment + body_sentiment + C(tag) + hour  + C(year) + month + day  + C(weekday)', data=df)
#model = ols('score ~ upvote_ratio + (overall_sentiment) * C(tag) + C(hour)  + C(year) + C(month) + C(day)  + C(weekday)', data=df)
model = ols('score ~ upvote_ratio + C(year) + C(tag) + C(top_word_by_tfidf)', data=df)
fitted_model = model.fit()
print(fitted_model.summary())


