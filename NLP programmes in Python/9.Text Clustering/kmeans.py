import nltk
import re
import csv
import string
import collections
import numpy as np

from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

""""Pre - Processing: tokenization, stopwords removal, remove words(with size 1), lower capitalization &  lemmatization"""
def preprocessing(text):
 # text = text.decode("utf8")

    # remove punctuation
    text = punctuation(text)

    # remove extra spaces
    text = re.sub(' +', ' ', text)

    # tokenize into words
    tokens = text.split(" ")

    # remove number
    tokens = [word for word in tokens if word.isalpha()]

    # remove stopwords
    stop = stopwords.words('english')
    tokens = [token for token in tokens if token not in stop]

    # remove words less than three letters
    tokens = [word for word in tokens if len(word) >= 3]

    # lower capitalization
    tokens = [word.lower() for word in tokens]

    # keep only real words
    tokens = KeepRealWords(tokens)

    # lemmatize
    lmtzr = WordNetLemmatizer()
    tokens = [lmtzr.lemmatize(word) for word in tokens]

    # return only tokens with size over 1
    if len(tokens) > 0:
        preprocessed_text = " ".join(tokens)
        return preprocessed_text

    return None

def KeepRealWords(text):

    wpt = WordPunctTokenizer()
    only_recognized_words = []

    for s in text:
        tokens = wpt.tokenize(s)
        if tokens:  # check if empty string
            for t in tokens:
                if wordnet.synsets(t):
                    only_recognized_words.append(t)  # only keep recognized words

    return only_recognized_words

def punctuation(text):
    translator = str.maketrans(string.punctuation, ' '*len(string.punctuation)) # map punctuation to space
    return (text.translate(translator))


""""Read Data"""
# Open sms corpus.
sms_file = open('SMSSpamCollection.txt', encoding="utf8") # Check the structure of this file!
sms_data = []
sms_labels = []

# CSV Reader LABEL & DATA  are separated by TAB.
csv_reader = csv.reader(sms_file,delimiter='\t')
# Store labels and data.
for line in csv_reader:
    sms_text = preprocessing(line[1])
    if ( sms_text != None):
        # adding the sms_id
        sms_labels.append( line[0])
        # adding the cleaned text We are calling preprocessing method
        sms_data.append(sms_text)

sms_file.close()

"""Sampling steps (70:30)"""
trainset_size = int(round(len(sms_data)*0.70))
# I chose this threshold for 70:30 train and test split.
print('The training set size for this classifier is ' + str(trainset_size) + '\n')
x_train = np.array([''.join(el) for el in sms_data[0:trainset_size]])   # train sms_data (70%).
y_train = np.array([el for el in sms_labels[0:trainset_size]])          # train sms_labels (70%).
x_test = np.array([''.join(el) for el in sms_data[trainset_size+1:len(sms_data)]])  # test sms_data (30%).
y_test = np.array([el for el in sms_labels[trainset_size+1:len(sms_labels)]])       # test sms_labels (30%).

"""We are building a TFIDF vectorizer here"""
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2),  stop_words='english',  strip_accents='unicode',  norm='l2')
X_train = vectorizer.fit_transform(x_train)
X_test = vectorizer.transform(x_test)

"""Text Clustering - K Means"""
from sklearn.cluster import KMeans, MiniBatchKMeans
print('--> Text Clustering - K Means')
true_k = 5
km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
kmini = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1, init_size=1000, batch_size=1000, verbose=False) #verbose=opts.verbose
# we are using the same test,train data in TFIDF form as we did in text classification

km_model = km.fit(X_train)
print("For K-mean clustering ")
clustering = collections.defaultdict(list)
for idx, label in enumerate(km_model.labels_):
        clustering[label].append(idx)
print(clustering)

kmini_model = kmini.fit(X_train)
print("For K-mean Mini batch clustering ")
clustering = collections.defaultdict(list)
for idx, label in enumerate(kmini_model.labels_):
        clustering[label].append(idx)
print(clustering)
