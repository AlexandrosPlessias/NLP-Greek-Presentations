import re, csv, string

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from gensim import corpora, models
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer


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

# CSV Reader LABEL & DATA  are separated by TAB.
csv_reader = csv.reader(sms_file,delimiter='\t')
# Store labels and data.
for line in csv_reader:
    sms_text = preprocessing(line[1])
    if ( sms_text != None):
        # adding the cleaned text We are calling preprocessing method
        sms_data.append(sms_text)

sms_file.close()

# Reading documents of SMS data
documents = [document for document in sms_data]
texts = [[word for word in document.lower().split()] for document in documents]

# Converting the list of documents to a BOW model and then, to a typical TF-IDF corpus
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

# Given the number of topics, the model tries to take all the documents from the corpus to build a LDA model
n_topics = 5
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=n_topics)
lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=n_topics)

# We need to print some top terms related to that topic.
print("LDA Model: ")
for i in range(0, n_topics):
    temp = lda.show_topic(i, 10)
    terms = []
    for term in temp:
        terms.append(term[0])
    print("Top 10 terms for topic #" + str(i) + ": " + ", ".join(terms))

# We need to print some top terms related to that topic.
print("\nLSI Model: ")
for i in range(0, n_topics):
    temp = lsi.show_topic(i, 10)
    terms = []
    for term in temp:
        terms.append(term[0])
    print("Top 10 terms for topic #" + str(i) + ": " + ", ".join(terms))
