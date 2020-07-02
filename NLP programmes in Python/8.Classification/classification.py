import re
import csv
import string
import numpy as np

from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


def ModelBuilding(sms_data,sms_labels):
    """
    This is an example pipeline to building a text classifier.
    1. sampling
    2. TfidfVectorizer conversion
    3. building a naive_bayes model
    4. print the accuracy and other metrics
    5. print most relevant features
    """

    # sampling steps
    train_set_size = int(round(len(sms_data)*0.70))
    # i chose this threshold for 70:30 train and test split.
    print('The training set size for this classifier is ' + str(train_set_size) + '\n')
    x_train = np.array([''.join(el) for el in sms_data[0:train_set_size]])
    y_train = np.array([el for el in sms_labels[0:train_set_size]])

    x_test = np.array([''.join(el) for el in sms_data[train_set_size+1:len(sms_data)]])
    y_test = np.array([el for el in sms_labels[train_set_size+1:len(sms_labels)]])

    # We are building a TFIDF vectorizer here.
    vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2),  stop_words='english',  strip_accents='unicode',  norm='l2')
    X_train = vectorizer.fit_transform(x_train)
    X_test = vectorizer.transform(x_test)

    # Naive Bayes.
    clf = MultinomialNB().fit(X_train, y_train)
    y_nb_predicted = clf.predict(X_test)
    print(y_nb_predicted)
    print(' \nConfusion_matrix:')
    cm = confusion_matrix(y_test, y_nb_predicted)
    print(cm)
    print('\nHere is the classification report:')
    print(classification_report(y_test, y_nb_predicted))

    # print the top features
    coefs = clf.coef_
    coefs_with_fns = sorted(zip(coefs[0], vectorizer.get_feature_names()))
    n = 10
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print('\t%.4f\t%-15s\t\t%.4f\t%-15s' % (coef_1, fn_1, coef_2, fn_2))


def PreProcessing(text):

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

def main():
    smsdata = open('corpus\SMSSpamCollection.txt', encoding="utf8") # check the structure of this file!
    sms_data = []
    sms_labels = []
    csv_reader = csv.reader(smsdata,delimiter='\t')
    for line in csv_reader:

        sms_text = PreProcessing(line[1])

        if ( sms_text != None):
            # adding the sms_id
            sms_labels.append( line[0])
            # adding the cleaned text We are calling preprocessing method
            sms_data.append(sms_text)

    smsdata.close()

    # we are calling the model building function here
    ModelBuilding(sms_data,sms_labels)


if __name__ == '__main__':
    main()
