import nltk
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

print("\n\n--> Naive bayes")
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train, y_train)
y_nb_predicted = clf.predict(X_test)
print("Naive Bayes Prediction:")
print(y_nb_predicted)
print('\nConfusion_matrix \n ')
cm = confusion_matrix(y_test, y_nb_predicted)
print(cm)
print('\nHere is the classification report:')
print(classification_report(y_test, y_nb_predicted))
# Print the top features
coefs = clf.coef_ #coefficients relate
intercept = clf.intercept_
feature_names = vectorizer.get_feature_names()
coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
n=10
top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
for (coef_1, fn_1), (coef_2, fn_2) in top:
    print('\t%.4f\t%-15s\t\t%.4f\t%-15s' % (coef_1, fn_1, coef_2, fn_2))

print("\n\n-->Classification And Regression Tree - CART")# Decision trees
from sklearn import tree
clf = tree.DecisionTreeClassifier().fit(X_train.toarray(), y_train)
y_tree_predicted = clf.predict(X_test.toarray())
print("\nDecision Tree Prediction:")
print(y_tree_predicted)
print(' \nHere is the classification report:')
print(classification_report(y_test, y_tree_predicted))
print('\nConfusion_matrix \n ')
cm = confusion_matrix(y_test, y_tree_predicted)
print(cm)

print("\n\n-->Stochastic Gradient Descent - SGD") #mostly used
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
clf=SGDClassifier(alpha=.0001, max_iter=50, tol=0).fit(X_train, y_train)#  Remove n_iter=50 & add max_iter=n_iter, tol=0.
y_pred = clf.predict(X_test)
print('\nHere is the classification report:')
print(classification_report(y_test, y_pred))
print('\nConfusion_matrix \n ')
cm = confusion_matrix(y_test, y_pred)
print(cm)
# Print the top features
coefs = clf.coef_ #coefficients relate
intercept = clf.intercept_
feature_names = vectorizer.get_feature_names()
coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
n=15
top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
print()
for (coef_1, fn_1), (coef_2, fn_2) in top:
    print('\t%.4f\t%-15s\t\t%.4f\t%-15s' % (coef_1, fn_1, coef_2, fn_2))

# Logistic regression
# clf=SGDClassifier(loss='log', alpha=.0001, max_iter=50, tol=0).fit(X_train, y_train)#  Remove n_iter=50 & add max_iter=n_iter, tol=0.

print("\n\n-->Support vector machines - SVM") #state-of-the-art
from sklearn.svm import LinearSVC
svm_classifier = LinearSVC().fit(X_train, y_train)
y_svm_predicted = svm_classifier.predict(X_test)
print('\nHere is the classification report:')
print(classification_report(y_test, y_svm_predicted))
print('\nConfusion_matrix \n ')
cm = confusion_matrix(y_test, y_pred)
print(cm)
# Print the top features
coefs = clf.coef_ #coefficients relate
intercept = clf.intercept_
feature_names = vectorizer.get_feature_names()
coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
n=15
top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
print()
for (coef_1, fn_1), (coef_2, fn_2) in top:
    print('\t%.4f\t%-15s\t\t%.4f\t%-15s' % (coef_1, fn_1, coef_2, fn_2))

print("\n\n-->RandomForestClassifier")
from sklearn.ensemble import RandomForestClassifier
RF_clf = RandomForestClassifier(n_estimators=10)
RF_clf.fit(X_train, y_train) # I ADD THIS
predicted = RF_clf.predict(X_test)
print('\nHere is the classification report:')
print(classification_report(y_test, predicted))
print('\nConfusion_matrix \n ')
cm = confusion_matrix(y_test, y_pred)
print(cm)
