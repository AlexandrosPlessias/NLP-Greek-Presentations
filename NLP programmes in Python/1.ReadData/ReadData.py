import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import csv

#Tokenization, stopwords removal, remove words(with size 1), lower capitalization &  lemmatization.
def preprocessing(text):
    # Tokenize into words.
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]

    # Remove stopwords.
    stop = stopwords.words('english')
    tokens = [token for token in tokens if token not in stop]

    # Remove words less than three letters.
    tokens = [word for word in tokens if len(word) >= 3]

    # Lower capitalization.
    tokens = [word.lower() for word in tokens]

    # Lemmatize.
    lmtzr = WordNetLemmatizer()
    tokens = [lmtzr.lemmatize(word) for word in tokens]
    preprocessed_text= ' '.join(tokens)

    return preprocessed_text

# Open sms corpus.
smsfile = open('corpus/SMSSpamCollection.txt', encoding="utf8") # Check the structure of this file!
sms_data = []
sms_labels = []

# CSV Reader LABEL & DATA  are separated by TAB.
csv_reader = csv.reader(smsfile,delimiter='\t')
# Store labels and data.

for line in csv_reader:
    # Adding the sms_id
    sms_labels.append(line[0])
    # Adding the cleaned text, We are calling pre-processing method.
    sms_data.append(preprocessing(line[1]))

smsfile.close()

# Write processed files.
import codecs
taggedfile = codecs.open('SMSSpamCollection_PROCESSED.txt', 'w', 'utf-8')
for i in range(0,len(sms_labels)):
    taggedfile.write(sms_labels[i]+"\t"+sms_data[i]+"\n")
taggedfile.close()


