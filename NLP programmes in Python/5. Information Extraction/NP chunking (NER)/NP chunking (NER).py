# NP chunking (NER)
import nltk

f=open("sample.txt")
text=f.read()
sentences = nltk.sent_tokenize(text)
tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]

for sent in tagged_sentences:
    print (nltk.ne_chunk(sent))
