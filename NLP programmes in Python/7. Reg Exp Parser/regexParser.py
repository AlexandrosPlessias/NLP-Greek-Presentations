import nltk

from nltk.chunk.regexp import *

from nltk.tree import Tree
from nltk.draw.tree import TreeView


# Anything that has a starting
# determiner followed by an adjective
# and then a noun is mostly a noun phrase

reg_parser = RegexpParser('''
NP: {<DT>? <JJ>* <NN>*} # NP
P: {<IN>} # Preposition
V: {<V.*>} # Verb
PP: {<P> <NP>} # PP -> P NP
VP: {<V> <NP|PP>*} # VP -> V (NP|PP)*
''')

test_sent = "Mr. Obama played a big role in the Health insurance bill"
test_sent_pos = nltk.pos_tag(nltk.word_tokenize(test_sent))
paresed_out = reg_parser.parse(test_sent_pos)

# Print parsed sentence.
print(paresed_out)

# Create tree.
t = Tree.fromstring(str(paresed_out))

# Create tree frame.
TreeView(t)._cframe.print_to_file('output.ps')
