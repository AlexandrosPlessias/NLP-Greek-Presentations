from nltk.parse.generate import generate
from nltk import CFG # toy CFG

toy_grammar = CFG.fromstring("""
S -> NP VP
VP -> V NP 
V -> 'eats' | 'drinks'
NP -> Det N 
Det -> 'a' | 'an' | 'the'
N -> 'president' | 'Obama' | 'apple'| 'coke' 
""")

"""
S indicate the entire sentence
VP is verb phrase the
V is verb
NP is noun phrase (chunk that has noun in it)
Det is determiner used in the sentences
N some example nouns
"""

# Print the Grammar.
print()
print(toy_grammar)

# Generate all the sentences.
print()
for sentence in generate(toy_grammar):
    print(sentence)

# Print the number of all available sentences.
print()
print (len(list(generate(toy_grammar))))

