from nltk import pos_tag, word_tokenize
from parser import parse

text = word_tokenize('To be, or not to be: that is the question')
tags = pos_tag(text)
print tags
# print parse([x[1] for x in tags], verbose=True)