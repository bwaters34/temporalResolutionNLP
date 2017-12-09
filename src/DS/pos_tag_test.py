from nltk import pos_tag, word_tokenize, sent_tokenize
from os import listdir
import codecs
from collections import defaultdict

# Some characters whose lines we will throw out
delimitable = ['\t', '|', '[', ']', '+']

def pos_tag_tokenizer(doc):
	tags_dict = defaultdict(float)

	# Format the document
	doc = doc.replace('\n', ' ')
	doc = doc.replace('\r', ' ')
	doc = doc.replace('  ', ' ')

	# Get the sentences
	sents = sent_tokenize(doc)

	# For all good sentences, pos-tag them
	for f in sents:
		if not any([(x in f) for x in delimitable]):
			tokens = word_tokenize(f)
			tags = pos_tag(tokens)
			for tag in tags:
				tags_dict[tag] +=1

	# done!
	return tags_dict

with codecs.open('../../Gutenberg Dataset/Yearly/Train/1904-48778.txt', 'r', encoding='utf-8') as f:
	# Get the document text
	doc = f.read()

temp = pos_tag_tokenizer(doc)
print len(temp)