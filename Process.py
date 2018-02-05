import string
import nltk
import unicodedata
import pickle
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
from nltk.stem.porter import PorterStemmer


class Process(object):
	def __init__(self, dataset):
		self.data = {}
		self.dataset = dataset

	def tokenize_stopwords_stemmer(self, text):
		stemmer = PorterStemmer()
		no_punctuation = text.lower().translate(None, string.punctuation)
		no_number = no_punctuation.translate(None,'0123456789')
		tokens = nltk.word_tokenize(no_number)
		bag_of_words = [stemmer.stem(w) for w in tokens if not w in stopwords.words('english')]
		return bag_of_words

	def run(self):
		ng20 = fetch_20newsgroups(subset='all',remove=('headers', 'footers', 'quotes'))
		data = {}
		corpus = []
		targets = []
		len_documents = len(ng20.data)
		for i in range(len_documents):
			if len(ng20.data[i]) > 20:
				bag_of_words = self.tokenize_stopwords_stemmer(unicodedata.normalize('NFKD', ng20.data[i]).encode('ascii','ignore'))
				corpus.append(bag_of_words)
				targets.append(ng20.target[i])

		data = {'corpus':corpus, 'targets':targets}
		pickle.dump(data, open('data/data_'+self.dataset+'.ipy', 'wb'), pickle.HIGHEST_PROTOCOL)		

	def load(self):
		self.data = pickle.load(open('data/data_'+self.dataset+'.ipy', 'rb'))