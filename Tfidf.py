from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np


class Tfidf(object):
	def __init__(self, process):
		self.process = process
		self.tfidf = []
		
	def run(self):
		texts = [" ".join(str(x) for x in j) for j in self.process.data[self.process.dataset]['corpus']] 
		document_term = TfidfVectorizer()
		tfidf = document_term.fit_transform(texts)
		pickle.dump(tfidf, open('data/tfidf_'+self.process.dataset+'.ipy', 'wb'), pickle.HIGHEST_PROTOCOL)	

	def load(self):
		self.tfidf = pickle.load(open('data/tfidf_'+self.process.dataset+'.ipy', 'rb'))