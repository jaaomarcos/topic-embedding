from gensim.models import Word2Vec
import pickle
import numpy as np


class W2v(object):
	def __init__(self, process):
		self.process = process
		self.w2v = []
	
	def run(self):
		texts = self.process.data['corpus']	
		self.model = Word2Vec(texts, min_count=1)
		for text in texts:
			doc = []
			for word in text:
				doc.append(self.model.wv[word])
			self.w2v.append(np.mean(doc, axis=0))
		pickle.dump(self.w2v, open('data/w2v_'+self.process.dataset+'.ipy', 'wb'), pickle.HIGHEST_PROTOCOL)
		pickle.dump(self.model, open('data/model_w2v_'+self.process.dataset+'.ipy', 'wb'), pickle.HIGHEST_PROTOCOL)
	
	def load(self):
		self.w2v = pickle.load(open('data/w2v_'+self.process.dataset+'.ipy', 'rb'))
		self.model = pickle.load(open('data/model_w2v_'+self.process.dataset+'.ipy', 'rb'))