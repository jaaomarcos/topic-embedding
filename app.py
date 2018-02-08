from Process import Process
from Tfidf import Tfidf
from W2v import W2v
from Lda import Lda

dataset = '20ng'
num_topics = 10
limit_labels = 3
p = 0.7

process = Process(dataset)
process.load()

tfidf = Tfidf(process)
tfidf.load()

w2v = W2v(process)
w2v.load()

for i in range(10):
	lda = Lda(process, tfidf, w2v, num_topics, p, limit_labels)
	print int(len(lda.texts_test))
	#lda.topic_tfidf()
	lda.load()
	lda.train()
	lda.test()
	num_topics+=10



