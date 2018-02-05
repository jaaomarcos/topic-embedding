from Process import Process
from Tfidf import Tfidf
from W2v import W2v
from Lda import Lda

dataset = '20ng'
num_topics = 50
limit_labels = 3
p = 0.7

process = Process(dataset)
process.load()

tfidf = Tfidf(process)
tfidf.load()

w2v = W2v(process)
w2v.load()

lda = Lda(process, tfidf, w2v, num_topics, p, limit_labels)
#lda.topic_tfidf()
#lda.train()
lda.load()
lda.test()




