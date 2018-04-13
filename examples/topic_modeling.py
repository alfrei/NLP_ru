import numpy as np
from nlp_scripts.feature_extractors import tfidf_extractor
from nlp_scripts.models import *
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import NMF

corpus = ["The fox jumps over the dog",
"The fox is very clever and quick",
"The dog is slow and lazy",
"The cat is smarter than the fox and the dog",
"Python is an excellent programming language",
"Java and Ruby are other programming languages",
"Python and Java are very popular programming languages",
"Python programs are smaller than Java programs"]

total_topics = 2
vectorizer, dt_matrix = tfidf_extractor(corpus, min_df=1)
feature_names = np.array(vectorizer.get_feature_names())

# LSI
print('LSI')
weights = lsi(dt_matrix, total_topics)
topics = topics_from_weights(weights, feature_names)
print_topics(topics, total_topics, 0.1, num_terms=10)

# LDA
print('-'*100)
print('LDA gensim')

norm_tokenized_corpus = [d.split() for d in corpus]
lda_gensim = train_lda_gensim(norm_tokenized_corpus, total_topics=2,
                              iterations=1000)
topics = [np.array(lda_gensim.show_topic(i)) for i in range(total_topics)]
print_topics(topics, total_topics, 0.0001, num_terms=100)

print('-'*100)
print('LDA sklearn')

lda = LatentDirichletAllocation(n_topics=total_topics,
                                max_iter=1000,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=42)
lda.fit(dt_matrix)
weights = lda.components_
topics = topics_from_weights(weights, feature_names)
print_topics(topics, total_topics, num_terms=10)

# NMF
nmf = NMF(n_components=total_topics, random_state=42,
          alpha=.1, l1_ratio=.5)
nmf.fit(dt_matrix)
weights = nmf.components_
topics = topics_from_weights(weights, feature_names)
print_topics(topics, total_topics, num_terms=10)