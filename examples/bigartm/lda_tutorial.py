import artm

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from numpy import array

cv = CountVectorizer(max_features=1000, stop_words='english')
n_wd = array(cv.fit_transform(fetch_20newsgroups().data).todense()).T
vocabulary = cv.get_feature_names()

bv = artm.BatchVectorizer(data_format='bow_n_wd',
                          n_wd=n_wd,  # матрица слова х документы
                          vocabulary=vocabulary)

model = artm.LDA(num_topics=15, dictionary=bv.dictionary)
model.fit_offline(bv, num_collection_passes=20)

# содержимое топиков
for t in model.get_top_tokens(8):
    print(t)

# обучение простой модели LDA из UCI-формата
batch_vectorizer = artm.BatchVectorizer(data_path='.', data_format='bow_uci',
                                        collection_name='kos', target_folder='kos_batches')
# регуляризация каждого топика
beta = [0.001] * 15
lda = artm.LDA(num_topics=15, alpha=0.01, beta=beta, cache_theta=True,
               num_document_passes=5, dictionary=batch_vectorizer.dictionary)
lda.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=10)

# матрица слова х топики
phi = lda.phi_
# матрица документы х топики
theta = lda.get_theta()
# перплексия по эпохам
print(lda.perplexity_value)

# предсказание темы для новых документах на текущей модели
# batch_vectorizer = artm.BatchVectorizer(data_path='kos_batches_test')
# theta_test = lda.transform(batch_vectorizer=test_batch_vectorizer)