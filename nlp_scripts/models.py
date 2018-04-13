import numpy as np
from scipy.sparse.linalg import svds
import networkx
from gensim import corpora, models

def low_rank_svd(matrix, singular_count=2):
    """ SVD for sparse matrix """
    u, s, vt = svds(matrix, k=singular_count)
    return u, s, vt


# summarizers
def lsa_summarizer(feature_matrix, num_sentences, num_topics=2, sv_threshold=0.5):
    """
    саммаризация корпуса документов методом LSA
    feature_matrix - sparse-матрица размерности [документы, токены]
    """
    dt_matrix = feature_matrix.astype(float)
    td_matrix = dt_matrix.transpose()
    td_matrix = td_matrix.multiply(td_matrix > 0)  # set negative values to 0
    u, s, vt = low_rank_svd(td_matrix, singular_count=num_topics)

    # filter s by threshold value
    min_sigma_value = np.max(s) * sv_threshold
    s[s < min_sigma_value] = 0

    # ss=sqrt(s^2 * vt^2)
    salience_scores = np.sqrt(np.dot(np.square(s), np.square(vt)))
    top_sentence_indices = salience_scores.argsort()[-num_sentences:][::-1].tolist()
    top_sentence_indices = set(sorted(top_sentence_indices))

    return top_sentence_indices


def textrank_summarizer(dt_matrix, num_sentences):
    """ summarize text with PageRank algo by document similarity matrix """
    similarity_matrix = (dt_matrix * dt_matrix.T)
    similarity_graph = networkx.from_scipy_sparse_matrix(similarity_matrix)
    scores = networkx.pagerank(similarity_graph)
    ranked_sentences = sorted(((score, index) for index, score in scores.items()),
                              reverse=True)
    top_sentence_indices = [ranked_sentences[index][1]
                            for index in range(num_sentences)]
    top_sentence_indices.sort()
    return top_sentence_indices


# topic modeling
def lsi(dt_matrix, total_topics=2):
    """ latent semantic indexing """
    # извлекаем веса из сингулярного разложения матрицы
    td_matrix = dt_matrix.transpose()
    td_matrix = td_matrix.multiply(td_matrix > 0)
    u, s, vt = low_rank_svd(td_matrix, singular_count=total_topics)
    weights = u.transpose() * s.reshape(-1, 1)

    return weights


def train_lda_gensim(norm_tokenized_corpus, total_topics=2, **kwargs):
    """ gensim one threaded LDA wrapper """
    dictionary = corpora.Dictionary(norm_tokenized_corpus)
    mapped_corpus = [dictionary.doc2bow(text)
                     for text in norm_tokenized_corpus]
    tfidf = models.TfidfModel(mapped_corpus)
    corpus_tfidf = tfidf[mapped_corpus]
    lda = models.LdaModel(corpus_tfidf,
                          id2word=dictionary,
                          num_topics=total_topics,
                          **kwargs
                         )
    return lda


def topics_from_weights(weights, feature_names):
    """ extract topics from weights """
    # индексы наибольших абсолютных значений по каждой теме
    sorted_indices = np.array([list(row[::-1]) for row in np.argsort(np.abs(weights))])

    sorted_weights = np.array([list(wt[index]) for wt, index in zip(weights, sorted_indices)])
    sorted_terms = np.array([list(feature_names[row]) for row in sorted_indices])

    topics = [np.hstack((terms.reshape(-1, 1), term_weights.reshape(-1, 1)))
              for terms, term_weights
              in zip(sorted_terms, sorted_weights)]

    return topics


def print_topics(topics, total_topics,
                 weight_threshold=0.0001,
                 display_weights=False,
                 num_terms=None):
    """ 
    print topics
    weight_threshold: минимальный вес отображаемого токена
    num_terms: число токенов по каждой теме
    """
    for index in range(total_topics):
        topic = topics[index]
        topic = [(term, float(wt))
                 for term, wt in topic]
        topic = [(word, round(wt, 2))
                 for word, wt in topic
                 if abs(wt) >= weight_threshold]

        if display_weights:
            print('Topic #' + str(index + 1))
            print(topic[:num_terms] if num_terms else topic)
        else:
            print('Topic #' + str(index + 1))
            tw = [term for term, wt in topic]
            print(tw[:num_terms] if num_terms else tw)

