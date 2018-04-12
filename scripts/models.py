import numpy as np
from scipy.sparse.linalg import svds


def low_rank_svd(matrix, singular_count=2):
    """ SVD for sparse matrix """
    u, s, vt = svds(matrix, k=singular_count)
    return u, s, vt


def lsa_summarizer(feature_matrix, num_sentences=5, num_topics=2, sv_threshold=0.5):
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
