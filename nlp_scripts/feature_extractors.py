
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def bow_extractor(corpus, **kwargs):
    """
    bag of words extractor wrapper
    """
    # default params
    default_params = dict(min_df=5)
    for k, v in default_params.items():
        if k not in kwargs: kwargs[k] = v

    vectorizer = CountVectorizer(**kwargs)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features


def tfidf_transformer(bow_matrix):
    """
    tfidf transformer wrapper
    """
    transformer = TfidfTransformer()
    tfidf_matrix = transformer.fit_transform(bow_matrix)
    return transformer, tfidf_matrix


def tfidf_extractor(corpus, **kwargs):
    """
    if-idf extractor wrapper
    """
    # default params
    default_params = dict(min_df=5)
    for k, v in default_params.items():
        if k not in kwargs: kwargs[k] = v

    vectorizer = TfidfVectorizer(**kwargs)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features


def word_embeddings_extractor(feature_matrix, feature_names, word_emb, to_weight=True, neg_prefix='не_'):
    """ 
    word embeddings average/weight extractor
    возвращает средневзвешенное/среднее значение векторов слов из матрицы документ х терм
    учитывает отрицания с заданным префиксом домножением вектора на -1
    feature_matrix: csr-matrix (sparse)
    feature_names: iterable
    word_emb: dict
    """
    wvec_size = len(word_emb['я'])
    doc_num = feature_matrix.shape[0]
    feature_names = np.array(feature_names)

    wvec_feature_matrix = np.zeros((doc_num, wvec_size))
    for idx in range(doc_num):

        # tokens
        tokens = feature_names[feature_matrix[idx].indices].tolist()
        token_num = len(tokens)
        # negatives 1,-1
        neg_weights = np.ones((token_num, 1))
        for i, t in enumerate(tokens):
            if t.startswith(neg_prefix):
                neg_weights[i] = -1
                tokens[i] = tokens[i].replace(neg_prefix, '')
        # word vec matrix for current document
        wvec = np.zeros((token_num, wvec_size))
        for i, t in enumerate(tokens):
            if t in word_emb: wvec[i] = word_emb[t]
        wvec = neg_weights * wvec
        # weighting
        if to_weight:
            weights = np.array(feature_matrix[idx, feature_matrix[idx].indices].todense()).reshape(-1, 1)
            wvec = weights * wvec

        # averaging
        wvec_feature_matrix[idx] = wvec.mean(0)

    return wvec_feature_matrix

def tfidf_extractor(corpus, **kwargs):
    """
    if-idf extractor wrapper
    """
    # default params
    default_params = dict(min_df=5)
    for k, v in default_params.items():
        if k not in kwargs: kwargs[k] = v

    vectorizer = TfidfVectorizer(**kwargs)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features


def word_embeddings_extractor(feature_matrix, feature_names, word_emb, to_weight=True, neg_prefix='не_'):
    """
    word embeddings average/weight extractor
    возвращает средневзвешенное/среднее значение векторов слов из матрицы документ х терм
    учитывает отрицания с заданным префиксом домножением вектора на -1
    feature_matrix: csr-matrix (sparse)
    feature_names: iterable
    word_emb: dict
    """
    wvec_size = len(word_emb['я'])
    doc_num = feature_matrix.shape[0]
    feature_names = np.array(feature_names)

    wvec_feature_matrix = np.zeros((doc_num, wvec_size))
    for idx in range(doc_num):

        # tokens
        tokens = feature_names[feature_matrix[idx].indices].tolist()
        token_num = len(tokens)
        # negatives 1,-1
        neg_weights = np.ones((token_num, 1))
        for i, t in enumerate(tokens):
            if t.startswith(neg_prefix):
                neg_weights[i] = -1
                tokens[i] = tokens[i].replace(neg_prefix, '')
        # word vec matrix for current document
        wvec = np.zeros((token_num, wvec_size))
        for i, t in enumerate(tokens):
            if t in word_emb: wvec[i] = word_emb[t]
        wvec = neg_weights * wvec
        # weighting
        if to_weight:
            weights = np.array(feature_matrix[idx, feature_matrix[idx].indices].todense()).reshape(-1, 1)
            wvec = weights * wvec

        # averaging
        wvec_feature_matrix[idx] = wvec.mean(0)

    return wvec_feature_matrix
