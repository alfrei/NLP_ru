
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


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
