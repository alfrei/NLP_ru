# coding=utf-8

import numpy as np
import pandas as pd

from scipy.sparse import hstack

from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from collections import Counter
import itertools

from scripts.utils import *
from scripts.preprocessing import *
from scripts.feature_engineering import *
from scripts.feature_extractors import *
from scripts.learn import *

# seeds
import random
random.seed(42)
np.random.seed(42)

# data load
print("Load data..")
# XLSX
df = load_excel("../input/LP coded-2.xlsx", 'Sheet1')
raw_corpus = df.Текст.tolist()
target = df.Тональность
# negative words
negative_words = txt_to_set("../dict/negative_words.txt")
# stopwords
stopwords = txt_to_set("../dict/stopwords.txt", "utf-8-sig")
# custom dictionary
custom_dict = txt_to_set("../dict/custom_dict.txt")
# freq global dictionary
word_counts = {}
with open("../dict/1grams.txt", "r", encoding="utf-8") as f:
    wc = f.readline().strip().split()
    while wc:
        word_counts[wc[1].lower()] = int(wc[0])
        wc = f.readline()
        if wc: wc = wc.strip().split()

# clean corpus
corpus = clean_corpus(raw_corpus)

# pre-feature engineering
X_num = pd.DataFrame()
X_num['ht'] = [count_ht(d) for d in corpus]
X_num['adwords'] = [count_adwords(d) for d in corpus]
X_num['phone'] = [count_phone(d) for d in corpus]
X_num['phone_emoji'] = [count_phone_emoji(d) for d in corpus]
X_num['money'] = [count_money(d) for d in corpus]
X_num['words_raw'] = [count_words(d) for d in corpus]
X_num['ht_fr'] = X_num['ht']/X_num['words_raw']
X_num['adwords_fr'] = X_num['adwords']/X_num['words_raw']
X_num['phone_fr'] = X_num['phone']/X_num['words_raw']
X_num['phone_emoji_fr'] = X_num['phone_emoji']/X_num['words_raw']
X_num['money_fr'] = X_num['money']/X_num['words_raw']

# final cleaning
tmp = []
for d in corpus:
    # d = remove_word(d, "ht_")  # remove hashtags
    # d = remove_word(d, "emoji_")  # remove emoji
    d = remove_repeated_emojies(d)
    tmp.append(d)
corpus = tmp
del tmp

# words frequency from dataset
data_counts = Counter()
for d in corpus:
    data_counts.update(d.split())
for w in stopwords:
    if w in data_counts:
        data_counts.pop(w)

# normalization
normalizer = TokenNormalizer(custom_dict, stopwords, negative_words, word_counts, data_counts)
corpus_norm, unknown, corrections = normalizer.normalize_tokens(corpus, negative="simple")

# post-feature engineering
X_num['words'] = np.array([count_words(d) for d in corpus_norm])
X_num['len'] = np.array([count_len(d) for d in corpus_norm])
X_num['emoji'] = np.array([count_emoji(d) for d in corpus_norm])
X_num['unknown'] = np.array([count_unknown(d, normalizer) for d in corpus_norm])
X_num['unique'] = np.array([count_unique(d) for d in corpus_norm])
X_num['emoji_fr'] = X_num['emoji']/X_num['words']
X_num['unknown_fr'] = X_num['unknown']/X_num['words']
X_num['unique_fr'] = X_num['unique']/X_num['words']

# filter empty documents
empty_index = [True if d else False for d in corpus_norm]
X_num = X_num[empty_index]
target = target[empty_index]
corpus_norm = list(itertools.compress(corpus_norm, empty_index))

# vectorization
vectorizer, X_tfidf = tfidf_extractor(corpus_norm, min_df=5, ngram_range=(1,3), max_features=20000)
X = hstack([X_tfidf, X_num])
target_factors = target.factorize()
y = target_factors[0]
class_prior = np.unique(y, return_counts=True)[1]/y.shape[0]

# classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# svm = SGDClassifier(max_iter=2000, tol=1e-4, n_jobs=-1, random_state=42, penalty='l2', alpha=0.01)
# svm too sensitive to alpha hyperparameter, need more data

lr = LogisticRegression(class_weight='balanced', C=0.8)
nb = MultinomialNB(alpha=8, class_prior=class_prior)
rf = RandomForestClassifier(max_depth=25, n_estimators=500, class_weight='balanced', n_jobs=-1, random_state=42, 
                            max_features=40, criterion='entropy', min_impurity_decrease=1e-5)
xgb = XGBClassifier(objective='multi:softprob', seed=42, random_state=42, n_jobs=-1,
                    max_depth=4, learning_rate=0.02, n_estimators=700,
                    subsample=0.8, colsample_bytree=0.2, gamma=2)

models = [lr, xgb, nb, rf]
preds = np.zeros((len(models),X_test.shape[0],class_prior.shape[0]))
for i, model in enumerate(models):
    preds[i, :] = fit_model(X_train,X_test,y_train,y_test,model)

# weighting/blending
weights = np.array([10, 3, 0.1, 5]).reshape(1,4)
blends = np.tensordot(weights, preds, [1,0])[0]


print("-"*60)
print("ensemble")
print(metrics.accuracy_score(y_test, np.apply_along_axis(arr=blends, axis=1, func1d=max_index)))
print(metrics.confusion_matrix(y_test, np.apply_along_axis(arr=blends, axis=1, func1d=max_index)))
