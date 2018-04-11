from pprint import pprint
import re
import codecs
from nltk.collocations import BigramCollocationFinder
from nltk.collocations import TrigramCollocationFinder
from nltk.collocations import BigramAssocMeasures
from nltk.collocations import TrigramAssocMeasures
from nltk.corpus import stopwords

# Гарри Поттер и методы рационального мышления
hpmor = codecs.open("../input/hpmor.htm", "r", "utf-8")
corpus = []
while True:
    l = hpmor.readline()
    if l == '': break
    l = re.sub(r"[^а-яё \t-]", "", l.lower()).strip().split()
    if l: corpus.extend(l)

bigram_measures = BigramAssocMeasures()
trigram_measures = TrigramAssocMeasures()

stop = set(stopwords.words('russian'))
stop.update(['гарри','поттер','профессор'])  # добавим самые популярные слова из текста в стоп-лист
corpus_ = list(filter(lambda x: x not in stop, corpus))

finder = BigramCollocationFinder.from_words(corpus_)
finder3 = TrigramCollocationFinder.from_words(corpus_)

# фильтры по частотам и стоп-слова
finder.apply_freq_filter(5)
finder.apply_word_filter(lambda w: len(w) < 3 or w.lower() in stop)
finder3.apply_freq_filter(5)
finder3.apply_word_filter(lambda w: len(w) < 3 or w.lower() in stop)

# биграммы и триграммы
raw_bigrams = finder.nbest(bigram_measures.raw_freq, 100)
pmi_bigrams = finder.nbest(bigram_measures.pmi, 100)
raw_trigrams = finder3.nbest(trigram_measures.raw_freq, 100)
pmi_trigrams = finder3.nbest(trigram_measures.pmi, 100)


def print_ngram(ngram, slice, title):
    print('-'*50)
    print(title)
    pprint(ngram[:slice])

n = 30
print_ngram(raw_bigrams, n, 'raw 2grams')
print_ngram(pmi_bigrams, n, 'pmi 2grams')
print_ngram(raw_trigrams, n, 'raw 3grams')
print_ngram(pmi_trigrams, n, 'pmi 3grams')
