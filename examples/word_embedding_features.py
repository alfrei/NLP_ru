from pprint import pprint
import numpy as np
from tqdm import tqdm
import re
from nlp_scripts.feature_extractors import tfidf_extractor, word_embeddings_extractor

# common crawl + wiki fasttext
# https://s3-us-west-1.amazonaws.com/fasttext-vectors/word-vectors-v2/cc.ru.300.vec.gz
# https://github.com/facebookresearch/fastText/blob/master/docs/crawl-vectors.md
#
# load word embeddings

n_words = 200000
word_vec = {}
with open("../dict/word_embeddings/cc.ru.300.vec", "r", encoding='utf-8') as f:
    print(f.readline())
    for i in tqdm(range(n_words)):
        l = f.readline().split()
        if l == '': break
        k, v = l[0].lower(), list(map(float, l[1:]))
        word_vec[k] = v

# text
brods = """
Мне жаль, что тебя не застал летний ливень
В июльскую ночь, на балтийском заливе
Не видела ты волшебства этих линий -

Волна, до которой приятно коснуться руками,
Песок, на котором рассыпаны камни
Пейзаж, не меняющийся здесь веками.

Мне жаль, что мы снова не сядем на поезд,
Который пройдет часовой этот пояс
По стрелке которую тянет на полюс.

Что не отразит в том купе вечеринку,
Окно, где все время меняют картинку,
И мы не проснемся на утро в обнимку.

Поздно ночью
Через все запятые дошел, наконец, до точки
Адрес, почта -
Не волнуйся, я не посвящу тебе больше ни строчки

Тихо, звуки
По ночам до меня долетают редко
Пляшут буквы
Я пишу и не жду никогда ответа

Мысли, рифмы
Свет остался, остался звук, остальное стерлось
Гаснут цифры
Я звонил, чтобы просто услышать голос

Всадник замер
Замер всадник, реке стало тесно в русле
Кромки, грани
Я люблю, не нуждаясь в ответном чувстве...
"""


def clear_and_tokenize(text):
    text = text.lower().strip().split('\n\n')
    tokenized = [re.sub('[^а-яё !?]', ' ', d).strip() for d in text]
    tokenized = [re.sub('[ ]{2,}', ' ', d) for d in tokenized]
    tokenized = [re.sub(r'( не )', ' не_', d) for d in tokenized]
    return tokenized

brods = clear_and_tokenize(brods)
vec, feature_matrix = tfidf_extractor(brods, min_df=1)
feature_names = vec.get_feature_names()
wvec_features = word_embeddings_extractor(feature_matrix, feature_names, word_vec)

corpus = np.array(brods)
print("means")
print(wvec_features.mean(1))
