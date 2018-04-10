
import pymorphy2
from tqdm import tqdm
import re
import emoji


def clean_corpus(raw_corpus, save_apostrof=True,
                 remove_emoji=False, remove_hashtags=False, collapse_emoji=False):
    """ очистка текста
        1. Отбор хэштегов
        2. Перевод эмодзи
        3. Очистка от знаков препинания и чисел
        raw_corpus -- итерируемый объект (list, df.Series, ..)
    """

    print("Clean corpus..")
    output_corpus = []

    for d in tqdm(raw_corpus):

        if d is None: d = ''
        d = str(d)
        d = d.lower()

        # удалить ники
        d = re.sub("@\w*", "", d)

        # заменить ссылки на link
        d = re.sub(r"http[\S]+", "http_link", d)

        # хэштеги
        d = d.replace("#", " ht_")

        # emoji
        d = emoji.demojize(d, delimiters=(" emoji_", " "))

        d = d.replace(":)", " emoji_pos_smile ")
        d = d.replace(":D", " emoji_pos_smile ")
        d = re.sub("\){2,}", " emoji_pos_smile ", d)

        d = d.replace("D(", " emoji_neg_smile ")
        d = d.replace("D:", " emoji_neg_smile ")
        d = d.replace(":|", " emoji_neg_smile ")
        d = re.sub("\({2,}", " emoji_neg_smile ", d)

        # apostrof
        if save_apostrof:
            d = re.sub(r"(([^\w?!'-])+)", " ", d)  # удалить все символы кроме букв, чисел, пробелов и знаков !?'-
            # удалить все апострофы, кроме тех, что внутри букв: д'артаньян, l'oreal и тп:
            d = re.sub("(^|.)[\']+([^\w]|$)", r"\1\2", d)
            d = re.sub("(^|[^\w])[\']+(.|$)", r"\1\2", d)
        else:
            d = re.sub(r"(([^\w?!-])+)", " ", d)  # удалить все символы кроме букв, чисел, пробелов и знаков !?-

        # удаляем пунктуацию кроме !?_-
        d = re.sub(r"[^\w\s!?_-]", r"", d)

        # удаляем числа и слова, содержащие цифры
        d = re.sub(r"([\w]*\d+[\w]*)", r"", d)

        # удаляем идущие подряд -!?_
        d = re.sub(r"([!?_-])+", r"\1", d)

        # удаляем отдельно стоящие _-
        d = re.sub(r"(^|\s)[-_](\s|$)", r"\1\2", d)

        # отделяем ! и ? пробелами
        d = re.sub(r"([?!])", r" \1 ", d)

        # удаляем -_ в начале и конце слов
        d = re.sub(r"(\s)[_-](\w)", r"\1\2", d)
        d = re.sub(r"(\w)[_-](\s)", r"\1\2", d)

        # удаление хэштегов/эмодзи
        if remove_hashtags: d = remove_word(d, "ht_")
        if remove_emoji: d = remove_word(d, "emoji_")
        if collapse_emoji: d = remove_repeated_emojies(d)

        d = collapse_space_symbols(d)
        output_corpus.append(d)

    return output_corpus


def collapse_space_symbols(s):
    return re.sub(r"( )+", r" ", s).strip()


def remove_repeated_emojies(s):
    regex = re.compile("(emoji_[^ ]+( |$)){2,}")
    s = regex.sub(r"\1", s)
    s = collapse_space_symbols(s)
    return s


def remove_word(s, text):
    regex = re.compile(text + "[^\s]" + "*")
    s = regex.sub(" ", s)
    s = collapse_space_symbols(s)
    return s


def tokenize(tokenizer, text):
    tokens = []
    for s in text:
        tokens += tokenizer(s)
    return tokens


class TokenNormalizer():
    """
    приведение токенов к нормальному виду
    """

    morph = pymorphy2.MorphAnalyzer()  # морфологический анализатор (со встроенным словарем)
    ru_alphabet = set('абвгдеёжзийклмнопрстуфхцчшщъыьэюя')
    en_alphabet = set('abcdefghijklmnopqrstuvwxyz')
    token_punct = ["!", "?"]

    def __init__(self, custom_dict, stopwords, negative_words, word_counts, data_counts):
        """ 
        custom_dict: внешний словарь    
        word_counts: словарь частот слов из 1грамм НКРЯ 
                     http://www.ruscorpora.ru/corpora-freq.html
        data_counts: словарь частот из датасета
        stopwords -- список стопслов
        negative_words -- список отрицательных частиц
        """
        self.custom_dict = custom_dict
        self.stopwords = stopwords
        self.negative_words = negative_words
        self.word_counts = word_counts
        self.data_counts = data_counts

    @classmethod
    def detect_language(cls, token, mode="bool"):
        """ Определение языка - кириллица/латиница 
            Возвращает True, если кириллица (русский язык)
            Если mode="int", возвращает число кириллических и латинских символов в слове
        """
        if token.startswith("emoji_") or token.startswith("ht_"): return None
        sum_cyr, sum_lat = 0, 0
        for c in token:
            if c in cls.ru_alphabet: sum_cyr += 1
            if c in cls.en_alphabet: sum_lat += 1
        if mode == "bool":
            return sum_cyr > sum_lat
        else:
            return sum_cyr, sum_lat

    def get_lemma(self, token):
        if self.morph.word_is_known(token):
            p = self.morph.parse(token)[0]
            token = p.normal_form
        return token

    def token_is_known(self, token):
        return self.morph.word_is_known(token) or token in self.custom_dict or token in self.token_punct

    def remove_repeated_letters(self, token):

        """ Заменяем повторяющикся символы в словах, спассибо=>спасибо """

        new_word = token
        if not self.token_is_known(token):

            repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
            match_substitution = r'\1\2\3'

            old_word = token
            new_word = repeat_pattern.sub(match_substitution, old_word)

            while old_word != new_word:
                # проверяем полученное слово на наличие в словаре
                if self.token_is_known(new_word): return new_word
                old_word = new_word
                new_word = repeat_pattern.sub(match_substitution, old_word)

            # если не нашли слово в словаре - удаляем все повторения и ищем в словаре 
            repeat_pattern = re.compile(r'(\w)(\1{1,})')
            match_substitution = r'\1'
            old_word = token
            new_word = repeat_pattern.sub(match_substitution, old_word)
            if self.token_is_known(new_word): return new_word

            # если не нашли слово в словаре - удаляем все символы, повторяющиеся трижды
            repeat_pattern = re.compile(r'(\w)(\1)(\1{1,})')
            match_substitution = r'\1\2'
            new_word = repeat_pattern.sub(match_substitution, token)

        return new_word

    def remove_unknown_short_tokens(self, token, min_len=4):
        if not self.token_is_known(token) and len(token) < min_len:
            return ""
        return token

    def correct_single_latin(self, t):
        corr_word = t
        freq = -1
        for i in range(len(t)):
            if t[i] in self.en_alphabet:
                for j in self.ru_alphabet:
                    new_word = t[:i] + j + t[i + 1:]
                    if self.token_is_known(new_word):
                        if new_word in self.word_counts:
                            cur_freq = self.word_counts[new_word]
                        else:
                            cur_freq = 0
                        if cur_freq > freq:
                            freq = cur_freq
                            corr_word = new_word
        return [corr_word]

    @staticmethod
    def splits(word):
        return [(word[:i], word[i:])
                for i in range(len(word) + 1)]

    @staticmethod
    def find_max_index(variants, counts):
        max_idx = -1
        max_count = -1
        for i, v in enumerate(variants):
            if v in counts:
                count = counts[v]
                if count > max_count:
                    max_count = count
                    max_idx = i
        return max_idx

    def deep_correction(self, t):
        if self.detect_language(t):
            # все подходящие варианты
            pairs = self.splits(t)
            deletes = [a + b[1:] for (a, b) in pairs if b]
            transposes = [a + b[1] + b[0] + b[2:] for (a, b) in pairs if len(b) > 1]
            replaces = [a + c + b[1:] for (a, b) in pairs for c in self.ru_alphabet if b]
            inserts = [a + c + b for (a, b) in pairs for c in self.ru_alphabet]
            variants = list(set([v for v in deletes + transposes + replaces + inserts
                                 if self.token_is_known(v)]))

            # наиболее частый вариант в датасете(приоритет)
            max_idx = self.find_max_index(variants, self.data_counts)
            if max_idx > -1: return variants[max_idx]

            # наиболее частый вариант в частотном словаре
            max_idx = self.find_max_index(variants, self.word_counts)
            if max_idx > -1: return variants[max_idx]
        return t

    def correct(self, t, chars):
        """ исправление ошибок """
        if chars is not None and not self.token_is_known(t):
            if chars[1] == 1:
                # замена 1 латинского символа на русские буквы и проверка по словарю
                return self.correct_single_latin(t)
            elif chars[1] == 0:
                # подбор наиболее вероятного исправления ошибки
                t_list = re.sub(r"[-']", " ", t).split()
                t_out = ['' for t in t_list]
                for i, t in enumerate(t_list):
                    if self.token_is_known(t):
                        t_out[i] = t
                    else:
                        t_out[i] = self.deep_correction(t)
                return t_out
        return [t]

    def concat_negative(self, tokens, mode="simple"):

        if mode == "simple":
            output = []
            i = 0
            while i < len(tokens):
                if tokens[i] in self.negative_words and i < len(tokens) - 1:
                    output.append('_'.join(tokens[i:i + 2]))
                    i += 2
                else:
                    output.append(tokens[i])
                    i += 1
            return output

        if mode == "pos":
            pass
        return tokens

    def normalize_tokens(self, documents, negative=None, pos=True):
        """ Нормализация токенов в документах 
            negative -- метод сцепления отрицательных частиц с последующими словами;
                        simple - отрицательная частица из списка negwords сцепляется с последующим словом
                        pos - `интеллектуальное` сцепление на основе частей речи
                        None - не делаем сцепление
            pos -- определение частей речи, обязательно True при negative=`pos`
            spell_check -- проверка правописания для несловарных русских слов 
                           и подбор оптимального варианта исправления по word embeddings
                           если близких слов из контекста нет, то оставляем слово таким же
        """
        if negative == "pos":
            assert pos is True, "negative=pos требует параметр pos=True"

        output = []
        unknown = []
        corrections = []
        for d in tqdm(documents):
            tokens = d.split()
            corr_tokens = []

            # коррекция ошибок 
            for t in tokens:

                t = self.remove_repeated_letters(t)
                t = self.remove_unknown_short_tokens(t)
                t_list = self.correct(t, self.detect_language(t, "int"))

                if t and t_list and t != t_list[0]: corrections.append((t, t_list))
                corr_tokens += t_list

            # удаление стоп-слов 
            tokens = [t for t in corr_tokens if t not in self.stopwords]

            # POS-tagging 
            # if pos==True:
            #    pass

            # лемматизация 
            tokens = [self.get_lemma(t) for t in tokens]

            # добавляем все несловарные слова в unknown
            [unknown.append(t) for t in tokens if not self.token_is_known(t) and self.detect_language(t)]

            # сцепляем отрицания
            if not negative is None: tokens = [self.get_lemma(t) for t in tokens]

            output.append(' '.join(tokens))
        return output, unknown, corrections
