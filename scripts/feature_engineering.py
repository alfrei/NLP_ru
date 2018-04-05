import re


# pre feature engineering

def count_words(d):
    return len(d.split())


def count_ht(d):
    regex = re.compile("ht_")
    return len(regex.findall(d))


def count_adwords(d):
    regex = re.compile(r"( заказ )|( акция )|( проф )")
    return len(regex.findall(d))


def count_phone(d):
    regex = re.compile(r"( номер тел )|( моб)|( тел )|([\d]+[-]+[\d]+[-]+[\d]+)")
    return len(regex.findall(d))


def count_phone_emoji(d):
    regex = re.compile(r"emoji_[^\s]+")
    return len([e for e in regex.findall(d) if 'phone' in e])


def count_money(d):
    regex = re.compile(r"([$€])|( руб )|( р )|( рублей )] ")
    return len(regex.findall(d))


# post feature engineering
def count_len(d):
    l = 0
    for t in d.split():
        if 'emoji_' in t:
            l += 1
        else:
            l += len(t)
    return l


def count_emoji(d):
    regex = re.compile("emoji_")
    return len(regex.findall(d))


def count_unique(d):
    return len(set(d.split()))


def count_unknown(d, normalizer):
    return len([t for t in d.split() if normalizer.token_is_known(t)])
