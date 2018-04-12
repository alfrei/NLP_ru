from gensim.summarization import summarize
from scripts.feature_extractors import tfidf_extractor
from scripts.models import lsa_summarizer
import re

text = """
Elephants are large mammals of the family Elephantidae
and the order Proboscidea. Two species are traditionally recognised,
the African elephant and the Asian elephant. Elephants are scattered
throughout sub-Saharan Africa, South Asia, and Southeast Asia. Male
African elephants are the largest extant terrestrial animals. All
elephants have a long trunk used for many purposes,
particularly breathing, lifting water and grasping objects. Their
incisors grow into tusks, which can serve as weapons and as tools
for moving objects and digging. Elephants' large ear flaps help
to control their body temperature. Their pillar-like legs can
carry their great weight. African elephants have larger ears
and concave backs while Asian elephants have smaller ears
and convex or level backs.
"""

# lsa example
documents = list(filter(None, [d.strip() for d in
                               re.sub(r"[\n,']", ' ', text).split('.')]))
_, feature_matrix = tfidf_extractor(documents, min_df=1)
lsa_idx = lsa_summarizer(feature_matrix, num_topics=2, num_sentences=4)

for i, d in enumerate(documents):
    if i in lsa_idx: print(d)

# standard gensim textrank summarizer
print('-'*60)
[print(d) for d in summarize(re.sub(r"[\n]", ' ', text), split=True, ratio=0.5)]

# TODO: add textrank summarization
print('-'*60)
