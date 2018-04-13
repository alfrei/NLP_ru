# пример абсолютного импорта модуля из других папок
# absolute import example

import sys
sys.path.append('C:/dev/PyCharm/projects/NLP_ru/')  # absolute path in system

from nlp_scripts.models import *
print(low_rank_svd.__doc__)
