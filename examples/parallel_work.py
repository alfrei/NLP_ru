import time
from nlp_scripts.utils import parallel


def to_lower(s):
    return s.lower()


def do_parallel(n_jobs):
    print("%d cores time" % n_jobs)
    start = time.time()
    data = ['A'+str(i) for i in range(10**7)]
    print(len(data))
    result = parallel(data, to_lower, n_jobs=n_jobs)
    print(len(result))
    print(result[:10])
    print(time.time() - start)
    return result

if __name__ == "__main__":
    checker = ['a'+str(i) for i in range(10**7)]
    res1 = do_parallel(1)
    res4 = do_parallel(4)
    print(checker == res1, checker == res4)