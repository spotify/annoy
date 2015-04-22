from __future__ import print_function
import random, time
from annoy import AnnoyIndex

try:
    xrange
except NameError:
    # Python 3 compat
    xrange = range


def precision(f=40, n=1000):
    t = AnnoyIndex(f)
    for i in xrange(n):
        v = []
        for z in xrange(f):
            v.append(random.gauss(0, 1))
        t.add_item(i, v, 1)

    t.build(2 * f)
    t.save('test.tree')

    limits = [10, 100, 1000, 10000]
    k = 10
    prec_sum = {}
    prec_n = 1000
    time_sum = {}

    for i in xrange(prec_n):
        j = random.randrange(0, n)
        print('finding nbs for', j)
        
        closest = set(t.get_nns_by_item(j, n, [1], j* 100)[:k])
        for limit in limits:
            t0 = time.time()
            toplist = t.get_nns_by_item(j, limit, [1], j* 100)
            T = time.time() - t0
            
            found = len(closest.intersection(toplist))
            hitrate = 1.0 * found / k
            prec_sum[limit] = prec_sum.get(limit, 0.0) + hitrate
            time_sum[limit] = time_sum.get(limit, 0.0) + T

        for limit in limits:
            print('limit: %-9d precision: %6.2f%% avg time: %.6fs'
                  % (limit, 100.0 * prec_sum[limit] / (i + 1),
                     time_sum[limit] / (i + 1)))

if __name__ == '__main__':
    precision()

        
