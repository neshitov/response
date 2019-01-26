from multiprocessing import Pool
import numpy as np
import time

def f(x):
    return x**2

X=list(np.random.randn(10**7))

with Pool(processes=1) as p:
    start = time.time()
    g = p.map(f, X)
    print('finished in',time.time()-start)
