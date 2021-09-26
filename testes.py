import numpy as np
from numpy.random import default_rng

cromossome_size = 15

lista = np.arange(25).tolist()
child1 = list(range(5,25,4))
child2 = list(range(6,25,4))
child1.extend(child2)
child1.sort()
print(child1)
