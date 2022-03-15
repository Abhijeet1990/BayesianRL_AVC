import numpy as np

a = np.array([1, 2, 3])
print(list(a))
c = np.vstack(a)


print(c)
print(np.vstack(c))


arr = np.array([0, 1, 2])
c = np.tile(arr, 2)
print(c)