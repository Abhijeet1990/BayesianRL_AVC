s = 1.0

min_v = 0.9
max_v = 1.1

range =(max_v-min_v)*10

import numpy as np
x = [0.9,0.91,0.96,0.99,1.01,1.02,1.1]
y = np.linspace(0.91,1.1,19)
print(np.digitize(x[2],y))