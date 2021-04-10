import numpy as np
a = [2,4,1,3,1,2,3,4]
print(np.unique(a))


indexes = np.unique(a, return_index=True)[1]
print([a[index] for index in sorted(indexes)])