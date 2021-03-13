import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

# print(type(fig))
# print(type(ax))

# x = np.random.randint(1,10, size=10)
# y = x*2
# plt.plot(x,y)
# plt.show()

# x = np.random.uniform(0,1,100)
# y = np.random.uniform(0,1,100)
# ax.scatter(x,y)
# plt.show()

# z = np.random.uniform(0, 1, (8, 8))
# ax.imshow(z)
# plt.show()

# hist
z = np.random.normal(0, 1, 100)
ax.hist(z)
plt.show()


# hexbin
