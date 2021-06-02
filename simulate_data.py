import pickle
from data_generator_ours import *
import time

# %%

graph_type = 'BA'
edge_type = 'lognormal'
graph_size = 20
set_ = "test"
weight_scale = True
ee = 1e-1

graph_hyper = {'k': 4,
               'p': 0.2}

s = time.time()
data = generate_BA_parallel(num_samples=64,
                            num_signals=100,
                            num_nodes=graph_size,
                            graph_hyper=3,
                            weighted=edge_type,
                            ee=ee,
                            weight_scale=weight_scale)
e = time.time()
print(e - s)

# %%

import matplotlib.pyplot as plt
import seaborn as sns

W = data['W'][0].A
plt.figure()
sns.heatmap(W)
plt.show()

# %%

with open('/home/vanessa/Documents/code/graph/glad_new/'
          'data/syn/dataset_{}_{}_{}nodes_{}_{}_{}.pickle'
                  .format(graph_type, edge_type, graph_size, set_, weight_scale, ee), 'wb') as handle:
    pickle.dump(data, handle, protocol=4)
