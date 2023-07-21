import numpy as np
from sklearn.manifold import TSNE


x = np.array([[1, 2, 4], [6, 8, 10], [5, np.nan, 6], [4, np.inf, 7]])
#x = x[~np.isnan(x).any(axis=1)]
#x = x[~np.isinf(x).any(axis=1)]
x = x[np.isfinite(x).all(axis=1)]
tsne = TSNE(n_components=2, perplexity=1, n_iter=5000).fit_transform(x)



