import pandas as pd
import numpy as np
from pyclustertend import hopkins
from sklearn.preprocessing import scale

path = "/home/ubuntu/users/mateusz/Scene-Representation/project/src/output_embeddings/depth.csv"
data = pd.read_csv(path)
array = data.iloc[:, [1, 2]].to_numpy()
print(hopkins(scale(array), array.shape[0]))



