from data_structuring import *
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


df_train = pd.read_parquet("data_files/bdd100k_classificaton_train.parquet", engine="fastparquet")
df_val = pd.read_parquet("data_files/bdd100k_classificaton_val.parquet", engine="fastparquet")

X_train, X_test, y_train, y_test = train_test_split(df_train["name"].values, df_train[["weather", "scene", "timeofday"]].values, shuffle=True, random_state=1, stratify=df_train["weather"])
   
class CustomDataset(Dataset):
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = labels

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        root = "/Users/matt/Documents/Imperial College London/Modules/MSc - Scene Representation and Pre-Tagging/dataset/bdd100k/images/100k"
        image = np.array(Image.open(os.path.join(root, "train", self.imgs[index])).convert("RGB"))
        labels = self.labels[index]

        return image, (labels[0], labels[1], labels[2])


training_data = CustomDataset(X_test, y_test)
f = open("data_files/tags.json")
tags = json.load(f)

for idx in range(10): # Visualising first 10 data points for testing.
    img, labels = training_data[idx]
    visualise_data_with_tags(img, labels, tags)

