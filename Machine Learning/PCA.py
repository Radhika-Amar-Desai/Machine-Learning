import pandas
from sklearn.decomposition import PCA
# read the csv file
file = pandas.read_csv("data.csv")
# modifying the diagnosis col value since it contains string values
def assign_int(col):
    for i in range(len(file[col])):
        file[col][i] = ord(file[col][i])
assign_int("diagnosis")
# applying pca after removing nan values
file = file.dropna(axis=1)
pca = PCA(n_components=2)
pca.fit_transform(file)
print(pca.explained_variance_ratio_)
print(pca.get_feature_names_out)