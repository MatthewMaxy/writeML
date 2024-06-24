import csv
import numpy as np
import matplotlib.pyplot as plt

def normalize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    normalized_data = (data-mean)/std
    return normalized_data

def pca(data, n_components = 2):
    mean = np.mean(data, axis=0)
    data -= mean
    cov_matrix = np.cov(data.T)
    value, vector = np.linalg.eig(cov_matrix)

    sorted_indices = np.argsort(value)[::-1]
    sorted_vectors = vector[:, sorted_indices]

    principal_components = sorted_vectors[:, :n_components]
    X_transformed = data.dot(principal_components)
    return X_transformed

if __name__ == "__main__":
    red_wine_data = []
    white_wine_data = []

    with open('data/winequality-red.csv', 'r') as file:
        reader = csv.reader(file, delimiter=';')
        next(reader)  # 跳过标题行
        for row in reader:
            red_wine_data.append(row[:11])  # 只取前11列特征

    with open('data/winequality-white.csv', 'r') as file:
        reader = csv.reader(file, delimiter=';')
        next(reader)  # 跳过标题行
        for row in reader:
            white_wine_data.append(row[:11])  # 只取前11列特征

    red_wine_datas =np.hstack([np.array(red_wine_data, dtype=np.float64),np.ones((len(red_wine_data), 1))])
    white_wine_datas =np.hstack([np.array(white_wine_data, dtype=np.float64),np.zeros((len(white_wine_data), 1))])

    dataset = np.vstack([red_wine_datas, white_wine_datas])
    
    normalized_dataset = normalize(dataset[:, :11])

    pca_dataset = pca(normalized_dataset[:, :11], 2)
    data_pca = np.hstack([pca_dataset, dataset[:, 11].reshape(dataset.shape[0],1)])
    print(type(data_pca))
    plt.title('PCA')
    plt.scatter(data_pca[data_pca[:, 2] == 1][:, 0], data_pca[data_pca[:, 2] == 1][:, 1], color='r',alpha=0.7)
    plt.scatter(data_pca[data_pca[:, 2] == 0][:, 0], data_pca[data_pca[:, 2] == 0][:, 1], color='b',alpha=0.3)
    plt.show()


    