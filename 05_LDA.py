import csv
import numpy as np
import matplotlib.pyplot as plt

def normalize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    normalized_data = (data-mean)/std
    return normalized_data

def calculate_Sw_mean(data):
    mean = np.mean(data, axis=0)
    n = data.shape[0]
    s = np.dot((data-mean).T, (data-mean))
    # s = 0
    # for i in range(n):
    #     s+=(data[i,:]-mean).T*(data[i,:]-mean)
    return mean, s

def lda(data_red, data_white, n_components=2):
    mean_red, Sw_red = calculate_Sw_mean(data_red)
    mean_white, Sw_white = calculate_Sw_mean(data_white)
    n1, n2 = data_red.shape[0], data_white.shape[0]
    Sw = (n1*Sw_red + n2*Sw_white)/(n1+n2)

    data_all = np.vstack([data_red, data_white])
    mean, n = np.mean(data_all, axis=0), data_all.shape[0]

    Sb=(n1*(mean-mean_red).T.dot((mean-mean_red))
        +n2*(mean-mean_white).T.dot((mean-mean_white))) /(n1+n2)

    value, vector=np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
    
    sorted_indices = np.argsort(value)[::-1]
    sorted_vectors = vector[:, sorted_indices]

    principal_components = sorted_vectors[:, :n_components]
    X_transformed = data_all.dot(principal_components)
    return np.array(X_transformed)

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
    x_lda = lda(red_wine_datas[:, :11], white_wine_datas[:, :11], 2)
    data_lda = np.hstack([x_lda, dataset[:, 11].reshape(dataset.shape[0],1)])
    plt.title('lda')
    plt.scatter(data_lda[data_lda[:, 2] == 1][:, 0], data_lda[data_lda[:, 2] == 1][:, 1], color='r',alpha=0.7)
    plt.scatter(data_lda[data_lda[:, 2] == 0][:, 0], data_lda[data_lda[:, 2] == 0][:, 1], color='b',alpha=0.3)
    plt.show()
    

    