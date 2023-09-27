import time
import numpy as np
from gensim.models import Word2Vec
import pandas as pd
from sklearn.cluster import KMeans
from data_processing import data_processing
from sklearn.ensemble import RandomForestClassifier
from bs4 import BeautifulSoup


# 为每个单词分配一个集群，构成质心袋
def create_bag_of_centroids(wordlist, word_centroid_map):
    num_centroids = max(word_centroid_map.values()) + 1
    bag_of_centroids = np.zeros(num_centroids, dtype="float32")
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    return bag_of_centroids


if __name__ == '__main__':
    model = Word2Vec.load("300")
    # 开始时间
    start = time.time()
    # 设置每五个单词一个集群
    word_vectors = model.wv.vectors
    num_clusters = int(word_vectors.shape[0] / 5)
    # 初始化Kmeans提取kmeans_clustering
    kmeans_clustering = KMeans(n_clusters=num_clusters)
    idx = kmeans_clustering.fit_predict(word_vectors)
    # 获得结束时间，并计算运行时长
    end = time.time()
    elapsed = end - start
    # 创建一个索引，查找每个单词所属的集群
    word_centroid_map = dict(zip(model.wv.index_to_key, idx))
    for cluster in range(0, 10):
        words = []
        for i in range(0, len(word_centroid_map.values())):
            if list(word_centroid_map.values())[i] == cluster:
                words.append(list(word_centroid_map.keys())[i])
    # 读取数据
    train = pd.read_csv('labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
    test = pd.read_csv('testData.tsv', header=0, delimiter="\t", quoting=3)
    # 处理训练集、测试集的review
    clean_train_reviews = []
    for review in train["review"]:
        clean_train_reviews.append(data_processing.review_to_wordlist(review, remove_stopwords=True))
    clean_test_reviews = []
    for review in test["review"]:
        clean_test_reviews.append(data_processing.review_to_wordlist(review, remove_stopwords=True))
    train_centroids = np.zeros((train["review"].size, num_clusters), dtype="float32")
    # 为处理好的review分配集群
    counter = 0
    for review in clean_train_reviews:
        train_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
        counter += 1
    test_centroids = np.zeros((test["review"].size, num_clusters), dtype="float32")
    counter = 0
    for review in clean_test_reviews:
        test_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
        counter += 1
    # 随机森林
    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(train_centroids, train["sentiment"])
    result = forest.predict(test_centroids)
    # 将训练好的数据放到文件里
    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    output.to_csv("part3.csv", ' ', index=False, quoting=3)
