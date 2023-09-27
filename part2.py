import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import logging
import nltk.data
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from data_processing import data_processing


# 构建特征矩阵
def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,), dtype="float32")
    new_words = 0
    index2word_set = set(model.wv.index_to_key)
    for word in words:
        if word in index2word_set:
            new_words = new_words + 1
            featureVec = np.add(featureVec, model.wv[word])
    featureVec = np.divide(featureVec, new_words)
    return featureVec


# 获得平均特征矩阵
def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    for review in reviews:
        if counter % 1000 == 0:
            print("Review %d of %d" % (counter, len(reviews)))
        reviewFeatureVecs[int(counter)] = makeFeatureVec(review, model, num_features)
        counter = counter + 1
    return reviewFeatureVecs


# 获得处理过的review
def getCleanReviews(reviews):
    clean_reviews = []
    for review in reviews["review"]:
        clean_reviews.append(data_processing.review_to_wordlist(review, remove_stopwords=False))
    return clean_reviews


if __name__ == '__main__':
    # 获取数据
    train = pd.read_csv('labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
    test = pd.read_csv('testData.tsv', header=0, delimiter="\t", quoting=3)
    unlabeled_train = pd.read_csv('unlabeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
    # 调用tokenizer，将段落中的句子进行拆分
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = []
    for review in train["review"]:
        sentences += data_processing.review_to_sentences(review, tokenizer)
    for review in unlabeled_train["review"]:
        sentences += data_processing.review_to_sentences(review, tokenizer)
    # 显示运行日志
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # 设定训练参数
    num_features = 300
    min_word_count = 40
    num_workers = 4
    context = 10
    downsampling = 1e-3
    # 建立模型
    model = Word2Vec(sentences, workers=num_workers, vector_size=num_features, min_count=min_word_count, window=context,
                     sample=downsampling, seed=1)
    model_name = "300"
    model.save(model_name)
    # 获得训练数据集的特征矩阵和测试数据集的特征矩阵
    trainDataVecs = getAvgFeatureVecs(getCleanReviews(train), model, num_features)
    testDataVecs = getAvgFeatureVecs(getCleanReviews(test), model, num_features)
    # 随机森林进行预测
    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(trainDataVecs, train["sentiment"])
    result = forest.predict(testDataVecs)
    # 将数据预测的结果写入文件中
    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    output.to_csv("part2.csv", ' ', index=False, quoting=3)
