import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from data_processing import data_processing
import pandas as pd
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    # 读取数据
    train = pd.read_csv('labeledTrainData.tsv', header=0, delimiter='\t', quoting=3)
    test = pd.read_csv('testData.tsv', header=0, delimiter='\t', quoting=3)
    # 创建一个空的数组
    clean_train_reviews = []
    # review的值放到clean_train_reviews中
    for i in range(0, len(train["review"])):
        clean_train_reviews.append(" ".join(data_processing.review_to_wordlist(train["review"][i], True)))
    # CountVectorizer会将文本中的词语转换为词频矩阵，它通过fit_transform函数计算各个词语出现的次数
    vectorizer = CountVectorizer(analyzer="word", tokenizer=None, stop_words=None, max_features=5000,
                                 token_pattern=r"(?u)\b\w+\b")
    # 拟合模型，并返回文本矩阵
    train_data_features = vectorizer.fit_transform(clean_train_reviews)
    # 将每个词语出现的次数写成矩阵
    np.asarray(train_data_features)
    # 随机森林
    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(train_data_features, train["sentiment"])
    clean_test_reviews = []
    for i in range(0, len(test["review"])):
        clean_test_reviews.append(" ".join(data_processing.review_to_wordlist(test["review"][i], True)))
    test_data_features = vectorizer.transform(clean_test_reviews)
    np.asarray(test_data_features)
    test_predict = forest.predict(test_data_features)
    # 输出结果
    output = pd.DataFrame(data={"id": test["id"], "sentiment": test_predict})
    output.to_csv('part1.csv', ' ', index=False, quoting=3)
