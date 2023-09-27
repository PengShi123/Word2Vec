import re
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk
import pandas as pd


class data_processing(object):
    @staticmethod
    def review_to_wordlist(review, remove_stopwords=False):
        review_text = BeautifulSoup(review, 'lxml').get_text()
        review_text = re.sub("[a-zA-z]", " ", review_text)
        words = review_text.lower().split()
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops]
        return words

    @staticmethod
    def review_to_sentences(review, tokenizer, remove_stopwords=False):
        raw_sentences = tokenizer.tokenize(review)
        sentences = []
        for raw_sentence in raw_sentences:
            if len(raw_sentence) > 0:
                sentences.append(data_processing.review_to_wordlist(raw_sentence, remove_stopwords))
        return sentences
