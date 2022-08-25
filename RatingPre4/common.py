import pathlib
import sys
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics import classification_report


def get_input():
    input_ = sys.stdin.readlines()
    data_ls = [x.rstrip('\n') for x in input_]
    data = {"text": [], "rating": []}
    for row in data_ls:
        if row == "":
            continue
        row_ls = row.split('\t')
        data["text"].append(row_ls[2])
        data["rating"].append(int(row_ls[1]))
    num = len(data["text"])
    return data, num


def replace_by_space(sentence):
    sentence = re.sub(r'~|-{2,}|\.{3,}', ' ', sentence)
    return sentence


def remove_junk_words(sentence):
    sentence = re.sub(r"<.*?>", "", sentence)
    sentence = re.sub(r"[^\w/\-$%\s0-9]", "", sentence)
    return sentence


def stem(sentence):
    ps = PorterStemmer()
    words = sentence.split(" ")
    ret_sentence = list()
    for word in words:
        if word in ps.stem(word):
            ret_sentence.append(word)
    return " ".join(ret_sentence)


def stop(sentence):
    stop_set = set(stopwords.words("english"))
    words = sentence.split(" ")
    ret_sentence = list()
    for word in words:
        if word not in stop_set:
            ret_sentence.append(word)
    return " ".join(ret_sentence)


def process_data(data, flag=False):
    data["text"] = list(map(replace_by_space, data["text"]))
    data["text"] = list(map(remove_junk_words, data["text"]))
    if flag:
        data["text"] = list(map(stem, data["text"]))
        data["text"] = list(map(stop, data["text"]))
    return data


def test_predict(model, test_data, y, num):
    y_ = model.predict(test_data)
    for idx in range(int(num*0.2)):
        print(idx+int(num*0.8), y_[idx])
    # print(classification_report(y, y_, zero_division=0))