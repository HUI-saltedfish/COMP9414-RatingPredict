import re
import sys
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

flag_sentiment = False


input_data = sys.stdin.readlines()
input_data = list(map(lambda x: x.rstrip('\n'), input_data))
df = {"text": [], "rating": []}
for x in input_data:
    if x == "":
        continue
    list_x = x.split("\t")
    df["text"].append(list_x[-1])
    if flag_sentiment:
        rating = int(list_x[1])
        if rating <= 3:
            df['rating'].append(1)
        elif rating <= 4:
            df['rating'].append(2)
        elif rating <= 5:
            df["rating"].append(3)
    else:
        df['rating'].append(int(list_x[1]))
df = pd.DataFrame(df)
num = df.shape[0]


def process_data(sentence, flag=False):
    sentence = re.sub(r'-{2,}|\.{3,}|~', " ", sentence)
    sentence = re.sub(r'<.*?>', "", sentence)
    sentence = re.sub(r"[^\w\s\d/\-$%]", "", sentence)
    if flag:
        ps = PorterStemmer()
        set_stop = stopwords.words("english")
        sentence_list = sentence.split(" ")
        sentence_ = list()
        for word in sentence_list:
            if word in ps.stem(word) and word not in set_stop:
                sentence_.append(word)
        sentence = " ".join(sentence_)
    return sentence


def model_predict(model, testdata, label):
    y_ = model.predict(testdata)
    for idx in range(int(num * 0.2)):
        print(idx+int(num * 0.8), y_[idx])
    # print(metrics.classification_report(label, y_, zero_division=0))

    # print('accuracy:', metrics.accuracy_score(label, y_))
    # print('Macro-precision:', metrics.precision_score(label, y_, average='macro'))
    # print('Micro-precision:', metrics.precision_score(label, y_, average='micro'))
    # print('Weighted- precision:', metrics.precision_score(label, y_, average='weighted'))
    # print('Macro-recall:', metrics.recall_score(label, y_, average='macro'))
    # print('Micro-recall:', metrics.recall_score(label, y_, average='micro'))
    # print('Weighted-recall:', metrics.recall_score(label, y_, average='micro'))
    # print('Macro-F1-score:', metrics.f1_score(label, y_, labels=[1, 2, 3, 4], average='macro'))
    # print('Micro-F1-score:', metrics.f1_score(label, y_, labels=[1, 2, 3, 4], average='micro'))
    # print('Weighted-F1-score:', metrics.f1_score(label, y_, labels=[1, 2, 3, 4], average='weighted'))


df["text"] = df["text"].apply(process_data)
cv = CountVectorizer(lowercase=True, token_pattern=r'[/\-$%a-zA-z0-9]{2,}')
train_data = cv.fit_transform(df["text"].loc[:int(num * 0.8)])
test_data = cv.transform(df["text"].loc[int(num * 0.8):])

classifier = BernoulliNB()
model = classifier.fit(train_data, df["rating"].loc[:int(num * 0.8)])
model_predict(model, test_data, df["rating"].loc[int(num * 0.8):])



