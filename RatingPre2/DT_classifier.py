import sys
import re
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier

cv = CountVectorizer(lowercase=False, token_pattern=r'[/\-$%\w0-9]{2,}', max_features=1000)


data = sys.stdin.read()
data_row_ls = data.split('\n')
instance_number = list()
rating = list()
text = list()
for row in data_row_ls:
    if row == "":
        continue
    row_ls = row.split('\t')
    instance_number.append(int(row_ls[0]))
    rating.append(int(row_ls[1]))
    text.append(row_ls[2])

total_num = len(instance_number)


def replace(sentence):
    pattern = re.compile(r'-{2,}|~|\.{3,}')
    sentence_ = re.sub(pattern, " ", sentence)
    return sentence_


def remove(sentence):
    pattern = re.compile(r"<.*?>")
    sentence_ = re.sub(pattern, "", sentence)
    pattern = re.compile(r"[^a-zA-Z/\-$%\s\d]")
    sentence_ = re.sub(pattern, "", sentence_)
    return sentence_


def preprocess_text(sentence):
    sentence_ = replace(sentence)
    sentence_ = remove(sentence_)
    return sentence_

text = list(map(preprocess_text, text))
train_cv = cv.fit_transform(text[:int(total_num*0.8)])
test_cv = cv.transform(text[int(total_num*0.8):])

def predict_rating(model, test_cv):
    pre = model.predict(test_cv)
    for idx in range(int(total_num*0.8), total_num):
        print(idx, pre[idx - int(total_num*0.8)])
    # print(classification_report(rating[int(total_num*0.8):], pre, zero_division=0))



length = 0.1 * 0.1
model = DecisionTreeClassifier(min_samples_leaf=length, criterion='entropy', random_state=0)
model = model.fit(train_cv, rating[:int(total_num*0.8)])
predict_rating(model, test_cv)


