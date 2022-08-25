import sys
import re
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier


def std_input():
    review_tsv = sys.stdin.readlines()
    review_tsv = list(map(lambda x: x.rstrip('\n'), review_tsv))
    rating = [int(x.split('\t')[1]) for x in review_tsv]
    text = [x.split('\t')[2] for x in review_tsv]
    return text, rating


def preprocess(sequential):
    sequential = re.sub(r'-{2,}|~|\.{3,}', " ", sequential)
    sequential = re.sub(r'<.*?>', "", sequential)
    sequential = re.sub(r"[^a-zA-Z/\-$%\s\d]", '', sequential)
    return sequential


def model_predict(model, test_feature, label, train_num):
    pre = model.predict(test_feature)
    for i in range(len(pre)):
        print(i + train_num, pre[i])
    # print(classification_report(label, pre, zero_division=0))


def main():
    text, rating = std_input()

    num = len(text)
    train_num = int(num * 0.8)
    test_num = int(num * 0.2)
    length = 0.01

    processed_text = [preprocess(sin_text) for sin_text in text]
    flag = False
    token_pattern = r'[/\-$%\w\d]{2,}'
    max_features = 1000
    cv = CountVectorizer(lowercase=flag, token_pattern=token_pattern, max_features=max_features)
    train_feature = cv.fit_transform(processed_text[:train_num])
    test_feature = cv.transform(processed_text[train_num:])

    clf = DecisionTreeClassifier(min_samples_leaf=length, criterion='entropy', random_state=0)
    model = clf.fit(train_feature, rating[:train_num])
    model_predict(model, test_feature, rating[train_num:], train_num)


if __name__ == '__main__':
    main()
