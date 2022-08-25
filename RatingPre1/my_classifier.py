from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import sys
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer



def split_data(df_):
    n = df.shape[0]
    train_data = df_.loc[:int(n*0.8)].copy()
    test_data = df.loc[int(n*0.8):].copy()
    return train_data, test_data


def removing_tags(data):
    pattern = re.compile(r'<.*?>')
    data = re.sub(pattern, '', data)
    return data


def delete_symbol(data):
    pattern = re.compile(r"[^a-zA-Z/\-$%\s\d]")
    data = re.sub(pattern, '', data)
    return data


def replacing_by_space(data):
    pattern = re.compile(r'--|~|\.{3,}')
    data = re.sub(pattern, ' ', data)
    return data


def stemming(data):
    port_stem = PorterStemmer()
    word_ls = data.split(" ")
    after_stem_word = list()
    for sin_word in word_ls:
        if sin_word in port_stem.stem(sin_word):
            after_stem_word.append(sin_word)
    after_data = ' '.join(after_stem_word)
    return after_data


def stop_words(data):
    stop_words_set = set(stopwords.words('english'))
    word_ls = data.split(" ")
    after_stop_word = list()
    for sin_word in word_ls:
        if sin_word not in stop_words_set:
            after_stop_word.append(sin_word)
    after_data = " ".join(after_stop_word)
    return after_data


def remove_junk_word(df):
    df["text"] = df["text"].apply(replacing_by_space)
    df["text"] = df["text"].apply(removing_tags)
    df["text"] = df["text"].apply(delete_symbol)
    df["text"] = df["text"].apply(stemming)
    df["text"] = df["text"].apply(stop_words)
    return df


df = {"instance number": [], "rating": [], "text": []}
data = sys.stdin.readlines()
for row in data:
    row = row[:-2]
    row_ls = row.split('\t')
    df['instance number'].append(int(row_ls[0]))
    df['rating'].append(int(row_ls[1]))
    df['text'].append(row_ls[2])
df = pd.DataFrame(df)
train_data, test_data = split_data(df)

train_data = remove_junk_word(train_data)
test_data = remove_junk_word(test_data)

countVec = CountVectorizer(lowercase=False, token_pattern=r'[/\-$%\w\d]{2,}', max_features=1000)
bag_words_train = countVec.fit_transform(train_data["text"])
# print(countVec.vocabulary_)
# print(bag_words_train)

bag_words_test = countVec.transform(test_data['text'])

# print(bag_words_test)


def pred(model, bag_words_test, test_data):
    y_pred = model.predict(bag_words_test)
    for i in range(test_data.shape[0]):
        print(i + int(df.shape[0] * 0.8), y_pred[i])
    # print(classification_report(test_data['rating'], y_pred, zero_division=0))


clf = LinearSVC(class_weight='balanced')
model = clf.fit(bag_words_train, train_data["rating"])
pred(model, bag_words_test, test_data)
