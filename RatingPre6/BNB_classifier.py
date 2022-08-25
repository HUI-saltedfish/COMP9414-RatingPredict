import re
import sys
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

count_vector = CountVectorizer(lowercase=False, token_pattern=r'[/\-$%a-zA-z0-9]{2,}')
string1 = r'-{2,}|~|\.{3,}'
string2 = r'<.*?>'
string3 = r"[^a-zA-Z/\-$%\s\d]"


instance_number = list()
rating = list()
text = list()

texts = sys.stdin.readlines()
for sin_text in texts:
    if sin_text == "":
        continue
    text.append(sin_text.split("\t")[-1])
    rating.append(int(sin_text.split('\t')[-2]))
    instance_number.append(int(sin_text.split('\t')[0]))
num_all = len(instance_number)

after_text = list()
for sin_text in text:
    sin_text = re.sub(string1, " ", sin_text)
    sin_text = re.sub(string2, "", sin_text)
    sin_text = re.sub(string3, '', sin_text)
    after_text.append(sin_text)

matrix_train = count_vector.fit_transform(after_text[:int(num_all*0.8)])
matrix_text = count_vector.transform(after_text[int(num_all*0.8):])

DT = BernoulliNB()
md = DT.fit(matrix_train, rating[:int(num_all*0.8)])
y = md.predict(matrix_text)
for i in range(len(y)):
    print(i + int(num_all*0.8), y[i])
