from common import get_input, process_data, test_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer

data, num = get_input()
data = process_data(data, flag=False)

cv = CountVectorizer(lowercase=False, token_pattern=r'[/\-$%\w\d]{2,}', max_features=1000)
train_data = cv.fit_transform(data["text"][:int(num * 0.8)])
test_data = cv.transform(data["text"][int(num * 0.8):])

classifier = DecisionTreeClassifier(min_samples_leaf=0.01, criterion='entropy', random_state=0)
model = classifier.fit(train_data, data["rating"][:int(num * 0.8)])
test_predict(model, test_data, data["rating"][int(num * 0.8):], num)
