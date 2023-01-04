#用SVC模型进行训练

import pickle
from word_sequece import WordSequence
from dataset import get_dataloader
from sklearn.metrics import accuracy_score

model = pickle.load(open("D:\大四上\大数据引论\代码\models\model3.pkl", "rb"))
print(type(model))
test_dataloader = get_dataloader(train=False)
for idx, (input, target) in enumerate(test_dataloader):
    output = model.predict(input)
    accuracy = accuracy_score(target, output)
    print(accuracy)


