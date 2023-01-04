#用SVC模型进行预测

import sklearn.svm as svm
from word_sequece import WordSequence
from dataset import get_dataloader
import pickle

model = svm.SVC(kernel="rbf", decision_function_shape="ovr")

train_dataloader = get_dataloader(train=True)
#print(len(train_dataloader))
for idx,(input,target) in enumerate(train_dataloader):
    model.fit(input,target)
pickle.dump(model, open("D:\大四上\大数据引论\代码\models\model11.pkl", "wb"))