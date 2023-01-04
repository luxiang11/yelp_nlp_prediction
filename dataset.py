"""
准备数据
"""
from torch.utils.data import DataLoader,Dataset
import torch
import utils
import pandas as pd
import pickle
from word_sequece import WordSequence

ws = pickle.load(open("D:\senior_up\大数据引论\代码\models\ws_8000.pkl","rb"))

class myDataset(Dataset):
    def __init__(self,train=True):
        if train:
            self.data_text_path = r"D:\senior_up\大数据引论\train\text_train.csv"
            self.data_stars_path = r"D:\senior_up\大数据引论\train\stars_train.csv"
        else:
            self.data_text_path = r"D:\senior_up\大数据引论\test\text_test.csv"
            self.data_stars_path = r"D:\senior_up\大数据引论\test\stars_test.csv"
        self.total = pd.read_csv(self.data_stars_path)['stars']

    def __getitem__(self, idx):
        file_text = pd.read_csv(self.data_text_path)['text'][idx]
        file_stars = pd.read_csv(self.data_stars_path)['stars'][idx]

        text = utils.tokenlize(file_text)
        stars = int(file_stars) - 1

        return text, stars

    def __len__(self):
        return len(self.total)

def collate_fn(batch):
    """
    对batch数据进行处理
    :param batch: [一个getitem的结果，getitem的结果,getitem的结果]
    :return: 元组
    """
    reviews,labels = zip(*batch)
    reviews = torch.LongTensor([ws.transform(i,max_len=50) for i in reviews])
    labels = torch.LongTensor(labels)

    return reviews,labels

def get_dataloader(train=True):
    dataset = myDataset(train)
    batch_size = 32
    return DataLoader(dataset,batch_size=batch_size,shuffle=True,collate_fn=collate_fn)


if __name__ == '__main__':
    for idx,(review,label) in enumerate(get_dataloader(train=True)):
        print(review.shape)
        print(review)
        print(label)
        break