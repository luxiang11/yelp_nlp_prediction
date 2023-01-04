from bert_serving.client import BertClient
import numpy as np
import pandas

bc = BertClient(check_length=False)
res = pandas.read_csv('data.csv', usecols=['review'])
res = np.array(res).tolist()
for i in range(len(res)):
    res[i] = res[i][0]
res = bc.encode(res)
np.save('vec-res.npy', res)

print(np.load('vec-res.npy').shape)