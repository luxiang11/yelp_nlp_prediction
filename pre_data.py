import numpy as np
import pandas as pd
from csv import reader

with open(r'D:\大数据\train\business.csv', 'rt', encoding='utf-8') as b_file:
    b_content = list(reader(b_file, delimiter=','))
    b_content = np.array(b_content)[1:]
    b_data = [b_content[i][1] for i in range(len(b_content)) 
              if int(b_content[i][10]) >= 50]
    print(len(b_data))

with open(r'D:\大数据\train\reviews.csv', 'rt', encoding='utf-8') as r_file:
    r_content = list(reader(r_file, delimiter=','))
    r_content = np.array(r_content)[1:]
    r_dict = {}
    for review in r_content:
        r_dict[review[2]] = [r_dict.get(review[2], [0,0])[0] + 1, r_dict.get(review[2], [0,0])[1] + float(review[4])]
    r_data = [r_content[i] for i in range(len(r_content)) 
              if r_dict[r_content[i][2]][0] >= 5 and r_content[i][3] in b_data and 1.5 <= r_dict[r_content[i][2]][1]/r_dict[r_content[i][2]][0] <= 4.5]
    print(len(r_data))
    
r_data = [[r_data[i][8],r_data[i][4]] for i in range(len(r_data))]

test = pd.DataFrame(data=r_data)
#print(test['0'][0])
#print(type(test['0'][0]))
test.to_csv(r'D:\大数据\train\data.csv', header=['review','star'], index=0)