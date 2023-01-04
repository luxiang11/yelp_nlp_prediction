# 环境依赖
python 3.9

# 模型框架
pytorch==1.12.1

# 目录结构描述
|--README.txt                //help
|--data_preprocessing        //数据预处理
|  |--predata.py             //数据清洗
|  |--word_sequence.py       //文本序列化
|  |--utils.py               //分词
|  |--tf-idf.py              //加权平均
|  |--bert.py                //bert模型
|--dataset.py                //构建自己的dataset和dataloader
|--model.py                  //搭建神经网络
|--train                     //模型训练
|  |--train.py               //神经网络模型训练
|  |--train_SVC.py           //支持向量机模型训练
|--test                      //模型测试
|--|--test.py
|--|--predict_SVC.py