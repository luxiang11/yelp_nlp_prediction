import math

def tf_idf(reviews):
    #总词频统计
    review_times = {}
    for word_list in reviews:
        for i in word_list:
            review_times[i] = review_times.get(i, 0) + 1
 
    #计算每个词的TF值
    word_tf = {}  #存储tf值
    for i in review_times:
        word_tf[i] = review_times[i]/sum(review_times.values())
 
    #计算每个词的IDF值
    doc_num = len(reviews)
    word_idf = {} #存储idf值
    word_re = {} #存储包含该词的评论数
    for i in review_times:
        for j in reviews:
            if i in j:
                word_re[i] = word_re.get(i, 0) + 1
    for i in review_times:
        word_idf[i] = math.log(doc_num/(word_re[i]+1))
 
    #计算每个词的TF*IDF的值
    word_tf_idf={}
    for i in review_times:
        word_tf_idf[i]=word_tf[i]*word_idf[i]