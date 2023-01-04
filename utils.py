"""
实现额外的方法
"""
import re

def tokenlize(sentence, mode='bigram'):
    """
    进行文本分词
    :param sentence: str
    :return: [str,str,str]
    """

    fileters = ['!', '"', '#', '\$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '\.', '/', ':', ';', '<', '=', '>',
                '\?', '@', '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”', '“',' a ',' and ',' that ',' is ',' an ',' the ',' in ',' at ',' about ',' by ',' near ',' with ',' on ',' it ',' for ','[0-9]+',' be ',' i ',' also ',' we ']
    sentence = sentence.lower() #把大写转化为小写
    #print(sentence)
    sentence = re.sub("<br />"," ",sentence)
    sentence = re.sub("|".join(fileters)," ", sentence)
    sentence = re.sub("|".join(fileters)," ", sentence)
    result = [i for i in sentence.split(" ") if len(i)>0]
    #print(result)
    if mode == 'bigram':
        result_bi = [result[i] + ' ' + result[i+1] for i in range(len(result) - 1)]
    #a = input('')

    return result


def tokenlize_mini(sentence, mode='bigram'):
    """
    进行文本分词
    :param sentence: str
    :return: [str,str,str]
    """

    fileters = ['!', '"', '#', '\$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '\.', '/', ':', ';', '<', '=', '>',
                '\?', '@', '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”', '“',' a ',' and ',' that ',' is ',' an ',' the ',' in ',' at ',' about ',' by ',' near ',' with ',' on ',' it ',' for ','[0-9]+',' be ',' i ',' also ',' we ']
    sentence = sentence.lower() #把大写转化为小写
    #print(sentence)
    sentence = re.sub("<br />"," ",sentence)
    sentence = re.sub("|".join(fileters)," ", sentence)
    sentence = re.sub("|".join(fileters)," ", sentence)

    return sentence

