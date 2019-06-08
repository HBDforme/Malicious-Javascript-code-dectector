
#计算字符串的字符熵

from math import log

#计算数据集信息熵的函数
def Ent(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}

    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt


#构建文本文件的字符数据集，并计算字符熵
def caculaEnt(path,filename):
    # 把文本读入字符串中
    file = open(path + '\\' + filename, 'r+', encoding="ISO-8859-1")
    str = file.read()
    testSet = list(str)
    rel = Ent(testSet)
    return rel





