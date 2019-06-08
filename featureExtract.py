
import os
import riskFunction
import caculaEntofChar
import getLongestWord
import commentTypeFreq
import depthAst
import speChar
import h5py

path = 'E:\\Zpaper\\sample'

#sign代表样本的标识，1代表正常代码，-1代表恶意代码
def createData(path,sign):
    count = 0
    sample = []
    for filedir in os.listdir(path):
        path_new = path+'\\'+filedir
        for filename in os.listdir(path_new):
            filename = str(filename)
            des = filename.find('_depth')
            des = -des
            if(des>0):
                # print(des)
                vec = createMatrix(path_new, filename)
                sample.append(vec)
                count = count + 1
    nat = []
    for i in range(count):
        nat.append(sign)
    return sample,nat

#生成每个文件对应的样本特征向量
def createMatrix(path,filename):

    risk = riskFunction.callRiskTimes(path,filename,riskFunction.riskFunctionList)
    ent = caculaEntofChar.caculaEnt(path,filename)
    longsize = getLongestWord.getLongestWord(path,filename)
    commentfreq = commentTypeFreq.commentFreq(path,filename,commentTypeFreq.commentType)
    depth = depthAst.depthast(path,filename+'_depth')
    charfreq = speChar.sepfreq(path,filename,speChar.speCharList)
    sampel_vector = [risk,ent,longsize,commentfreq,depth,charfreq]
    return sampel_vector

#提取特征
def extractFeature(path):
    x = []
    y = []
    neg_x = []
    neg_y = []
    for type in os.listdir(path):
        if(type=='Benign_1'or type=='Benign_2' or type=='Benign_3' or type=='Benign_4'):
            temp_x,temp_y = createData(path+'\\'+type,1)
            x.extend(temp_x)
            y.extend(temp_y)
        else:
            pass
            # temp_neg_x,temp_neg_y = createData(path+'\\'+type,-1)
            # neg_x.extend(temp_neg_x)
            # neg_y.extend(temp_neg_y)
    x.extend(neg_x)
    y.extend(neg_y)
    return x,y

#保存提取的特征向量
def feature_save(x,y):
    #保存提取的特征矩阵
    f = h5py.File("feature.hdf5","w")
    f.create_dataset('feature_sample',data = x)
    f.create_dataset('feature_target', data = y)


x,y = extractFeature(path)
feature_save(x,y)
