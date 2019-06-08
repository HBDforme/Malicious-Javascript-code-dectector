

#只有在文件结构未建立时先执行

import os
import shutil
import re

path = 'E:\\Zpaper\\sample'
path02 = 'E:\\Zpaper\\sample\\Benign'
# path02 = 'E:\\Zpaper\\MJDetector-master\\TestingSet\\goodjs'
# path03 = 'E:\\Zpaper\\MJDetector-master\\TrainingSet\\badjs'
# path04 = 'E:\\Zpaper\\MJDetector-master\\TrainingSet\\goodjs'


def cleanTxt(filename):
    jug = filename.count('.txt')
    if(jug > 0):
        str_del = re.escape('.txt')
        newname = re.sub(str_del,"",filename)
        return newname
    else:
        return filename

def cleanName(path):
    #change the path
    os.chdir(path)
    for filedir in os.listdir(path):
        newpath = path+'\\'+filedir
        for filename in os.listdir(newpath):
            newname = cleanTxt(filename)
            os.rename(newpath+'\\'+filename,newpath+'\\'+newname)


#建立提取特征的文件结构
def bulidfilestruction(path):
    os.chdir(path)
    for filedir in os.listdir(path):
        newpath = path + '\\' + filedir
        for filename in os.listdir(newpath):

            dirname = 'sample_'+ filename
            newdir = newpath+'\\'+dirname
            os.mkdir(newdir)
            shutil.move(newpath+'\\'+filename,newdir)

def bulidBenign(path02):
    os.chdir(path02)
    for filename in os.listdir(path02):
        dirname = 'sample_' + filename
        newdir = path02 + '\\' + dirname
        os.mkdir(newdir)
        shutil.move(path02+ '\\' + filename, newdir)


path001 = 'E:\\Zpaper\\sample\\Benign_1'
path002 = 'E:\\Zpaper\\sample\\Benign_2'
path003 = 'E:\\Zpaper\\sample\\Benign_3'
path004 = 'E:\\Zpaper\\sample\\Benign_4'
bulidBenign(path001)
bulidBenign(path002)
bulidBenign(path003)
bulidBenign(path004)

#建立文件结构
# cleanName(path)
# bulidfilestruction(path)

#统计数据集数量
def count(path):
    os.chdir(path)
    for filedir in os.listdir(path):
        number = 0
        newpath = path + '\\' + filedir
        for sample_dir in os.listdir(newpath):
            number=number+1
        print('the number of '+filedir+'is '+str(number))

# count(path)

#分割正常代码集合，以避免nodejs打开文件过多的错误
# import os
# import shutil
# path1 = 'E:\\Zpaper\\sample'
# path2 = 'E:\\Zpaper\\sample\\Benign'
# os.chdir(path1)
# newdir1 = path1 + '\\' + 'Benign_1'
# newdir2 = path1 + '\\' + 'Benign_2'
# newdir3 = path1 + '\\' + 'Benign_3'
# newdir4 = path1 + '\\' + 'Benign_4'
# os.mkdir(newdir1)
# os.mkdir(newdir2)
# os.mkdir(newdir3)
# os.mkdir(newdir4)
#
# for filename in os.listdir(path2):
#     if(int(filename)<6000):
#         shutil.move(path2+ '\\' + filename, newdir1)
#     elif(int(filename)<12000):
#         shutil.move(path2+ '\\' + filename, newdir2)
#     elif (int(filename) < 18000):
#         shutil.move(path2+ '\\' + filename, newdir3)
#     else:
#         shutil.move(path2+ '\\' + filename, newdir4)