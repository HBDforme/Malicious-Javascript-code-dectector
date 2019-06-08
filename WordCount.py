# -*- coding: UTF-8 -*-



#传入需要统计词频的路径，文件名，和统计词集合
def word_count(path,filename,str):
    file = open(path +'\\'+filename,'r+',encoding = "ISO-8859-1")
    times = 0
    for line in file.readlines():
        # print(line.count(str))
        times = times + line.count(str)
        if (not line):  # 若读取结束了
            break
    file.close()
    return times










