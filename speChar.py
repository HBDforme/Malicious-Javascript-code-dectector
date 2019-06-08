
import WordCount
import re
speCharList = ['!', '"','#','$', '%','&',"'",'('')', '*','+',',', '-', '.','/',':',';', '<','=','>','?','@','[',']', '^', '_','{','}','|','~'
 ]

def sepfreq(path,filename,speCharList):
    file = open(path + '\\' + filename, 'r+', encoding="ISO-8859-1")
    str = file.read()
    #清楚空格
    str = re.sub(' ', '', str)
    #转为字符列表
    testSet = list(str)
    alltime = 0
    for spechar in speCharList:
        rel = WordCount.word_count(path, filename, spechar)
        alltime = alltime + rel
        if(len(testSet)==0):
            return 0
        else:
            return alltime / len(testSet)

