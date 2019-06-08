
import WordCount
import re
commentType = [
    r'<!--\+//-->',
    "//",
    "<!--"
]

def commentFreq(path,filename,commentSet):

    file = open(path+'\\'+filename,'r+',encoding = "ISO-8859-1")
    str = file.read()
    match1 = re.findall(r'<!--.*?//.*?-->',str)
    match2 = re.findall(r'<!--.*?',str)
    match3 = re.findall(r'.*?//',str)
    ofu = len(match1)
    all = len(match2)+len(match3)
    if (all == 0):
        return 0
    else:
        return ofu / (all - ofu)

# str = '<!--aajgfkj//dghdkjsgh--><!--//-->'
# match1 = re.findall(r'<!--.*?//.*?-->',str)
# match2 = re.findall(r'<!--.*?',str)
# match3 = re.findall(r'.*?//',str)
# ofu = len(match1)
# all = len(match2) + len(match3)
# print(ofu)
# print(all)
# print(ofu / (all - ofu))





