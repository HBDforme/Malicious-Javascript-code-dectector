import re



def getLongestWord(path,filename):
    file = open(path + '\\' + filename, 'r+', encoding="ISO-8859-1")
    str = file.read()
    wordSet = re.split(r' ', str)

    longest = 0
    for word in wordSet:
        if (longest < len(word)):
             longest = len(word)
    return longest



