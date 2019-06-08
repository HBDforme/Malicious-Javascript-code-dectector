

def depthast(path,filename):
    file = open(path + '\\' + filename, 'r+', encoding="utf8")
    str = file.read()
    depth = int(str)
    return depth