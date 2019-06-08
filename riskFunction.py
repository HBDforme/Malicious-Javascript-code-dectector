
import WordCount
riskFunctionList = [
    'eval',
    'window.setInterval',
    'window.setTimeout',
    'location.replace',
    'location.assign',
    'getUserAgent',
    'getAppName',
    'getCookie',
    'setCookie',
    'document.addEventListerner',
    'element.addEventListerner',
    'document.write',
    'element.changeAttribute',
    'document.writeIn',
    'element.innerHTML',
    'element.insertBefore',
    'element.appendChild',
    'element.replaceChild',
    'String.charAt',
    'String.charCodeAt',
    'String.fromCharCode',
    'String.indexOf',
    'String.split'
]

def callRiskTimes(path,filename,functionList):
    alltime = 0
    for func in functionList:
        rel = WordCount.word_count(path, filename, func)
        alltime = alltime + rel
    return alltime



