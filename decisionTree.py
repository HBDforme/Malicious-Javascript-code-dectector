
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.externals import joblib


# from sklearn.datasets import load_iris

# 例子
# X = [[0, 0], [1, 1],[1, 1]]
# Y = [-1, 1,1]
# clf = DecisionTreeClassifier()
# clf = clf.fit(X,Y)
#
# print(clf.predict([[0.,0.]]))
# print(clf.predict_proba([[2., 2.]]))




# iris = load_iris()
# x = iris.data[:,2:]
# y = iris.target

# tree_clf = DecisionTreeClassifier(max_depth=2)
# tree_clf.fit(x,y)
#
# dot_data=export_graphviz(
#     tree_clf,
#     out_file="iris.dot",
#     feature_names=iris.feature_names[2:],
#     class_names=iris.target_names,
#     rounded=True,
#     filled=True
# )
#
# joblib.dump(clf, "train_model.m")

# graph = pydotplus.graph_from_dot_data(dot_data)
# # 保存图像到pdf文件
# graph.write_pdf("iris.pdf")   dot -Tpnp iris.dot -o iris.png


feature_name = [
    'CallRiskTimes',
    'EntChar',
    'LongSize',
    'CommentFreq',
    'DepthAst'
]

def decisiontTrain(x,y):
    tree_clf = DecisionTreeClassifier(max_depth=4)
    tree_clf.fit(x, y)

    # export_graphviz(
    #     tree_clf,
    #     feature_names = feature_name,
    #     out_file="dect_DT.dot",
    #     rounded=True,
    #     filled=True
    # )
    # joblib.dump(tree_clf, "train_model_decisionTree.m")



def decisionPredit(x):
    # 调模型
    clf = joblib.load("train_model_decisionTree.m")
    predit_nat = clf.predict(x)
    return predit_nat

from sklearn.model_selection import train_test_split
import h5py
import numpy as np
f = h5py.File("feature.hdf5","r")
x = f['feature_sample'][()]
y = f['feature_target'][()]

#切数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

tree_clf = DecisionTreeClassifier(max_depth=8)
tree_clf.fit(x_train, y_train)
joblib.dump(tree_clf, "DT.m")
pred = tree_clf.predict(x_test)
acurr = np.mean(pred == y_test)
print(acurr)

from sklearn import svm
from sklearn.naive_bayes import GaussianNB

svm_clf = svm.SVC(gamma='auto')
svm_clf.fit(x_train, y_train)
joblib.dump(svm_clf, "SVM.m")
pred = svm_clf.predict(x_test)
acurr = np.mean(pred == y_test)
print(acurr)

gnb = GaussianNB()
pred = gnb.fit(x_train, y_train).predict(x_test)
joblib.dump(gnb, "NB.m")
acurr = np.mean(pred == y_test)
print(acurr)

# from sklearn.model_selection import cross_val_score
# scores1 = cross_val_score(tree_clf, x, y, cv=5)
# scores2 = cross_val_score(svm_clf, x, y, cv=5)
# scores3 = cross_val_score(gnb, x, y, cv=5)
# print(scores1,scores2,scores3)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores1.mean(), scores1.std() * 2))
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores2.mean(), scores2.std() * 2))
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores3.mean(), scores3.std() * 2))