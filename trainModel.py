
import h5py
from sklearn.metrics import precision_recall_curve
from sklearn.utils.fixes import signature
from sklearn.metrics import average_precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import make_scorer, accuracy_score
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE,ADASYN
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import balanced_accuracy_score
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.externals import joblib

#把两大数据集的数据结合
def combieSetData():
    f = h5py.File("MJdect.hdf5", "r")
    x1 = f['feature_sample'][()]
    y1 = f['feature_target'][()]
    f1 = h5py.File("feature.hdf5", "r")
    x = f1['feature_sample'][()]
    y = f1['feature_target'][()]
    x2 = np.concatenate((x1, x))
    y2 = np.concatenate((y1, y))
    return x2,y2

#欠采样
def undersampleSet():
    x,y = combieSetData()
    rus = RandomUnderSampler(random_state=0)
    x, y = rus.fit_resample(x, y)

    return x,y

#过采样
def get_SMOTE_data():
    X,y = combieSetData()
    X_resampled, y_resampled = SMOTE().fit_resample(X, y);
    return X_resampled, y_resampled

#过采样
def get_ADASYN_data():
    X,y = combieSetData()
    X_resampled, y_resampled = ADASYN().fit_resample(X, y);
    return X_resampled, y_resampled

#使用网格搜索进行调参
def search_train_model():
    DT_clf = DecisionTreeClassifier()
    SVM_clf = svm.SVC()
    NB_clf = GaussianNB()


    x,y = undersampleSet()
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0)


    #决策树的网格搜索
    regressor = DT_clf

    parameters = {'max_depth': range(1, 10)}
    scoring_fnc = make_scorer(accuracy_score)

    kfold = KFold(n_splits=10)

    grid = GridSearchCV(regressor, parameters, scoring_fnc, cv=kfold,return_train_score=True)
    grid = grid.fit(X_train, y_train)
    reg = grid.best_estimator_

    print('best score: %f' % grid.best_score_)
    print('best parameters:')
    for key in parameters.keys():
        print('%s: %d' % (key, reg.get_params()[key]))

    print('test score: %f' % reg.score(X_test, y_test))

    print(pd.DataFrame(grid.cv_results_).T)

    #svm的网格搜索
    regressor_svm = SVM_clf

    parameters_svm = {'gamma': [0.001,0.01,0.1,1,10,100], 'C': [0.001,0.01,0.1,1,2,3,3.5,10,100]}
    grid_svm = GridSearchCV(regressor_svm, parameters_svm, scoring_fnc, cv=kfold,return_train_score=True)
    grid_svm = grid_svm.fit(X_train, y_train)
    reg_svm = grid_svm.best_estimator_

    print('best score: %f' % grid_svm.best_score_)
    print('best parameters:')
    for key in parameters_svm.keys():
        print('%s: %d' % (key, reg_svm.get_params()[key]))

    print('test score: %f' % reg_svm.score(X_test, y_test))


# search_train_model()

#欠采样算法
def bagtrain(clf):
    x,y = combieSetData()
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
    bc = BaggingClassifier(base_estimator=clf, random_state=0)
    bc.fit(X_train, y_train)
    y_pred = bc.predict(X_test)

    score = balanced_accuracy_score(y_test, y_pred)
    print(score)
    score = recall_score(y_test, y_pred,average = 'weighted')
    print(score)
    score = precision_score(y_test, y_pred, average='weighted')
    print(score)
    return y_test,y_pred

def train_ensemble():
    DT_clf = DecisionTreeClassifier(max_depth=9)
    DT2_clf = DecisionTreeClassifier(max_depth=7)
    SVM_clf = svm.SVC(C=100,gamma='auto')
    NB_clf = GaussianNB()

    bagtrain(DT_clf)
    bagtrain(DT2_clf)
    bagtrain(SVM_clf)
    bagtrain(NB_clf)

# train_ensemble()


#随机森林
def forest_train():
    x,y = combieSetData()
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
    brf = BalancedRandomForestClassifier(n_estimators=100, random_state=0)
    brf.fit(X_train, y_train)
    y_pred = brf.predict(X_test)
    score = balanced_accuracy_score(y_test, y_pred)
    print(score)
    fi = brf.feature_importances_
    print(fi)

    return y_test,y_pred


def train_model():
    # x,y = combieSetData()
    # rus = RandomUnderSampler(random_state=0)
    # x, y = rus.fit_resample(x, y)

    x,y = undersampleSet()
    # x,y = get_SMOTE_data()
    # x,y = get_ADASYN_data()
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0)

    # print(x.shape)
    # print(y.shape)




    DT_clf = DecisionTreeClassifier(max_depth=9)
    SVM_clf = svm.SVC(C=100, gamma='auto')
    NB_clf = GaussianNB()

    predit1 = DT_clf.fit(X_train,y_train).predict(X_test)
    predit2 = SVM_clf.fit(X_train,y_train).predict(X_test)
    predit3 = NB_clf.fit(X_train,y_train).predict(X_test)

    #决策树的可视化
    # from sklearn.tree import export_graphviz
    # export_graphviz(
    #     DT_clf,
    #     out_file="DT_clf.dot",
    #     feature_names=['risk', 'ent', 'longsize', 'commentfreq', 'depth', 'charfreq'],
    #     class_names=['benign', 'malicious'],
    #     rounded=True,
    #     filled=True
    # )

    print('accuracy is: ')
    print(balanced_accuracy_score(y_test,predit1))
    print('recall is: ')
    print(recall_score(y_test,predit1,average = 'weighted'))
    print('precision is:')
    print(precision_score(y_test,predit1,average = 'weighted'))


    print('accuracy is: ')
    print(balanced_accuracy_score(y_test,predit2))
    print('recall is: ')
    print(recall_score(y_test,predit2,average = 'weighted'))
    print('precision is:')
    print(precision_score(y_test,predit2,average = 'weighted'))

    print('accuracy is: ')
    print(balanced_accuracy_score(y_test,predit3))
    print('recall is: ')
    print(recall_score(y_test,predit3,average = 'weighted'))
    print('precision is:')
    print(precision_score(y_test,predit3,average = 'weighted'))



# train_model()


#绘制验证曲线
def val_curve(param,param_name,clf,titlename):
    x, y = undersampleSet()
    train_scores, test_scores = validation_curve(
        clf, x, y, param_name=param_name, param_range=param,
        cv=10, scoring="accuracy", n_jobs=1)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve with "+titlename)
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.ylim(0.85, 1.1)
    lw = 2

    plt.semilogx(param, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param, test_scores_mean, label="Cross-validation score",
                 color="navy",linestyle='--', lw=lw)
    plt.fill_between(param, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy",linestyle='--', lw=lw)
    plt.legend(loc="best")
    plt.show()

# val_curve()

def plot_val_curve():
    param1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    clf1 = DecisionTreeClassifier()
    val_curve(param1,'max_depth',clf1,'DecisionTree')

    param2 = [0.001,0.01,0.1,1,2,3,3.5,10,100]
    clf2 = svm.SVC(gamma='auto')
    val_curve(param2,'C',clf2,'SVM')

# plot_val_curve()



#绘制PR曲线
def pr_curve(y_test,predit,clf_name):
    average_precision = average_precision_score(y_test, predit)
    precision, recall, _ = precision_recall_curve(y_test, predit)
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})

    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.5, 1.05])
    plt.xlim([0.5, 1.0])
    plt.title(clf_name + ' Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.show()

# pr_curve()


def plot_pr_curve():
    DT_clf = DecisionTreeClassifier(max_depth=9)
    ytest,y_predit = bagtrain(DT_clf)
    pr_curve(ytest, y_predit, 'Dtree')

    SVM_clf = svm.SVC(C=100, gamma='auto')
    ytest, y_predit = bagtrain(SVM_clf)
    pr_curve(ytest, y_predit, 'SVM')

    NB_clf = GaussianNB()
    ytest, y_predit = bagtrain(NB_clf)
    pr_curve(ytest, y_predit, 'NB')

# plot_pr_curve()


def plot_under_pr():
    #获得数据集
    x,y = undersampleSet()
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0)
    #决策树
    DT_clf = DecisionTreeClassifier(max_depth=9)
    pred = DT_clf.fit(X_train, y_train).predict(X_test)
    pr_curve(y_test,pred,'Dtree')

    SVM_clf = svm.SVC(C=100, gamma='auto')
    pred = SVM_clf.fit(X_train, y_train).predict(X_test)
    pr_curve(y_test, pred,'SVM')

    NB_clf = GaussianNB()
    pred = NB_clf.fit(X_train, y_train).predict(X_test)
    pr_curve(y_test, pred,'NB')


# plot_under_pr()

#绘制学习曲线
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std,linestyle='--', alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o--', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt



def learn_curve():
    x1,y1 = combieSetData()

    X_train, X, y_train, y = train_test_split(
        x1, y1, test_size=0.3)

    DT_clf = DecisionTreeClassifier(max_depth=9)
    SVM_clf = svm.SVC(C=100,gamma='auto')
    NB_clf = GaussianNB()

    title = "Learning Curves (Naive Bayes)"
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    estimator = NB_clf
    plot_learning_curve(estimator, title, X, y, ylim=(0.94, 0.98), cv=cv, n_jobs=4)

    title = r"Learning Curves (SVM)"
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    estimator = SVM_clf
    plot_learning_curve(estimator, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)

    title = r"Learning Curves (DTree)"
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    estimator = DT_clf
    plot_learning_curve(estimator, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)

    plt.show()

# learn_curve()

#保存训练好的模型
def sava_model(clf,clf_name):
    joblib.dump(clf, clf_name)