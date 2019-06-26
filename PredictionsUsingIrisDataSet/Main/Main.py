import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
from IPython.display import Image
from sklearn.externals.six import StringIO
import pydot

iris = load_iris()
idx = [0,50,100]

# training data
trainTarget = np.delete(iris.target, idx)
trainData = np.delete(iris.data, idx,axis=0)

# testing data
testTarget = iris.target[idx]
testData = iris.data[idx]

clf = tree.DecisionTreeClassifier()
clf.fit(trainData, trainTarget)

print (testTarget)
print(clf.predict(testData))

# printing the decision tree
dotData = StringIO()
dot_data = tree.export_graphviz(clf, out_file=dotData,
                    feature_names=iris.feature_names,
                    class_names=iris.target_names,
                    filled=True, rounded=True,
                    impurity=False)
graph = pydot.graph_from_dot_data(dotData.getvalue())
graph[0].write_pdf("iris.pdf")
