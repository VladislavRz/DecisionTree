import numpy as np
import pandas
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# [0:'setosa' 1:'versicolor' 2:'virginica']


class Node:
    threshold = 0
    feature = 0
    left_node = None
    right_node = None
    is_leaf = False
    decision = None

    def get_decision(self, el):
        if self.is_leaf:
            return self.decision

        elif el[self.feature] < self.threshold:
            return self.left_node.get_decision(el)

        else:
            return self.right_node.get_decision(el)


class Tree:
    def __init__(self, depth=50, samples=2):
        self.max_depth = depth
        self.min_samples = samples
        self.root_node = Node()

    def classify(self):
        pass

    def grow_tree(self, data, target, node, depth=0):
        if any(self.stop_condition(target, depth)):
            classes, counts = np.unique(target, return_counts=True)
            decision = classes[np.where(counts == max(counts))][0]
            node.is_leaf = True
            node.decision = decision
            return

        feature, threshold = self.optimize_threshold(data, target)
        left_tar, right_tar, left_data, right_data = self.split(data, target, threshold, feature)

        node.left_node = Node()
        node.right_node = Node()
        node.threshold = threshold
        node.feature = feature
        self.grow_tree(left_data, left_tar, node.left_node, depth + 1)
        self.grow_tree(right_data, right_tar, node.right_node, depth + 1)

        return

    def get_gini(self, data):
        _, counts_classes = np.unique(data, return_counts=True)
        squared_probabilities = np.square(counts_classes / data.size)
        gini = 1 - sum(squared_probabilities)

        return gini

    def get_gini_split(self, left_tar, right_tar):
        total_size = left_tar.size + right_tar.size
        left_proportion = left_tar.size/total_size
        right_proportion = right_tar.size/total_size

        gini_split = left_proportion * self.get_gini(left_tar) + right_proportion * self.get_gini(right_tar)
        return gini_split

    def optimize_threshold(self, data, target):
        best_threshold = None
        best_feature = None
        min_gini = np.inf
        transp_data = data.transpose()

        for i, batch in enumerate(transp_data):
            for unique_el in np.unique(batch):
                left_tar, right_tar, _, _ = self.split(data, target, unique_el, i)
                gini_split = self.get_gini_split(left_tar, right_tar)
                if gini_split < min_gini:
                    min_gini = gini_split
                    best_threshold = unique_el
                    best_feature = i

        return best_feature, best_threshold

    def split(self, data, target, threshold, feature):
        left_target = []
        right_target = []
        left_data = []
        right_data = []

        for i, el in enumerate(data):
            if el[feature] < threshold:
                left_target.append(target[i])
                left_data.append((data[i]))
            else:
                right_target.append(target[i])
                right_data.append(data[i])

        return np.array(left_target), np.array(right_target), np.array(left_data), np.array(right_data)

    def stop_condition(self, target, depth):
        return np.unique(target).size == 1, depth == self.max_depth, target.size < self.min_samples


if __name__ == '__main__':
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target,
                                                        train_size=0.67, random_state=42)
    tree = Tree()
    tree.grow_tree(x_train, y_train, tree.root_node)

    right_decisions = []
    tree_decisions = []

    for i in y_test:
        right_decisions.append(str(iris.target_names[i]))

    for i in x_test:
        tree_decisions.append(str(iris.target_names[tree.root_node.get_decision(i)]))

    print(f"|{"Правильные решения":^20}|{"Решения дерева":^20}|")
    for i in range(y_test.size):
        print(f"|{right_decisions[i]:^20}|{tree_decisions[i]:^20}|")




