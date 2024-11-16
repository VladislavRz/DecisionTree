import numpy as np


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

    def train(self, data, target):

        self.grow_tree(data, target, self.root_node)

        return

    def grow_tree(self, data, target, node, depth=0):
        if any(self.stop_condition(target, depth)):
            self.create_leaf(target, node)
            return

        feature, threshold = self.optimize_threshold(data, target)
        left_tar, right_tar = self.split_target(data, target, threshold, feature)
        left_data, right_data = self.split_data(data, threshold, feature)

        self.create_nodes(node, feature, threshold)

        self.grow_tree(left_data, left_tar, node.left_node, depth + 1)
        self.grow_tree(right_data, right_tar, node.right_node, depth + 1)

        return

    def create_leaf(self, target, node):
        classes, counts = np.unique(target, return_counts=True)
        decision = classes[np.where(counts == max(counts))][0]
        node.is_leaf = True
        node.decision = decision

        return

    def create_nodes(self, node, feature, threshold):
        node.left_node = Node()
        node.right_node = Node()
        node.threshold = threshold
        node.feature = feature

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
                left_tar, right_tar = self.split_target(data, target, unique_el, i)
                gini_split = self.get_gini_split(left_tar, right_tar)
                if gini_split < min_gini:
                    min_gini = gini_split
                    best_threshold = unique_el
                    best_feature = i

        return best_feature, best_threshold

    def split_target(self, data, target, threshold, feature):
        left_target = [target[i] for i, el in enumerate(data) if el[feature] < threshold]
        right_target = [target[i] for i, el in enumerate(data) if el[feature] >= threshold]

        return np.array(left_target), np.array(right_target)

    def split_data(self, data, threshold, feature):
        left_data = [el for el in data if el[feature] < threshold]
        right_data = [el for el in data if el[feature] >= threshold]

        return np.array(left_data), np.array(right_data)

    def stop_condition(self, target, depth):
        return np.unique(target).size == 1, depth == self.max_depth, target.size < self.min_samples
