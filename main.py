class Node:
    threshold = 0
    left_node = None
    right_node = None

    def __init__(self, left_node, right_node, threshold):
        self.left_node = left_node
        self.right_node = right_node
        self.threshold = threshold

    def split(self, dataset):
        pass


class Leaf:
    className = None


class Tree:
    def __init__(self, dataset, min_recursion):
        pass

    def classify(self):
        pass

    def grow_tree(self):
        pass

    def get_gini(self, data):
        pass

    def optimize_threshold(self):
        pass

    def stop_condition(self):
        pass


if __name__ == '__main__':
    pass
