# Decision-Tree-Classifier
# Decision Tree Classifier

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def gini_index(groups, classes):
    total_instances = sum(len(group) for group in groups)
    gini = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            proportion = [row[-1] for row in group].count(class_val) / size
            score += proportion * proportion
        gini += (1.0 - score) * (size / total_instances)
    return gini

def split_dataset(dataset, feature, threshold):
    left = [row for row in dataset if row[feature] < threshold]
    right = [row for row in dataset if row[feature] >= threshold]
    return left, right

def get_best_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    best_index, best_value, best_score, best_groups = 999, 999, float('inf'), None
    for feature in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = split_dataset(dataset, feature, row[feature])
            gini = gini_index(groups, class_values)
            if gini < best_score:
                best_index, best_value, best_score, best_groups = feature, row[feature], gini, groups
    return best_index, best_value, best_groups

def create_tree(dataset, max_depth, min_size, depth=0):
    left, right = dataset
    if not left or not right:
        return Node(value=most_common_class(left + right))
    if depth >= max_depth or len(left) < min_size or len(right) < min_size:
        return Node(value=most_common_class(left + right))
    feature, threshold, groups = get_best_split(dataset)
    left_node = create_tree(groups[0], max_depth, min_size, depth + 1)
    right_node = create_tree(groups[1], max_depth, min_size, depth + 1)
    return Node(feature, threshold, left_node, right_node)

def most_common_class(dataset):
    classes = {}
    for row in dataset:
        label = row[-1]
        if label not in classes:
            classes[label] = 0
        classes[label] += 1
    return max(classes.items(), key=lambda x: x[1])[0]

def predict(node, row):
    if node.value is not None:
        return node.value
    if row[node.feature] < node.threshold:
        return predict(node.left, row)
    else:
        return predict(node.right, row)

def decision_tree_classifier(dataset, max_depth, min_size):
    root = create_tree(dataset, max_depth, min_size)
    return root
