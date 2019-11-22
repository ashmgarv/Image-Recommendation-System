#The implementation is referred from https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
import sys
sys.path.append('../')
from random import seed
import utils
import classification.helper as helper

#Calculate gini index for the given split
#gini_index = 1 - sum(split_proportion for each class / num_instances)
def gini_index(splits, classes):
    #Getting total number of instances for the split
    instances = float(sum([len(split) for split in splits]))
    gini = 0.0

    #Calculate gini index iteratively for each split and add them up
    for split in splits:
        length = len(split)
        if not length:
            continue

        class_proportion = 0.0

        #Calculate prpoportion for each class in the group to finally calculate the gini index
        for cur_class in classes:
            arr = [data[-1] for data in split]
            p = arr.count(cur_class) / length
            class_proportion += p**2

        gini += (1.0 - class_proportion) * (length / instances)
    return gini

def create_split(index, value, data):
    left_split = []
    right_split = []
    for row in data:
        if row[index] < value:
            left_split.append(row)
        else:
            right_split.append(row)
    return left_split, right_split

#In our dataset, we should check every value on each attribute as a candiate split, and then evaluate the cost of the split
#And find the best possible split we could make
#Once we find the best split, it can be used as a node in our tree
#This approach follows an exhaustive greedy algorithm
def get_best_split(data):
    classes = set()
    for row in data:
        classes.add(row[-1])
    classes = list(classes)

    best_index, best_value, best_score, best_splits = 999, 999, 999, None
    for i in range(len(data[0]) - 1):
        for row in data:
            splits = create_split(i, row[i], data)
            gini = gini_index(splits, classes)
            if gini < best_score:
                best_index, best_value, best_score, best_splits = i, row[i], gini, splits
    return {'index' : best_index, 'value' : best_value, 'splits':best_splits}

def get_terminal_node_value(split):
    classes = [data[-1] for data in split]
    return max(set(classes), key=classes.count)

def split(node, maximum_depth, minimum_size, depth_cur_node):
    #Get values of left and right splits and delete them as they are no longer needed
    left, right = node['splits']
    del(node['splits'])

    #Check if the data has not been splitted yet
    if not left or not right:
        node['left'] = node['right'] = get_terminal_node_value(left + right)
        return

    #Check if we have reached the maximum depth
    if depth_cur_node >= maximum_depth:
        node['left'], node['right'] = get_terminal_node_value(left), get_terminal_node_value(right)
        return

    #Create a terminal node for left child if the list of rows is too small
    #Else, go on and split the node in a breadth first fashion
    if len(left) <= minimum_size:
        node['left'] = get_terminal_node_value(left)
    else:
        node['left'] = get_best_split(left)
        split(node['left'], maximum_depth, minimum_size, depth_cur_node + 1)

    #Repeat the same process for right child as well
    if len(right) <= minimum_size:
        node['right'] = get_terminal_node_value(right)
    else:
        node['right'] = get_best_split(right)
        split(node['right'], maximum_depth, minimum_size, depth_cur_node + 1)

#Build the decision tree
def build_decision_tree(training_data, maximum_depth, minimum_size):
    root_node = get_best_split(training_data)
    split(root_node, maximum_depth, minimum_size, 1)
    return root_node

def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

#Decision Tree using CART for tree induction
def decision_tree(train_data, test_data, maximum_depth, minimum_size):
    tree = build_decision_tree(train_data, maximum_depth, minimum_size)
    predictions = list()
    for row in test_data:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return(predictions)

def evaluate(dataset):
    n_folds = 3
    max_depth = 15
    min_size = 30
    scores = helper.evaluate_algorithm(dataset, decision_tree, n_folds, max_depth, min_size)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

# Print a decision tree
# def print_decision_tree(node, depth=0):
#     if isinstance(node, dict):
#         print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
#         print_decision_tree(node['left'], depth+1)
#         print_decision_tree(node['right'], depth+1)
#     else:
#         print('%s[%s]' % ((depth*' ', node)))

if __name__ == "__main__":
    seed(1)
    from feature_reduction.feature_reduction import reducer
    images, data_matrix = utils.get_all_vectors('moment')
    vectors, eigen_values, latent_vs_old = reducer(data_matrix, 10, "pca")
    dm = helper.build_labelled_matrix(vectors, images, 'aspectOfHand')
    evaluate(dm)