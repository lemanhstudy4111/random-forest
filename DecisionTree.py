import math
import numpy as np
from Utils import Bootstrap, StratifiedValidation, Performance

class Node:
    def __init__(self, ft_split=None, data=None, isLeaf=False, threshold=None):
        self.id = id(self)
        self.ft_split = ft_split
        self.data = data
        self.threshold = threshold
        self.isLeaf = isLeaf
        self.children = []

class Graph:
    def __init__(self):
        self.root = None
    
    def add_root(self, node_id):
        self.root = node_id
        return
    
    def add_edge(self, parent, child, edge_value):
        parent.children.append((child, edge_value))
        return
    
    def isEmpty(self):
        return len(self.adj_graph) == 0
    
    def get_next_node(self, node, branch):
        condition = False
        for child, edge_value in node.children:
            if edge_value.isdigit():
                condition = np.int64(edge_value) == branch
            if edge_value == branch or condition:
                return child
        raise Exception("No node found")
        
class DecisionTree:
    def __init__(self, dataset):
        self.dtree = Graph()
        self.dataset = dataset
        self.num_atts = len(self.dataset.dtype.names) - 1
        self.label_name = "label"
        
    #TODO: See how many uniques return    
    def count_majority(self, dataset):
        uniques, counts = np.unique(dataset[self.label_name], return_counts=True)
        max_values = np.argwhere(counts == np.amax(counts)).flatten()
        rng = np.random.default_rng()       
        return uniques[rng.choice(max_values)]
        
    #TODO: refactor
    def count_class(self, dataset, col_name, isCategorical):
        res = []
        if isCategorical:
            unique_values = np.unique(dataset[col_name])
            for val in unique_values:
                mask = dataset[col_name] == val
                label0 = np.sum((mask) & (dataset[self.label_name] == 0))
                label1 = np.sum((mask) & (dataset[self.label_name] == 1))
                res.append([label0, label1])
        else:
            mean_col = np.average(dataset[col_name])
            mask_less = dataset[col_name] < mean_col
            mask_more = dataset[col_name] >= mean_col
            masks = [mask_less, mask_more]
            for mask in masks:
                label0 = np.sum((mask) & (dataset[self.label_name] == 0))
                label1 = np.sum((mask) & (dataset[self.label_name] == 1))
                res.append([label0, label1])
        return res
    
    def info_gain(self, class_count_arr):
        counts = np.array(class_count_arr)
        total_cnt = np.sum(counts)
        if total_cnt == 0:
            return 0
        probs = np.sum(counts, axis=0) / total_cnt
        probs = probs[probs > 0]  # Avoid log(0)
        parent_entropy = -np.sum(probs * np.log2(probs))
        weighted_children_entropy = 0
        for child_counts in counts:
            total_child_cnt = np.sum(child_counts)
            if total_child_cnt == 0:
                continue
            child_probs = child_counts / total_child_cnt
            child_probs = child_probs[child_probs > 0]
            child_entropy = -np.sum(child_probs * np.log2(child_probs))
            weighted_children_entropy += (total_child_cnt / total_cnt) * child_entropy
        return parent_entropy - weighted_children_entropy
    
    def get_att_list(self):
        col_indices = np.random.choice(self.num_atts, size=math.floor(np.sqrt(self.num_atts)), replace=False)
        return col_indices
    
    def check_stop(self, split_dataset, n, curr_depth, max_depth):
        # All samples same class
        if np.all(split_dataset[self.label_name] == 1) or np.all(split_dataset[self.label_name] == 0):
            return True
        # Number of samples less than n
        # if split_dataset.shape[0] < n:
        #     return True
        # Check depthdepth
        if max_depth is not None and curr_depth >= max_depth:
            return True
        return False
    
    def create_leaf_node(self, split_dataset, node):
        node.isLeaf = True
        node.data = self.count_majority(split_dataset)
        return node
    
    def find_best_infogain(self, split_dataset):
        att_list = np.array(self.dataset.dtype.names)[self.get_att_list()]
        max_info_gain = -1
        best_attr = ""
        is_categorical = False
        
        for attribute in att_list:
            current_is_categorical = attribute[-3:] == "cat"
            class_count_arr = self.count_class(split_dataset, attribute, current_is_categorical)
            att_info_gain = self.info_gain(class_count_arr)
            
            if att_info_gain > max_info_gain:
                max_info_gain = att_info_gain
                best_attr = attribute
                is_categorical = current_is_categorical
        
        return best_attr, is_categorical
    
    def process_child_node(self, split_dataset, n, parent_node, row_part, edge_value, curr_depth, max_depth,threshold=None):
        child_node = Node()
                
        if row_part.shape[0] == 0:
            child_node.data = self.count_majority(split_dataset)
            child_node.isLeaf = True
        else:
            if threshold is not None:
                child_node.threshold = threshold
            child_node = self.create_decision_tree(row_part, n, curr_depth, max_depth)
        self.dtree.add_edge(parent_node, child_node, edge_value)
    
    def handle_categorical_split(self, split_dataset, n, parent_node, max_gain_att, curr_depth, max_depth):
        values_list = np.unique(self.dataset[max_gain_att])
        for value in values_list:
            row_part = split_dataset[np.where(split_dataset[max_gain_att] == value)]
            self.process_child_node(split_dataset, n, parent_node, row_part, str(value), curr_depth, max_depth)
    
    def handle_numerical_split(self, split_dataset, n, parent_node, max_gain_att, curr_depth, max_depth):
        threshold = np.average(split_dataset[max_gain_att])
        lower_part = split_dataset[split_dataset[max_gain_att] <= threshold]
        upper_part = split_dataset[split_dataset[max_gain_att] > threshold]
        
        self.process_child_node(split_dataset, n, parent_node, lower_part, f"<={threshold:.2f}", curr_depth, max_depth, threshold=threshold)
        self.process_child_node(split_dataset, n, parent_node, upper_part, f">{threshold:.2f}", curr_depth, max_depth, threshold=threshold)
        
    
    def create_decision_tree(self, split_dataset, n, curr_depth, max_depth):
        new_node = Node()
        # #stop criteria
        if self.check_stop(split_dataset, n, curr_depth, max_depth) == True:
            return self.create_leaf_node(split_dataset, new_node)
        max_gain_att, isCategorical = self.find_best_infogain(split_dataset)
        new_node.ft_split = max_gain_att
        new_node.isLeaf = False
        # increment depth
        next_depth = curr_depth + 1
        # create a branch for each value of the split node
        if isCategorical:
            self.handle_categorical_split(split_dataset, n, new_node, max_gain_att, next_depth, max_depth)
        else:
            new_node.threshold = np.average(split_dataset[new_node.ft_split])
            self.handle_numerical_split(split_dataset, n, new_node, max_gain_att, next_depth, max_depth)
        return new_node
    
    def train_decision_tree(self):
        root = self.create_decision_tree(self.dataset, 100, curr_depth=0, max_depth=5)
        self.dtree.add_root(root)
        return
    
    def predict(self, drow):
        curr_node = self.dtree.root
        while not curr_node.isLeaf:
            isCategorical = curr_node.ft_split[-3:] == "cat"
            branch = drow[str(curr_node.ft_split)]
            if not isCategorical:
                branch = f"<={curr_node.threshold:.2f}" if branch <= curr_node.threshold else f">{curr_node.threshold:.2f}"
            curr_node = self.dtree.get_next_node(curr_node, branch)
        return curr_node.data
        
    
class RandomForest:
    def __init__(self, dataset, ntree=1):
        self.ds = np.array(dataset)
        self.ntree = ntree
        self.dtrees = []
    
    def get_final_result(self, results):
        uniques, counts = np.unique(np.array(results), return_counts=True)
        return uniques[np.argmax(counts)]
        
    def train_random_forest(self):
        bootstrap_util = Bootstrap(self.ds)
        for i in range(self.ntree):
            dtree_ds = bootstrap_util.sample()
            dtree = DecisionTree(dtree_ds)
            dtree.train_decision_tree()
            self.dtrees.append(dtree)
    
    def predict(self, eval_data):
        all_predictions = []
        for row in eval_data:
            results = []
            for dtree in self.dtrees:
                res = dtree.predict(row)
                results.append(res)
            final_res = self.get_final_result(results)
            all_predictions.append(final_res)
        return all_predictions                
            
            
    
    
        

    
    
    
    