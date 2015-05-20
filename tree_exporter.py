import json

import pandas as pd
import pydot
from sklearn.datasets import load_iris
from sklearn.externals.six import StringIO
from sklearn import tree


class TreeExporter():
    model = None
    feature_names = []

    def __init__(self, model, feature_names):
        self.model = model
        self.tree_type = model.__class__.__name__
        self.feature_names = feature_names

    def get_best_leaves_indices(self, class_index, n_branches):
        """
        Searches and returns branches with best conversion rate
        if classification tree, leaves with highest class ratio is returned (class_index)
        if regression tree, leaves with highest value are returned
        """
        if self.tree_type == 'DecisionTreeClassifier':
            tree_values = pd.Series([leaf[0][class_index] / sum(leaf[0]) for leaf in self.model.tree_.value])
        elif self.tree_type == 'DecisionTreeRegressor':
            tree_values = pd.Series(self.model.tree_.value)
        else:
            raise Exception('Not a Decision Tree!')
        tree_values.sort(ascending=False)
        return list(tree_values[:n_branches].index)

    def get_node(self, index, full_info=False):
        """gets data from tree for a single node"""
        feature_ind = self.model.tree_.feature[index]
        feature = self.feature_names[feature_ind]
        threshold = self.model.tree_.threshold[index]
        node = {
            'feature': feature,
            'threshold': threshold,
        }
        if full_info:
            node['impurity'] = self.model.tree_.impurity[index]
            node['n_node_samples'] = self.model.tree_.n_node_samples[index]
        return node

    def get_parents(self, leaf_index):
        """returns a list of nodes of parents in top-to-bottom order"""
        children = pd.DataFrame()
        children['left'] = pd.Series(self.model.tree_.children_left)
        children['right'] = pd.Series(self.model.tree_.children_right)
        list_of_filters = []
        index = leaf_index
        for pair in children[::-1].iterrows():
            node_index = pair[0]
            left = pair[1].left
            right = pair[1].right
            if index in [left, right]:
                node = self.get_node(node_index)
                if index == left:
                    node['operator'] = '<='
                else:
                    node['operator'] = '>'
                index = node_index
                list_of_filters.append(node)

        reversed_list_of_filters = list_of_filters[::-1]
        return reversed_list_of_filters

    def get_best_branches(self, class_index=1, n_branches=5, print_out=True, filename=None):
        """returns branches that give the highest score"""
        best_leaves_indices = self.get_best_leaves_indices(class_index=class_index, n_branches=n_branches)
        best_branches = [self.get_parents(index) for index in best_leaves_indices]
        if print_out:
            print json.dumps(
                best_branches,
                indent=4
            )
        if filename:
            with open(filename, 'w') as outfile:
                json.dump(best_branches, outfile)

    # TODO: move to viz.py
    def export_png(self, filename='graph.png'):
        """exports png of tree using graphviz"""
        dot_data = StringIO()
        tree.export_graphviz(self.model, out_file=dot_data, feature_names=self.feature_names)
        graph = pydot.graph_from_dot_data(dot_data.getvalue())
        graph.write_png(filename)


if __name__ == "__main__":
    clf = tree.DecisionTreeClassifier(random_state=0)
    clf.fit(load_iris().data, load_iris().target)
    feature_names = load_iris().feature_names
    tree_exporter = TreeExporter(clf, feature_names)
    tree_exporter.get_best_branches()
    tree_exporter.export_png()