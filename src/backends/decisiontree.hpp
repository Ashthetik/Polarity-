#ifndef DECISIONTREE_HPP
#define DECISIONTREE_HPP

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <map>

using namespace std;

class Table {
public:
    vector<string> attrName; // Attribute Names
    vector<vector<string>> data; // Data

    vector<vector<string>> attrValueList; // Attribute Value List
    void extractAttrValue() {
        attrValueList.resize(attrName.size());

        #pragma omp parallel for
        for (int i = 0; i < attrName.size(); i++) {
            map<string, int> value;
            #pragma omp parallel for
            for (int j = 0; j < data.size(); j++) {
                value[data[j][i]] = 1;
            }

            #pragma omp parallel for
            for (auto it = value.begin(); it != value.end(); it++) {
                attrValueList[i].push_back(it->first);
            }
        }
    }
};

class Node {
public:
    int critAttrIndex; // Index of the attribute
    string attrValue; // Value of the attribute
    int treeIndex; // Index of the Tree
    bool isLeaf; // Is it a leaf node?
    string label; // Label of the leaf node

    vector<int> children; // Children of the node

    Node() {
        isLeaf = false;
    }
};

class DecisionTree {
public:
    Table initTable;
    vector<Node> tree;

    DecisionTree(Table table) {
        initTable = table;
        initTable.extractAttrValue();

        Node root;
        root.treeIndex = 0;
        tree.push_back(root);
        run(initTable, 0);
    }

    string guess(vector<string> row) {
        string label = "";
        int leafnode = dfs(row, 0);
        if (leafnode != -1) {
            label = tree[leafnode].label;
        }
        return label;
    }

    int dfs(vector<string>& row, int here) {
        if (tree[here].isLeaf) {
            return here;
        }

        int critAttrIndex = tree[here].critAttrIndex;

        #pragma omp parallel for
        for (int i = 0; i < tree[here].children.size(); i++) {
            int there = tree[here].children[i];

            if (row[critAttrIndex] == tree[there].attrValue) {
                return dfs(row, there);
            }
        }

        return -1;
    }

    bool isLeafNode(Table table) {
        #pragma omp parallel for
        for (int i = 1; i < table.data.size(); i++) {
            if (table.data[0].back() != table.data[i].back()) {
                return false;
            }
        }
    }

    double getInfoD(Table table) {
        double ret = 0.0;

        int itemCount = (int)table.data.size();
        map<string, int> labelCount;

        for(int i=0;i<table.data.size();i++) {
            labelCount[table.data[i].back()]++;
        }

        for(auto iter=labelCount.begin(); iter != labelCount.end(); iter++) {
            double p = (double)iter->second/itemCount;

            ret += -1.0 * p * log(p)/log(2);
        }

        return ret;
    }

    double getInfoAttrD(Table table, int attrIndex) {
        double ret = 0.0;
        int itemCount = (int)table.data.size();

        map<string, vector<int> > attrValueMap;
        for(int i=0;i<table.data.size();i++) {
            attrValueMap[table.data[i][attrIndex]].push_back(i);
        }

        for(auto iter=attrValueMap.begin(); iter != attrValueMap.end(); iter++) {
            Table nextTable;
            for(int i=0;i<iter->second.size(); i++) {
                nextTable.data.push_back(table.data[iter->second[i]]);
            }
            int nextItemCount = (int)nextTable.data.size();

            ret += (double)nextItemCount/itemCount * getInfoD(nextTable);
        }

        return ret;
    }

    double getGain(Table table, int attrIndex) {
        return getInfoD(table)-getInfoAttrD(table, attrIndex);
    }

    double getSplitInfoAttrD(Table table, int attrIndex) {
        double ret = 0.0;

        int itemCount = (int)table.data.size();

        map<string, vector<int> > attrValueMap;
        for(int i=0;i<table.data.size();i++) {
            attrValueMap[table.data[i][attrIndex]].push_back(i);
        }

        for(auto iter=attrValueMap.begin(); iter != attrValueMap.end(); iter++) {
            Table nextTable;
            for(int i=0;i<iter->second.size(); i++) {
                nextTable.data.push_back(table.data[iter->second[i]]);
            }
            int nextItemCount = (int)nextTable.data.size();

            double d = (double)nextItemCount/itemCount;
            ret += -1.0 * d * log(d) / log(2);
        }

        return ret;
    }

    double getGainRatio(Table table, int attrIndex) {
        return getGain(table, attrIndex) / getSplitInfoAttrD(table, attrIndex);
    }

    int getSelectedAttribute(Table table) {
        int maxAttrIndex = -1;
        double maxAttrValue = 0.0;

        // except label
        for(int i=0; i< initTable.attrName.size()-1; i++) {
            if(maxAttrValue < getGainRatio(table, i)) {
                maxAttrValue = getGainRatio(table, i);
                maxAttrIndex = i;
            }
        }

        return maxAttrIndex;
    }

    pair<string, int> getMajority(Table table) {
        string majorityLabel = "";
        int majorityCount = 0;

        map<string, int> labelCount;
        #pragma omp parallel for
        for (int i = 0; i < table.data.size(); i++) {
            labelCount[table.data[i].back()]++;

            if (labelCount[table.data[i].back()] > majorityCount) {
                majorityCount = labelCount[table.data[i].back()];
                majorityLabel = table.data[i].back();
            }
        }

        return make_pair(majorityLabel, majorityCount);
    }

    void run(Table table, int nodeIndex) {
        if (isLeafNode(table) == true) {
            tree[nodeIndex].isLeaf = true;
            tree[nodeIndex].label = table.data[0].back();
            return;
        }

        int selectedAttrIndex = getSelectedAttribute(table);

        map<string, vector<int>> attrViewMap;
        
        #pragma omp parallel for
        for (int i = 0; i < table.data.size(); i++) {
            attrViewMap[table.data[i][selectedAttrIndex]].push_back(i);
        }

        tree[nodeIndex].critAttrIndex = selectedAttrIndex;

        pair<string, int> majority = getMajority(table);
        if ((double)majority.second / table.data.size() > 0.8) {
            tree[nodeIndex].isLeaf = true;
            tree[nodeIndex].label = majority.first;
            return;
        }

        #pragma omp parallel for
        for (int i = 0; i < initTable.attrValueList[selectedAttrIndex].size(); i++) {
            Node nextNode;
            nextNode.treeIndex = tree.size();
            nextNode.attrValue = initTable.attrValueList[selectedAttrIndex][i];
            tree.push_back(nextNode);

            tree[nodeIndex].children.push_back(nextNode.treeIndex);

            Table nextTable;
            for (int j = 0; j < attrViewMap[nextNode.attrValue].size(); j++) {
                vector<string> newRow;
                for (int k = 0; k < table.data[attrViewMap[nextNode.attrValue][j]].size(); k++) {
                    if (k != selectedAttrIndex) {
                        newRow.push_back(table.data[attrViewMap[nextNode.attrValue][j]][k]);
                    }
                }
                nextTable.data.push_back(newRow);
            }

            run(nextTable, nextNode.treeIndex);
        }
    }
};

#endif