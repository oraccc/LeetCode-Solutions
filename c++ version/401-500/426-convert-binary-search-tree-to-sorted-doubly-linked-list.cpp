/*
// Definition for a Node.
class Node {
public:
    int val;
    Node* left;
    Node* right;

    Node() {}

    Node(int _val) {
        val = _val;
        left = NULL;
        right = NULL;
    }

    Node(int _val, Node* _left, Node* _right) {
        val = _val;
        left = _left;
        right = _right;
    }
};
*/
class Solution {
    Node* first = nullptr;
    Node* prev = nullptr;
public:
    Node* treeToDoublyList(Node* root) {
        if (root == nullptr) return nullptr;
        helper(root);
        first->left = prev;
        prev->right = first;

        return first;
    }

    void helper(Node *root) {
        if (root == nullptr) return;
        helper(root->left);

        if (first == nullptr) {
            first = root;
            prev = first;
        }
        else {
            prev->right = root;
            root->left = prev;
            prev = root;
        }

        helper(root->right);
    }
};