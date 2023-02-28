class TrieNode {
public:
    TrieNode* child[26];
    bool isWord;
    TrieNode(): isWord(false) {
        for (int i = 0; i < 26; ++i) {
            child[i] = nullptr;
        }
    }
};

class Trie {
    TrieNode *root;
public:
    Trie(): root(new TrieNode()){ 
    }
    
    void insert(string word) {
        TrieNode *temp = root;
        for (int i = 0; i < word.size(); ++i) {
            if (!temp->child[word[i]-'a']) {
                temp->child[word[i]-'a'] = new TrieNode();
            }
            temp = temp->child[word[i]-'a'];
        }
        temp->isWord = true;
    }
    
    bool search(string word) {
        TrieNode *temp = root;
        for (int i = 0; i < word.size(); ++i) {
            temp = temp->child[word[i]-'a'];
            if (!temp) return false;
        }
        return temp->isWord;
    }
    
    bool startsWith(string prefix) {
        TrieNode *temp = root;
        for (int i = 0; i < prefix.size(); ++i) {
            temp = temp->child[prefix[i]-'a'];
            if (!temp) return false;
        }
        return true;
    }
};