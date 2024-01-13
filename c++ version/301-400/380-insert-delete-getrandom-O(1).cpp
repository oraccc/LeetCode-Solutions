class RandomizedSet {
    vector<int> set;
    unordered_map<int, int> hash;
public:
    RandomizedSet() {
        
    }
    
    bool insert(int val) {
        if (hash.find(val) != hash.end()) return false;
        set.push_back(val);
        hash[val] = set.size()-1;
        return true;
    }
    
    bool remove(int val) {
        if (hash.find(val) == hash.end()) return false;
        int index = hash[val];
        hash[set.back()] = index;
        set[index] = set.back();
        set.pop_back();
        hash.erase(val);
        return true;
    }
    
    int getRandom() {
        return set[rand() % set.size()];
    }
};