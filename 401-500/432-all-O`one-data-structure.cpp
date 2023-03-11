struct Bucket {
    int count;
    unordered_set<string> keys;
};

class AllOne {
    list<Bucket> buckets;
    unordered_map<string, list<Bucket>::iterator> hash;
public:
    AllOne() {
        
    }
    
    void inc(string key) {
        if (hash.find(key) == hash.end()) {
            hash[key] = buckets.insert(buckets.begin(), {0, {key}});
        }
        auto curr = hash[key], n = next(curr);
        if (n == buckets.end() || n->count > curr->count + 1) {
            n = buckets.insert(n, {curr->count + 1, {}});
        }
        n->keys.insert(key);
        hash[key] = n;

        curr->keys.erase(key);
        if (curr->keys.empty()) {
            buckets.erase(curr);
        }
    }
    
    void dec(string key) {
        auto curr = hash[key], p = prev(curr);
        hash.erase(key);
        if (curr->count > 1) {
            if (curr == buckets.begin() || p->count + 1 < curr->count) {
                p = buckets.insert(curr, {curr->count-1, {}});
            }
            p->keys.insert(key);
            hash[key] = p;
        }

        curr->keys.erase(key);
        if (curr->keys.empty()) {
            buckets.erase(curr);
        }
    }
    
    string getMaxKey() {
        if (buckets.empty()) return "";
        else {
            return *(buckets.rbegin()->keys.begin());
        }
    }
    
    string getMinKey() {
        if (buckets.empty()) return "";
        else {
            return *(buckets.begin()->keys.begin());
        }
    }
};

/**
 * Your AllOne object will be instantiated and called as such:
 * AllOne* obj = new AllOne();
 * obj->inc(key);
 * obj->dec(key);
 * string param_3 = obj->getMaxKey();
 * string param_4 = obj->getMinKey();
 */