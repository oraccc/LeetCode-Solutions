class LRUCache {
    int size;
    unordered_map<int, list<pair<int, int>>::iterator> hash;
    list<pair<int, int>> cache;
public:
    LRUCache(int capacity) {
        size = capacity;
    }
    
    int get(int key) {
        auto it = hash.find(key);
        if (it == hash.end()) return -1;
        cache.splice(cache.begin(), cache, it->second);
        return it->second->second;
    }
    
    void put(int key, int value) {
        auto it = hash.find(key);
        if (it != hash.end()) {
            it->second->second = value;
            cache.splice(cache.begin(), cache, it->second);
            return;
        }
        cache.insert(cache.begin(), make_pair(key, value));
        hash[key] = cache.begin();
        if (cache.size() > size) {
            hash.erase(cache.back().first);
            cache.pop_back();
        }
    }
};
