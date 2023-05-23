class MaxQueue {
    deque<int> q;
    deque<int> maxQ;
public:
    MaxQueue() {
    }
    
    int max_value() {
        if (!q.empty()) return maxQ.front();
        else return -1;
    }
    
    void push_back(int value) {
        q.push_back(value);
        while (!maxQ.empty() && maxQ.back() < value) {
            maxQ.pop_back();
        }
        maxQ.push_back(value);
    }
    
    int pop_front() {
        if (q.empty()) return -1;
        int v = q.front();
        q.pop_front();
        if (maxQ.front() == v) {
            maxQ.pop_front();
        }
        return v;
    }
};

/**
 * Your MaxQueue object will be instantiated and called as such:
 * MaxQueue* obj = new MaxQueue();
 * int param_1 = obj->max_value();
 * obj->push_back(value);
 * int param_3 = obj->pop_front();
 */