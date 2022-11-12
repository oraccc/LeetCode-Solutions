class MedianFinder {
public:
    MedianFinder() {
        count = 0;
    }
    
    void addNum(int num) {
        if (maxQ.empty() || num <= maxQ.top()) {
            maxQ.push(num);
        }
        else {
            minQ.push(num);
        }
        ++count;
        if (minQ.size() > maxQ.size()) {
            maxQ.push(minQ.top());
            minQ.pop();
        }
        else if (maxQ.size() > minQ.size() + 1) {
            minQ.push(maxQ.top());
            maxQ.pop();
        }
    }
    
    double findMedian() {
        if (count % 2) {
            return maxQ.top();
        }
        else {
            return (maxQ.top() + minQ.top()) / 2.0;
        }
    }
private:
    priority_queue<int, vector<int>, less<int>> maxQ;
    priority_queue<int, vector<int>, greater<int>> minQ;
    int count;
};

/**
 * Your MedianFinder object will be instantiated and called as such:
 * MedianFinder* obj = new MedianFinder();
 * obj->addNum(num);
 * double param_2 = obj->findMedian();
 */