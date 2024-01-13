class CQueue {
public:
    stack<int> in, out;
    CQueue() {
    }
    
    void appendTail(int value) {
        in.push(value);
    }
    
    int deleteHead() {
        in2out();
        if (out.empty()) return -1;
        int tmp = out.top();
        out.pop();
        return tmp;
    }

    void in2out() {
        if (out.empty()) {
            while (!in.empty()) {
                int tmp = in.top();
                in.pop();
                out.push(tmp);
            }
        }
    }
};

/**
 * Your CQueue object will be instantiated and called as such:
 * CQueue* obj = new CQueue();
 * obj->appendTail(value);
 * int param_2 = obj->deleteHead();
 */