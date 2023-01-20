stack<int> in, out;
MyQueue() {
    
}

void push(int x) {
    in.push(x);
}

int pop() {
    in2out();
    int tmp = out.top();
    out.pop();
    return tmp;
}

int peek() {
    in2out();
    return out.top();
}

bool empty() {
    return in.empty() && out.empty();
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