// divide and conquer
vector<int> diffWaysToCompute(string expression) {
    vector<int> result;
    for (int i = 0; i < expression.size(); ++i) {
        char c = expression[i];
        if (c == '+' || c == '-' || c == '*') {
            vector<int> left = diffWaysToCompute(expression.substr(0, i));
            vector<int> right = diffWaysToCompute(expression.substr(i+1));
            for (const int &l : left) {
                for (const int &r : right) {
                    switch(c) {
                        case '+': result.push_back(l + r); break;
                        case '-': result.push_back(l - r); break;
                        case '*': result.push_back(l * r); break;
                    }
                }
            }
        }
    }
    if (result.empty()) {
        result.push_back(stoi(expression));
    }
    return result;
}

//add memoization
unordered_map<string, vector<int>> memo;
vector<int> diffWaysToCompute(string expression) {
    auto it = memo.find(expression);
    if (it != memo.end()) {
        return it -> second;
    }
    vector<int> result;
    for (int i = 0; i < expression.size(); ++i) {
        char c = expression[i];
        if (c == '+' || c == '-' || c == '*') {
            vector<int> left = diffWaysToCompute(expression.substr(0, i));
            vector<int> right = diffWaysToCompute(expression.substr(i+1));
            for (const int &l : left) {
                for (const int &r : right) {
                    switch(c) {
                        case '+': result.push_back(l + r); break;
                        case '-': result.push_back(l - r); break;
                        case '*': result.push_back(l * r); break;
                    }
                }
            }
        }
    }
    if (result.empty()) {
        result.push_back(stoi(expression));
    }
    memo.insert({expression, result});
    return result;
}