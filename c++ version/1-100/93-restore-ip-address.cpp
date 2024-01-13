vector<string> ans, list;
vector<string> restoreIpAddresses(string s) {
    if (s.size() < 4 || s.size() > 12) {
        return ans;
    }
    backtracking(0, 0, s);
    return ans;
}

void backtracking(int start, int splitCount, string &s) {
    if (start == s.size() && splitCount == 4) {
        string address = joinIP(list);
        ans.push_back(address);
    }
    if (s.size()-start < (4-splitCount) || s.size()-start > 3 * (4-splitCount)) {
        return;
    }
    for (int i = 0; i < 3; ++i) {
        if (start + i >= s.size()) {
            continue;
        }
        string str = s.substr(start, i+1);
        if (!isValidIP(str)) {
            continue;
        }
        list.push_back(str);
        backtracking(start+i+1, splitCount+1, s);
        list.pop_back();
    }
}

bool isValidIP(string &str) {
    if (str.size() > 1 && str[0] == '0') {
        return false;
    }
    int num = stoi(str);
    if (num > 255) {
        return false;
    }
    return true;
}

string joinIP(vector<string> &list) {
    string address;
    for (int i = 0; i < 3; ++i) {
        address.append(list[i]).append(".");
    }
    address.append(list[3]);
    return address;
}