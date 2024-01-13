vector<int> plusOne(vector<int>& digits) {
    list<int> ans;
    int n = digits.size();
    ++digits[n-1];
    int carry = 0;
    int val = 0;
    for (int i = n-1; i >= 0; --i) {
        val = (digits[i] + carry) % 10;
        carry = (digits[i] + carry) > 9 ? 1 : 0;
        ans.insert(ans.begin(), val);
    }
    if (carry == 1) ans.insert(ans.begin(), 1);
    return vector<int>(ans.begin(), ans.end());
}