string addStrings(string num1, string num2) {
    int len1 = num1.size(), len2 = num2.size();
    reverse(num1.begin(), num1.end());
    reverse(num2.begin(), num2.end());
    string ans;
    int carry = 0, sum = 0;
    char digit1 = 0, digit2 = 0;
    for (int i = 0; i < len1 || i < len2; ++i) {
        digit1 = (i < len1 ? num1[i] : '0');
        digit2 = (i < len2 ? num2[i] : '0');
        sum = (digit1 - '0') + (digit2 - '0') + carry;
        carry = (sum >= 10 ? 1 : 0);
        ans += to_string(sum % 10);
    }
    if (carry == 1) {
        ans += '1';
    }
    reverse(ans.begin(), ans.end());
    return ans;
}