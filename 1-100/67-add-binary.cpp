string addBinary(string a, string b) {
    int m = a.size(), n = b.size();
    reverse(a.begin(), a.end());
    reverse(b.begin(), b.end());
    string ans;
    int bit_a, bit_b, sum, carry = 0;
    for (int i = 0; i < max(m, n); ++i) {
        bit_a = (i < m ? a[i] - '0' : 0);
        bit_b = (i < n ? b[i] - '0' : 0);
        sum = (bit_a + bit_b + carry) % 2;
        carry = (bit_a + bit_b + carry) / 2;
        ans += to_string(sum);
    }
    if (carry != 0) {
        ans += "1";
    }
    reverse(ans.begin(), ans.end());
    return ans;
}