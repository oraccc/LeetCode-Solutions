string convertToBase7(int num) {
    if (num == 0) return "0";
    bool isNegative = num < 0;
    if (isNegative) {
        num = -num;
    }
    string ans;
    while (num) {
        int a = num / 7, b = num % 7;
        ans = to_string(b) + ans;
        num = a;
    }
    if (isNegative) {
        return '-' + ans;
    }
    else return ans;
}