bool hasAlternatingBits(int n) {
    long long m = n ^ (n>>1);
    return (m & (m+1)) == 0;
}