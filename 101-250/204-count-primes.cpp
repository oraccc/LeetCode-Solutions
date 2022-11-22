int countPrimes(int n) {
    if (n <= 2) return 0;
    vector<bool> prime(n, true);
    int i = 3, sqrtn = sqrt(n), count = n/2;
    while (i <= sqrtn) {
        for (int j = i * i; j < n; j += 2*i) {
            if (prime[j]) {
                --count;
                prime[j] = false;
            }
        }
        do {
            i += 2;
        } while(i <= sqrtn && !prime[i]);
    }
    return count;
}