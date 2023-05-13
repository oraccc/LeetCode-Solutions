vector<int> printNumbers(int n) {
    vector<int> res;
    if (n == 0) return res;
    for (int i=1,max=pow(10,n);i<max;i++)
    {
        res.push_back(i);
    }
    return res;

}