int mySqrt(int x) {
    if (x == 0) return 0;
    int l = 1, r = x;
    int mid, sqrt;
    while (l <= r) {
        mid = l + (r - l) / 2;
        sqrt = x / mid;
        if (sqrt == mid) return mid;
        else if (sqrt < mid) r = mid - 1;
        else l = mid + 1;
    }
    return r;
}