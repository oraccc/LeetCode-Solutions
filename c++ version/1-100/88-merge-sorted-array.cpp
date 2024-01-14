    2void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
    int end = m + n - 1;
    int firstEnd = m - 1, secondEnd = n - 1;
    while (firstEnd >= 0 && secondEnd >= 0) {
        if (nums1[firstEnd] > nums2[secondEnd]) {
            nums1[end] = nums1[firstEnd];
            --firstEnd;
        }
        else {
            nums1[end] = nums2[secondEnd];
            --secondEnd;
        }
        --end;
    }

    while (secondEnd >= 0) {
        nums1[end] = nums2[secondEnd];
        --end;
        --secondEnd;
    }
}