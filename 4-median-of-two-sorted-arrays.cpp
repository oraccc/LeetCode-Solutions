double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {

    if (nums1.size() > nums2.size())
    {
        return findMedianSortedArrays(nums2, nums1);
    }
    int m = nums1.size();
    int n = nums2.size();
    int bin = (m + n + 1) / 2;
    int left = 0;
    int right = m;
    while (left < right)
    {
        int a = (left + right + 1) / 2; 
        int b = bin - a;   
        if (nums1[a - 1] < nums2[b])
        {
            left = a;
        }
        else
        {
            right = a - 1;
        }
    }
    int i = right;

    int j = bin - i;

    int ileft = (i == 0 ? INT_MIN : nums1[i - 1]);
    int iright = (i == m ? INT_MAX : nums1[i]);
    int jleft = (j == 0 ? INT_MIN : nums2[j - 1]);
    int jright = (j == n ? INT_MAX : nums2[j]);

    if ((m + n) % 2 == 1)
    {
        return max(ileft, jleft);
    }
    else
    {
        return (max(ileft, jleft) + min(iright, jright)) / 2.0;
    }

}