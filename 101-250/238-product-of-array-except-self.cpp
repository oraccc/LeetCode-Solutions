// Space Complexity: O(N)
vector<int> productExceptSelf(vector<int>& nums) {
    int n = nums.size();
    vector<int> left(n, 1), right(n, 1);
    for (int i = 1; i < n; ++i) {
        left[i] = left[i-1] * nums[i-1];
    }
    for (int j = n-2; j >= 0; --j) {
        right[j] = right[j+1] * nums[j+1];
    }

    vector<int> ans(n, 1);
    for (int k = 0; k < n; ++k) {
        ans[k] = left[k] * right[k];
    }
    return ans;
}

// Space Complexity: O(1)
vector<int> productExceptSelf(vector<int>& nums) {
    int n = nums.size();
    vector<int> ans(n, 1);
    for (int i = 1; i < n; ++i) {
        ans[i] = ans[i-1] * nums[i-1];
    }
    int r = 1;
    for (int j = n-1; j >= 0; --j) {
        ans[j] = ans[j] * r;
        r *= nums[j];
    }

    return ans;
}