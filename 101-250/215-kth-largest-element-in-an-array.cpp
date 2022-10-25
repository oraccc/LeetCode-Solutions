int findKthLargest(vector<int>& nums, int k) {
    sort(nums.begin(), nums.end());
    return nums[nums.size()-k];
}

// Another Solution : Quick Sort (Time complexity : O(nlogn)) 
// C++ 的sort函数就是实现了快速排序，现在手动实现一遍

int partition(vector<int> &nums, int left, int right) {
    int i = left, j = right + 1;
    int k = nums[left];

    while (true) {
        while (nums[++i] < k)
            if (i == right) break;
        while (nums[--j] > k)
            if (j == left) break;
        if (i >= j) break;
        swap(nums[i], nums[j]);
    }

    swap(nums[left], nums[j]);
    return j;
}

void sort(vector<int> &nums, int left, int right) {
    if (right <= left) return;
    int j = partition(nums, left, right);
    sort(nums, left, j - 1);
    sort(nums, j + 1, right);
}

int findKthLargest(vector<int>& nums, int k) {
    int i = 0, j = nums.size() - 1;
    sort(nums, i, j);
    return nums[nums.size()-k];
}