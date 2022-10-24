void sortColors(vector<int>& nums) {
    int zero = 0, two = nums.size() - 1;
    int curr = 0;
    while (curr <= two) {
        if (nums[curr] == 0) {
            swap(nums[curr], nums[zero]);
            ++curr;
            ++zero;
        }
        else if (nums[curr] == 1)
            ++curr;
        else if (nums[curr] == 2) {
            swap(nums[curr], nums[two]);
            --two;
        }
    }
}