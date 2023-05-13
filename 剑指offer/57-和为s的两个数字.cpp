vector<int> twoSum(vector<int>& nums, int target) {
    vector<int> ans;
    int left = 0, right = nums.size()-1;
    while (left < right) {
        int sum = nums[left] + nums[right];
        if (sum == target) {
            ans.push_back(nums[left]);
            ans.push_back(nums[right]);
            return ans;
        }
        else if (sum < target) ++left;
        else if (sum > target) --right;
    }

    return ans;
}