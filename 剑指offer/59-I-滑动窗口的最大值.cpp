class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        vector<int> ans;
        deque<int> dp;
        if (nums.empty()) return ans;
        for (int i = 0; i < nums.size(); ++i) {
            if (!dp.empty() && dp.front() == i - k) {
                dp.pop_front();
            }
            while (!dp.empty() && nums[dp.back()] < nums[i]) {
                dp.pop_back();
            }
            dp.push_back(i);
            if (i >= k-1) {
                ans.push_back(nums[dp.front()]);
            }
        }
        return ans;
    }
};