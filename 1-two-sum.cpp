//Use hash table: value -> index

vector<int> twoSum(vector<int>& nums, int target) {
    unordered_map<int, int> mp;
    int n = nums.size();
    vector<int> answer;
    for (int i = 0; i < n; ++i){
        int remain = target - nums[i];
        auto pos = mp.find(remain);
        if (pos != mp.end()){
            answer = {i, mp[remain]};
            return answer;
        }
        else{
            mp[nums[i]] = i;
        }
    }

    return answer;
}