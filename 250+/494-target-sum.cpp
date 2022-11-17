int count = 0;
int findTargetSumWays(vector<int>& nums, int target) {
    int currSum = 0;
    dfs(0, currSum, nums, target);
    return count;
}

void dfs(int index, int currSum, vector<int> &nums, int target) {
    if (index == nums.size()) {
        if (currSum == target)
            ++count;
        return;
    }
    dfs(index+1, currSum + nums[index], nums, target);
    dfs(index+1, currSum -= nums[index], nums, target);
}

//better time complexity
int n;
map<pair<int,int>,int> cache;

int dfs(vector<int>& nums, int target, int currSum, int i)
{
    auto find = cache.find(make_pair(i, currSum));
    
    if (find != cache.end())
        return find->second;
    
    if (i == nums.size() && currSum == target)
        return 1;

    if (i >= nums.size())
        return 0;

    int count = dfs(nums, target, currSum + nums[i], i + 1) + dfs(nums, target, currSum - nums[i], i + 1);
    cache.insert(make_pair(make_pair(i, currSum), count));
    return count;
}

int findTargetSumWays(vector<int>& nums, int target) 
{
    return dfs(nums, target, 0, 0);
}