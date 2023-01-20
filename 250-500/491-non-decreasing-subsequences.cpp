vector<int> seq;
set<vector<int>> record;

vector<vector<int>> findSubsequences(vector<int>& nums) {
    backtracking(0, nums);
    return vector(record.begin(), record.end());
}

void backtracking(int index, vector<int> &nums) {
    if (seq.size() > 1 && record.count(seq) == 0) {
        record.insert(seq);
    }

    for (int i = index; i < nums.size(); ++i) {
        if (seq.empty() || seq.back() <= nums[i]) {
            seq.push_back(nums[i]);
            backtracking(i+1, nums);
            seq.pop_back();
        }
    }
}