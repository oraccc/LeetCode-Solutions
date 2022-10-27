//Solution: Time Out
int removeDuplicates(vector<int>& nums) {
    int duplicateNum = 0;
    for (int i = 1; i < nums.size() - duplicateNum; ) {
        if(nums[i-1] == nums[i]) {
            for (int j = i; j < nums.size() - 1 - duplicateNum; ++j)
                swap(nums[j], nums[j+1]);
            duplicateNum += 1;
        }
        else ++i;
        
    }
    return nums.size() - duplicateNum;
}

//Solution: Two Pointers left and right
int removeDuplicates(vector<int>& nums) {
    int left = 0, right = 1;
    while (right != nums.size()) {
        if (nums[left] == nums[right]) ++right;
        else {
            nums[++left] = nums[right++];
        }
    }
    return left + 1;
}