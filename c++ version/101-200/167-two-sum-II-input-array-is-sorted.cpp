vector<int> twoSum(vector<int>& numbers, int target) {
    int left = 0, right = numbers.size() - 1;
    while (left != right) {
        if (target == numbers[left] + numbers[right]) break;
        if (target < numbers[left] + numbers[right]) --right;
        else ++left;
    }

    return vector<int>{left + 1, right + 1};
}