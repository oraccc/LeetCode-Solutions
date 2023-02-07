int totalFruit(vector<int>& fruits) {
    unordered_map<int, int> pick;
    int left = 0, right = 0, count = 0;

    while (right < fruits.size()) {
        ++pick[fruits[right]];
        while (pick.size() > 2) {
            --pick[fruits[left]];
            if (pick[fruits[left]] == 0) {
                pick.erase(fruits[left]);
            }
            ++left;
        }
        count = max(count, right-left+1);
        ++right;
    }

    return count;
}