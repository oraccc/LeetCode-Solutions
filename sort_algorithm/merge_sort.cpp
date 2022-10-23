#include <iostream>
#include <vector>

using namespace std;

// Time Complexity: All O(nlogn)
// [l, r) 
void merge_sort(vector<int> &nums, int l, int r, vector<int> &temp)
{
    if (l + 1 >= r) return;

    int m = l + (r - l) / 2;
    merge_sort(nums, l, m, temp);
    merge_sort(nums, m, r, temp);

    int p = l, q = m, i = l;
    while (p < m || q < r)
    {
        if (q >=r || (p < m && nums[p] <= nums[q]))
            temp[i++] = nums[p++];
        else
            temp[i++] = nums[q++];
    }

    for (i = l; i < r; ++i)
        nums[i] = temp[i];
}

int main()
{
    vector<int> vi{2, 5, 8, 6, 7, 3, 1, 0, 9, 7, 3, 4, 0, 8, 2, 1};
    vector<int> tmp(vi.size());
    merge_sort(vi, 0, vi.size(), tmp);
    for (const int &i : vi) 
        cout << i << " ";
    cout << endl;
}