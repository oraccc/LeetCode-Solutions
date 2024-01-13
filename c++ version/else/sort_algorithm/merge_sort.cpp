#include <iostream>
#include <vector>

using namespace std;

// Time Complexity: All O(nlogn)
// [l, r) 
void mergeSort(vector<int> &nums, int l, int r, vector<int> &temp)
{
    if (l + 1 >= r) return;

    int m = l + (r - l) / 2;
    mergeSort(nums, l, m, temp);
    mergeSort(nums, m, r, temp);

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
    mergeSort(vi, 0, vi.size(), tmp);
    for (const int &i : vi) 
        cout << i << " ";
    cout << endl;
}