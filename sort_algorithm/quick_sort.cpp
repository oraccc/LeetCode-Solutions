#include <iostream>
#include <vector>

using namespace std;
// Time complexity : Average O(nlogn); Worst O(n^2)
// [l, r) different from #215
void quickSort1(vector<int> &nums, int left, int right) 
{
    if (left + 1 >= right) return;
    int first = left, last = right - 1, key = nums[first];

    while (first < last)
    {
        while (first < last && nums[last] >= key) --last;
        nums[first] = nums[last];
        while (first < last && nums[first] <= key) ++first;
        nums[last] = nums[first];
    }
    nums[first] = key;
    quickSort1(nums, left, first);
    quickSort1(nums, first + 1, right);
}

// [l, r) similar to #215

int partition(vector<int> &nums, int left, int right)
{
    int i = left, j = right;
    int key = nums[i];
    while (true)
    {
        while (key <= nums[--j])
            if (j == left) break;
        while (key >= nums[++i])
            if (i == right) break;
        if (i >= j) break;
        swap(nums[i], nums[j]);
    }
    swap(nums[j], nums[left]);
    return j;
}

void quickSort2(vector<int> &nums, int left, int right)
{
    if (left + 1 >= right) return;
    int j = partition(nums, left, right);
    quickSort2(nums, left, j);
    quickSort2(nums, j + 1, right);
}

int main()
{
    vector<int> vi{2, 5, 8, 6, 7, 3, 1, 0, 9, 7, 3, 4, 0, 8, 2, 1};
    // vector<int> vi{9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    // quick_sort1(vi, 0, vi.size());
    quickSort2(vi, 0, vi.size());
    for (const int &i : vi) 
        cout << i << " ";
    cout << endl;
}