#include <iostream>
#include <vector>

using namespace std;

// Time complexity: Best O(n); Worst O(n^2);

void insertSort(vector<int> &nums, int n)
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = i; j > 0 && nums[j] < nums[j-1]; --j)
            swap(nums[j], nums[j-1]);
    }
}

int main()
{
    vector<int> vi{2, 5, 8, 6, 7, 3, 1, 0, 9, 7, 3, 4, 0, 8, 2, 1};
    insertSort(vi, vi.size());

    for (const int &i : vi) 
        cout << i << " ";
    cout << endl;
}