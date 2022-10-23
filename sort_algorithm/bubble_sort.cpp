#include <iostream>
#include <vector>

using namespace std;

// Time complexity: Best O(n); Worst O(n^2);

void bubbleSort(vector<int> &nums, int n)
{
    bool swapped;
    for (int i = 1; i < n; ++i)
    {
        swapped = false;
        for (int j = 1; j < n-i+1; ++j)
        {
            if (nums[j] < nums[j-1])
            {
                swap(nums[j], nums[j-1]);
                swapped = true;
            }
        }
        if (!swapped)
            break;
    }
}

int main()
{
    vector<int> vi{2, 5, 8, 6, 7, 3, 1, 0, 9, 7, 3, 4, 0, 8, 2, 1};
    bubbleSort(vi, vi.size());

    for (const int &i : vi) 
        cout << i << " ";
    cout << endl;
}