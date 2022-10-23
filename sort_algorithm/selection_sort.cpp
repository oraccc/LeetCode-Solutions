#include <iostream>
#include <vector>

using namespace std;

// Time complexity: All O(n^2);

void selectionSort(vector<int> &nums, int n)
{
    int min;
    for (int i = 0; i < n - 1; ++i)
    {
        min = i;
        for (int j = i + 1; j < n; ++j)
        {
            if (nums[j] < nums[min])
            {
                min = j;
            }
        }
        swap(nums[min], nums[i]);
    }
}

int main()
{
    vector<int> vi{2, 5, 8, 6, 7, 3, 1, 0, 9, 7, 3, 4, 0, 8, 2, 1};
    selectionSort(vi, vi.size());

    for (const int &i : vi) 
        cout << i << " ";
    cout << endl;
}