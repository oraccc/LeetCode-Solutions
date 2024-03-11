from typing import List

def quick_sort(nums: List[int], left: int, right: int) -> List[int]:
    if left >= right: return nums
    pivot = nums[left]
    low = left
    high = right
    while left < right:
        while left < right and nums[right] >= pivot:
            right -= 1
        nums[left] = nums[right]
        while left < right and nums[left] <= pivot:
            left += 1
        nums[right] = nums[left]

    nums[right] = pivot
    quick_sort(nums, low, left-1)
    quick_sort(nums, left+1, high)
    return nums

def quick_sort_reverse(nums: List[int], left: int, right: int) -> List[int]:
    if left >= right: return nums
    pivot = nums[left]
    low = left
    high = right
    while left < right:
        while left < right and nums[right] <= pivot:
            right -= 1
        nums[left] = nums[right]
        while left < right and nums[left] >= pivot:
            left += 1
        nums[right] = nums[left]

    nums[right] = pivot
    quick_sort_reverse(nums, low, left-1)
    quick_sort_reverse(nums, left+1, high)
    return nums


if __name__ == "__main__":
    test = [2, 4, 7, 1, 3, 3, 9, 0]
    print(quick_sort(test, 0, len(test)-1))
    print(quick_sort_reverse(test, 0, len(test)-1))