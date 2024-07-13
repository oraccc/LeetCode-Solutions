# LeetCode 解题思路记录



### 1-两数之和

给定一个整数数组 `nums` 和一个整数目标值 `target`，请你在该数组中找出 **和为目标值** *`target`* 的那 **两个** 整数，并返回它们的数组下标。

你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。

你可以按任意顺序返回答案。

**思路**

设置一个字典保存整数到idx的一个映射，循环这个数组，如果这个可以找到**加上这个数字即可满足target的那个值**，那就可以直接返回对应的索引，否则就在字典中加上当前这个映射。这样如果有满足的，就一定可以找到便利到它之前的那个数值所在的位置。

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        num_to_idx_map = {}
        ans = []
        for i in range(len(nums)):
            if target-nums[i] in num_to_idx_map:
                ans = [i, num_to_idx_map[target-nums[i]]]
                return ans
            else:
                num_to_idx_map[nums[i]] = i
        
        return ans
```



### 283-移动零

给定一个数组 `nums`，编写一个函数将所有 `0` 移动到数组的末尾，同时保持非零元素的相对顺序。

**请注意** ，必须在不复制数组的情况下原地对数组进行操作。

**思路**：

使用双指针的思路，左指针指向当前**已经处理好的序列的尾部**，右指针指向待处理序列的头部。右指针不断向右移动，每次右指针指向非零数，则将左右指针对应的数交换，同时左指针右移。左指针左边均为非零数。



```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        left = right = 0
        while right < n:
            if nums[right] != 0:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
            right += 1
```

