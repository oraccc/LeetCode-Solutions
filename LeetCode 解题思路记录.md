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

---



### 3-无重复字符的最长字串

给定一个字符串 `s` ，请你找出其中不含有重复字符的 **最长子串** 的长度。

**思路**

使用双指针的思路，左指针和右指针之间的区域代表了没有重复字符的子串。

每次循环的时候，向右移动指针，同时记录右指针所在位置的字符出现的次数，如果发现此字符出现次数超过了1，那么移动左指针，使得左右指针之间没有重复的字符串。

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        max_length = 0
        n = len(s)
        left = right = 0
        char_count = [0]*128
        while right < n:
            char_count[ord(s[right])] += 1
            while char_count[ord(s[right])] > 1:
                char_count[ord(s[left])] -= 1
                left += 1
            max_length = max(max_length, right-left+1)
            right += 1
        return max_length
```

---



### 20-有效的括号

给定一个只包括 `'('`，`')'`，`'{'`，`'}'`，`'['`，`']'` 的字符串 `s` ，判断字符串是否有效。

有效字符串需满足：

1. 左括号必须用相同类型的右括号闭合。
2. 左括号必须以正确的顺序闭合。
3. 每个右括号都有一个对应的相同类型的左括号。

**思路**

使用栈进行解决

```python
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        for i in range(len(s)):
            if s[i] == "(" or s[i] == "{" or s[i] == "[":
                stack.append(s[i])
            elif s[i] == ")" and stack and stack[-1] == "(":
                stack.pop()
            elif s[i] == "}" and stack and stack[-1] == "{":
                stack.pop()
            elif s[i] == "]" and stack and stack[-1] == "[":
                stack.pop()
            else:
                return False
        
        return not stack
```

---



### 94-二叉树的中序遍历

给定一个二叉树的根节点 `root` ，返回 *它的 **中序** 遍历* 。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        ans = []

        def helper(root):
            if not root:
                return
            helper(root.left)
            ans.append(root.val)
            helper(root.right)
        
        helper(root)
        return ans
```

---



### 160-相交链表

给你两个单链表的头节点 `headA` 和 `headB` ，请你找出并返回两个单链表相交的起始节点。如果两个链表不存在相交节点，返回 `null` 。

图示两个链表在节点 `c1` 开始相交：

<img src="https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/14/160_statement.png" style="zoom:50%;" />

题目数据 **保证** 整个链式结构中不存在环。

**注意**，函数返回结果后，链表必须 **保持其原始结构** 。

**思路**

如果一个链表走完了，就从另一个链表的头开始走。若两个链表有相交的，则一定能相交。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        l1 = headA
        l2 = headB
        while l1 != l2:
            l1 = l1.next if l1 else headB
            l2 = l2.next if l2 else headA
        
        return l1

```

---



### 283-移动零

给定一个数组 `nums`，编写一个函数将所有 `0` 移动到数组的末尾，同时保持非零元素的相对顺序。

**请注意** ，必须在不复制数组的情况下原地对数组进行操作。

**思路**

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

