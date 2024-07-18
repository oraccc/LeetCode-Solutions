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



### 11-盛最多水的容器

给定一个长度为 `n` 的整数数组 `height` 。有 `n` 条垂线，第 `i` 条线的两个端点是 `(i, 0)` 和 `(i, height[i])` 。

找出其中的两条线，使得它们与 `x` 轴共同构成的容器可以容纳最多的水。

返回容器可以储存的最大水量。

**说明：**你不能倾斜容器。

**思路**

使用双指针来解决问题，一左一右分为位于容器的两端，每次移动的时候只需要保证移动左右位置中水位较低的，就有可能获得更大的容积。

> 双指针代表的是 可以作为容器边界的所有位置的范围。在一开始，双指针指向数组的左右边界，表示 数组中所有的位置都可以作为容器的边界，因为我们还没有进行过任何尝试。在这之后，我们每次将 对应的数字较小的那个指针 往 另一个指针 的方向移动一个位置，就表示我们认为 **这个指针不可能再作为容器的边界了**。

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        ans = 0
        left = 0
        right = len(height)-1
        while left < right:
            ans = max(ans, min(height[left], height[right])*(right-left))
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return ans
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



### 49-字母异位词分组

给你一个字符串数组，请你将 **字母异位词** 组合在一起。可以按任意顺序返回结果列表。

**字母异位词** 是由重新排列源单词的所有字母得到的一个新单词。

**思路1**

由于互为字母异位词的两个字符串包含的字母相同，因此两个字符串中的相同字母出现的次数一定是相同的，故可以将每个字母出现的次数使用列表表示，作为哈希表的键。

注意python不可以直接将list作为map的键，因此可以可以转换成tuple再存储。

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        ans_dict = collections.defaultdict(list)
        for s in strs:
            counts = [0]*128
            for char in s:
                counts[ord(char)] += 1
            ans_dict[tuple(counts)].append(s)
        return list(ans_dict.values())
```

**思路2**

直接将每个字符进行排序，并重组成新的字符串即可。

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        ans = {}
        for s in strs:
            sort_s = "".join(sorted(s))
            if sort_s in ans:
                ans[sort_s].append(s)
            else:
                ans[sort_s] = [s]
        return list(ans.values())
```

---



### 53-最大子数组和

给你一个整数数组 `nums` ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。**子数组**是数组中的一个连续部分。

**思路**

动态规划，dp[i]代表以nums[i]结尾的最大的连续子数组，因此若dp[i-1]时一个整数，那么加上目前的数一定是更大的连续子数组，否则就以自己重新开始。注意返回时，返回的是整个dp的最大值，因为并不能保证最大子数组一定是以最后一个元素结尾的。

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0]*n
        dp[0] = nums[0]
        for i in range(1, n):
            dp[i] = max(dp[i-1]+nums[i], nums[i])
        return max(dp)
```

---



### 56-合并区间

以数组 `intervals` 表示若干个区间的集合，其中单个区间为 `intervals[i] = [starti, endi]` 。请你合并所有重叠的区间，并返回 *一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间* 。

**思路**

首先对所有的区间进行排序，按照左边界从小到大进行排序。

接着选第一个的区间来决定左右边界，循环intervals，如果后面的区间与目前的有重叠，那么合并，否则就加入答案中。记得循环到最后还有区间没有加入答案。

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda x:x[0])
        ans = []
        left = intervals[0][0]
        right = intervals[0][1]
        for i in range(1, len(intervals)):
            curr = intervals[i]
            if curr[0] <= right:
                right = max(right, curr[1])
            else:
                ans.append([left, right])
                left = curr[0]
                right = curr[1]
        ans.append([left, right])
        return ans
```

---



### 70-爬楼梯

假设你正在爬楼梯。需要 `n` 阶你才能到达楼顶。

每次你可以爬 `1` 或 `2` 个台阶。你有多少种不同的方法可以爬到楼顶呢？

**思路**

动态规划，当爬第i阶时，有i-1和i-2两种爬法之和中爬法

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        if n <= 3:
            return n
        dp = [0]*n
        dp[0] = 1
        dp[1] = 2
        for i in range(2,n):
            dp[i] = dp[i-1]+dp[i-2]
        return dp[n-1]
```

---



### 73-矩阵置零

给定一个 `m x n` 的矩阵，如果一个元素为 **0** ，则将其所在行和列的所有元素都设为 **0** 。请使用 原地算法。

<img src="https://assets.leetcode.com/uploads/2020/08/17/mat1.jpg" style="zoom: 67%;" />

**思路**

传统的遍历思路，记录有0的行和列，并在遍历完之后依次修改

```python
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)
        m = len(matrix[0])
        zero_row = [False] * n
        zero_col = [False] * m 
        for i in range(n):
            for j in range(m):
                if matrix[i][j] == 0:
                    zero_row[i] = True
                    zero_col[j] = True
        
        for i in range(n):
            if zero_row[i]:
                for j in range(m):
                    matrix[i][j] = 0
        
        for j in range(m):
            if zero_col[j]:
                for i in range(n):
                    matrix[i][j] = 0
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



### 104-二叉树的最大深度

给定一个二叉树 `root` ，返回其最大深度。

二叉树的 **最大深度** 是指从根节点到最远叶子节点的最长路径上的节点数。

**思路**

递归计算以当前节点为根节点的树的最大深度。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        else:
            return 1+max(self.maxDepth(root.left), self.maxDepth(root.right))
```

---



### 136-只出现一次的数字

给你一个 **非空** 整数数组 `nums` ，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。

你必须设计并实现线性时间复杂度的算法来解决此问题，且该算法只使用常量额外空间。

**思路**

使用异或来使两两相同的值归零，只剩下一个单独的值

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        ans = 0
        for num in nums:
            ans = ans^num
        return ans
```

---



### 141-环形链表

给你一个链表的头节点 `head` ，判断链表中是否有环。

如果链表中有某个节点，可以通过连续跟踪 `next` 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 `pos` 来表示链表尾连接到链表中的位置（索引从 0 开始）。**注意：`pos` 不作为参数进行传递** 。仅仅是为了标识链表的实际情况。

*如果链表中存在环* ，则返回 `true` 。 否则，返回 `false` 。

**思路**

使用快慢指针，如果真的有环的话，那两者一定可以相遇

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        if not head:
            return False
        slow = fast = head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False
        
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



### 200-岛屿数量

给你一个由 `'1'`（陆地）和 `'0'`（水）组成的的二维网格，请你计算网格中岛屿的数量。

岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。

此外，你可以假设该网格的四条边均被水包围。

**思路**

一次遍历每一个点，如果发现是陆地，则岛屿数量加1，并使用dfs将相连的陆地全部找出来，并置为0，变为海洋，这样就不会重复计算陆地了

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        n = len(grid)
        m = len(grid[0])
        directions = [-1, 0, 1, 0, -1]

        def dfs_helper(i, j):
            if grid[i][j] == "0":
                return
            grid[i][j] = "0"
            for k in range(4):
                row = i + directions[k]
                col = j + directions[k+1]
                if row < n and row >= 0 and col < m and col >= 0:
                    dfs_helper(row, col)
        
        count = 0
        for i in range(n):
            for j in range(m):
                if grid[i][j] == "1":
                    count += 1
                    dfs_helper(i, j)
        return count

    
        
```

---



### 206-反转链表

给你单链表的头节点 `head` ，请你反转链表，并返回反转后的链表。

**思路1**

遍历

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        curr = head
        while curr:
            tmp = curr.next
            curr.next = prev 
            prev = curr
            curr = tmp
        return prev
```

**思路2**

递归来做，reverseList函数返回反转之后的链表头，接着将当前的head放到链表的最末尾去

返回新的链表头

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return head
        new_head = self.reverseList(head.next)
        head.next.next = head
        head.next = None
        return new_head
```

---



### 226-翻转二叉树

给你一棵二叉树的根节点 `root` ，翻转这棵二叉树，并返回其根节点。

**思路**

使用递归，每次翻转当前节点的左子树和右子树，然后交换当前节点的左右子树

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return root
        left_tree = self.invertTree(root.left)
        right_tree = self.invertTree(root.right)
        root.left = right_tree
        root.right = left_tree
        return root
```

---



### 234-回文链表

给你一个单链表的头节点 `head` ，请你判断该链表是否为回文链表。如果是，返回 `true` ；否则，返回 `false` 。

**思路**

先使用快慢指针找到链表的中心，然后反转链表的后半部分，接着一一比较前半部分链表和反转后的后半部分链表的每一个节点的值。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head):
        prev = None
        curr = head
        while curr:
            tmp = curr.next
            curr.next = prev
            prev = curr
            curr = tmp
        return prev

    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        slow = fast = head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        half = slow.next
        slow.next = None
        l1 = head
        l2 = self.reverseList(half)

        while l1 and l2:
            if l1.val == l2.val:
                l1 = l1.next
                l2 = l2.next
            else:
                return False
                
        return True
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

---



### 560-和为k的子数组

给你一个整数数组 `nums` 和一个整数 `k` ，请你统计并返回 *该数组中和为 `k` 的子数组的个数* 。

子数组是数组中元素的连续非空序列。

**思路**

使用前缀和可以快速计算区间的和，当计算当前的前缀和pre_sum时，可以检查pre_sum-k这个前缀和是否存在，有几个（因为从左往右遍历，因此此时pre_sum-k如果有，那便一定已经计算过了），计入总数中。使用dict来存储这个次数。这里为了方便使用collection.defaultdict来创建默认是0的dict。

```python
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        pre_sum = 0
        count = 0
        prefix_dict = collections.defaultdict(int)
        prefix_dict[0] = 1
        for num in nums:
            pre_sum += num
            count += prefix_dict[pre_sum-k]
            prefix_dict[pre_sum] += 1
        return count
```

---



### 994-腐烂的橘子

在给定的 `m x n` 网格 `grid` 中，每个单元格可以有以下三个值之一：

- 值 `0` 代表空单元格；
- 值 `1` 代表新鲜橘子；
- 值 `2` 代表腐烂的橘子。

每分钟，腐烂的橘子 **周围 4 个方向上相邻** 的新鲜橘子都会腐烂。

返回 *直到单元格中没有新鲜橘子为止所必须经过的最小分钟数。如果不可能，返回 `-1`* 。

<img src="https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2019/02/16/oranges.png" style="zoom:50%;" />

**思路**

因为求的是最短路径类似的问题，因此想到用BFS。注意可能有多个腐败源，因此先全部扫描一遍，计算目前新鲜的橘子的数目。若在BFS结束之后，还没有归零，说明没有办法全部覆盖到，因此返回-1。

queue存储的是一个tuple，代表坐标位置。注意为了防止重复计算，因此一旦遍历到一个新鲜的橘子，就要将其标记为腐烂。

```python
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        count = 0
        directions = [-1, 0, 1, 0, -1]
        queue = []
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    count += 1
                elif grid[i][j] == 2:
                    queue.append((i, j))
        ans = 0
        if count == 0:
            return ans
        while queue:
            ans += 1
            for _ in range(len(queue)):
                i, j = queue[0]
                queue.pop(0)
                for k in range(4):
                    row = i + directions[k]
                    col = j + directions[k+1]
                    if row < m and row >= 0 and col < n and col >=0 and grid[row][col]==1:
                        count -= 1
                        grid[row][col] = 2
                        queue.append((row, col))
        if count != 0:
            return -1
        else:
            return ans-1

```

---
