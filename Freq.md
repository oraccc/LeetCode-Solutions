# LeetCode 高频题



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



### 215-数组中的第k个最大的元素

给定整数数组 `nums` 和整数 `k`，请返回数组中第 `k` 个最大的元素。

请注意，你需要找的是数组排序后的第 `k` 个最大的元素，而不是第 `k` 个不同的元素。

你必须设计并实现时间复杂度为 `O(n)` 的算法解决此问题。

 **思路**

第k个最大的元素其实就是k个最大元素中最小的那个，那么可以用最小堆去完成这个工作，最小堆的栈顶永远维持着最小的元素，只需要维持一个k个大小的最小堆即可。

python中的最小堆语法：

`heapq.heapify(heap)`：初始化一个最小堆；

`heapq.heappush(heap, item)`：将元素加入最小堆中；

`heapq.heappop(heap)`：取出堆顶的元素；

如果想要实现最大堆，只需要每次放入堆的元素取反即可。

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        heap = nums[:k]
        heapq.heapify(heap)
        for i in range(k, len(nums)):
            heapq.heappush(heap, nums[i])
            heapq.heappop(heap)
        return heap[0]
```

---



### 15-三数之和

给你一个整数数组 `nums` ，判断是否存在三元组 `[nums[i], nums[j], nums[k]]` 满足 `i != j`、`i != k` 且 `j != k` ，同时还满足 `nums[i] + nums[j] + nums[k] == 0` 。请你返回所有和为 `0` 且不重复的三元组。

**注意：**答案中不可以包含重复的三元组。

**思路**

注意因为本题要求不能有重复的答案，因此需要额外的去重工作。首先固定一个值，然后使用双指针左右查找即可。对于固定的值，首先判断是不是之前已经被选过了，如果已经被选过了，那么就不应该在被选，直接跳过。同理，如果双指针的值已经找到答案了，那么就应该同时移动左右两个指针（只移动移动的话，可能不会是答案的），一样的，这样的指针移动时也应该去重。

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        ans = []
        nums.sort()
        for i in range(n):
            if nums[i] > 0:
                return ans
            if i >= 1 and nums[i] == nums[i-1]:
                continue
            target = -nums[i]
            left = i+1
            right = n-1
            while left < right:
                if nums[left] + nums[right] == target:
                    ans.append([nums[i], nums[left], nums[right]])
                    while left < right and nums[left] == nums[left+1]:
                        left += 1
                    while left < right and nums[right] == nums[right-1]:
                        right -= 1
                    left += 1
                    right -= 1
                elif nums[left] + nums[right] < target:
                    left += 1
                else:
                    right -= 1
        return ans
```

---



### 146-LRU缓存

请你设计并实现一个满足 LRU (最近最少使用) 缓存约束的数据结构。

实现 `LRUCache` 类：

- `LRUCache(int capacity)` 以 **正整数** 作为容量 `capacity` 初始化 LRU 缓存
- `int get(int key)` 如果关键字 `key` 存在于缓存中，则返回关键字的值，否则返回 `-1` 。
- `void put(int key, int value)` 如果关键字 `key` 已经存在，则变更其数据值 `value` ；如果不存在，则向缓存中插入该组 `key-value` 。如果插入操作导致关键字数量超过 `capacity` ，则应该 **逐出** 最久未使用的关键字。

函数 `get` 和 `put` 必须以 `O(1)` 的平均时间复杂度运行。

**思路**

LRU 缓存机制可以通过**哈希表**辅以**双向链表**实现，我们用一个哈希表和一个双向链表维护所有在缓存中的键值对。双向链表按照被使用的顺序存储了这些键值对，靠近头部的键值对是最近使用的，而靠近尾部的键值对是最久未使用的。

哈希表即为普通的哈希映射（HashMap），通过缓存数据的键映射到其在双向链表中的位置。

这样以来，我们首先使用哈希表进行定位，找出缓存项在双向链表中的位置，随后将其移动到双向链表的头部，即可在 O(1) 的时间内完成 get 或者 put 操作。具体的方法如下：

对于 get 操作，首先判断 key 是否存在：

* 如果 key 不存在，则返回 −1；

* 如果 key 存在，则 key 对应的节点是最近被使用的节点。通过哈希表定位到该节点在双向链表中的位置，并将其移动到双向链表的头部，最后返回该节点的值。

对于 put 操作，首先判断 key 是否存在：

* 如果 key 不存在，使用 key 和 value 创建一个新的节点，在双向链表的头部添加该节点，并将 key 和该节点添加进哈希表中。然后判断双向链表的节点数是否超出容量，如果超出容量，则删除双向链表的尾部节点，并删除哈希表中对应的项；

* 如果 key 存在，则与 get 操作类似，先通过哈希表定位，再将对应的节点的值更新为 value，并将该节点移到双向链表的头部。

```python
class ListNode:
    def __init__(self, key=None, val=None):
        self.key = key
        self.val = val
        self.next = None
        self.prev = None


class LRUCache:
    def __init__(self, capacity: int):
        self.head = ListNode()
        self.tail = ListNode()
        self.capacity = capacity
        self.dict = {}

        self.head.next = self.tail
        self.tail.prev = self.head


    def move_node_to_head(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = None
        node.next = None

        self.head.next.prev = node
        node.next = self.head.next
        self.head.next = node
        node.prev = self.head


    def get(self, key: int) -> int:
        if key not in self.dict.keys():
            return -1
        else:
            node = self.dict[key]
            self.move_node_to_head(node)
            return node.val

    def put(self, key: int, value: int) -> None:
        if key not in self.dict.keys():
            if len(self.dict) == self.capacity:
                self.dict.pop(self.tail.prev.key)
                self.tail.prev = self.tail.prev.prev
                self.tail.prev.next = self.tail

            new_node = ListNode(key, value)
            self.dict[key] = new_node
            self.head.next.prev = new_node
            new_node.next = self.head.next
            self.head.next = new_node
            new_node.prev = self.head
        else:
            self.dict[key].val = value
            node = self.dict[key]
            self.move_node_to_head(node)



# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
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



### Extra-快速排序

```python
def quick_sort(nums, left, right):
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



### 46-全排列

给定一个不含重复数字的数组 `nums` ，返回其 *所有可能的全排列* 。你可以 **按任意顺序** 返回答案。

**思路**

因为需要遍历全可能的结果，因此需要回溯。回溯结束的判断条件是目前的数组长度已经到达n了。同时需要设置一个数组visited，来记录有没有访问过该位置的元素。回溯的时候，若需要加入这个位置的元素，那么就将此位置为True，退出时，pop此元素，并且恢复此位为False。

注意python列表的性质，添加结果的时候需要使用切片操作进行复制。

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        visited = [False]*n
        pre = []
        ans = []
        def backtracking():
            if len(pre) == n:
                ans.append(pre[:])
                return
            for i in range(n):
                if not visited[i]:
                    visited[i] = True
                    pre.append(nums[i])
                    backtracking()
                    pre.pop()
                    visited[i] = False
        
        backtracking()
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



### 102-二叉树的层序遍历

给你二叉树的根节点 `root` ，返回其节点值的 **层序遍历** 。 （即逐层地，从左到右访问所有节点）。

**思路**

层序遍历使用队列即可解决

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        ans = []
        queue = []
        if not root:
            return ans
        queue.append(root)
        while queue:
            n = len(queue)
            tmp = []
            for _ in range(n):
                node = queue.pop(0)
                tmp.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            ans.append(tmp)

        return ans
```

---



### 4-寻找两个正序数组的中位数

给定两个大小分别为 `m` 和 `n` 的正序（从小到大）数组 `nums1` 和 `nums2`。请你找出并返回这两个正序数组的 **中位数** 。

算法的时间复杂度应该为 `O(log (m+n))` 。

**思路**

我们把数组 A 和数组 B 分别在 i 和 j 进行切割。

将 i 的左边和 j 的左边组合成「左半部分」，将 i 的右边和 j 的右边组合成「右半部分」。

为了保证 max ( A [ i - 1 ] , B [ j - 1 ]）） <= min ( A [ i ] , B [ j ]），因为 A 数组和 B 数组是有序的，所以 A [ i - 1 ] <= A [ i ]，B [ i - 1 ] <= B [ i ] 这是天然的，所以我们只需要保证 B [ j - 1 ] < = A [ i ] 和 A [ i - 1 ] <= B [ j ] 所以我们分两种情况讨论：

* B [ j - 1 ] > A [ i ]，并且为了不越界，要保证 j != 0，i != m
  * 此时很明显，我们需要增加 i 

- A [ i - 1 ] > B [ j ] ，并且为了不越界，要保证 i != 0，j != n
  - 此时和上边的情况相反，我们要减少 i ，增大 j 。

- 当 i = 0, 或者 j = 0，也就是切在了最前边。
  - 此时左半部分当 j = 0 时，最大的值就是 A [ i - 1 ] ；当 i = 0 时 最大的值就是 B [ j - 1] 。右半部分最小值和之前一样。
- 当 i = m 或者 j = n，也就是切在了最后边。
  - 此时左半部分最大值和之前一样。右半部分当 j = n 时，最小值就是 A [ i ] ；当 i = m 时，最小值就是B [ j ] 。

所有的思路都理清了，最后一个问题，增加 i 的方式。当然用二分了。初始化 i 为中间的值，然后减半找中间的，减半找中间的，减半找中间的直到答案。

```python
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        n = len(nums1)
        m = len(nums2)
        if n > m:
            nums1, nums2 = nums2, nums1
            n, m = m, n
        
        i_min = 0
        i_max = n
        while i_min <= i_max:
            i = (i_min + i_max)//2
            j = (m+n+1)//2 - i
            if i != 0 and j != m and nums1[i-1] > nums2[j]:
                i_max = i
            elif j != 0 and i != n and nums2[j-1] > nums1[i]:
                i_min = i+1
            else:
                if i == 0:
                    left_max = nums2[j-1]
                elif j == 0:
                    left_max = nums1[i-1]
                else:
                    left_max = max(nums1[i-1], nums2[j-1])

                if (m+n) % 2 == 1:
                    return left_max

                if i == n:
                    right_min = nums2[j]
                elif j == m:
                    right_min = nums1[i]
                else:
                    right_min = min(nums1[i], nums2[j])
                
                return (left_max + right_min)/2
        return 0
```

---



### 21-合并两个有序链表

将两个升序链表合并为一个新的 **升序** 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 

 **思路**

注意循环条件是l1 and l2

由于题目要求是拼接原来的链表，因此最好不要创建新的节点，直接对原来的链表做操作即可

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        l1 = list1
        l2 = list2
        dummy_head = ListNode()
        curr = dummy_head
        while l1 and l2:
            if l1.val <= l2.val:
                curr.next = l1
                l1 = l1.next
            else:
                curr.next = l2
                l2 = l2.next
            curr = curr.next
            
        if l1:
            curr.next = l1
        if l2:
            curr.next = l2
        return dummy_head.next

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





### 25-k个一组翻转链表

给你链表的头节点 `head` ，每 `k` 个节点一组进行翻转，请你返回修改后的链表。

`k` 是一个正整数，它的值小于或等于链表的长度。如果节点总数不是 `k` 的整数倍，那么请将最后剩余的节点保持原有顺序。

你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。

<img src="https://assets.leetcode.com/uploads/2020/10/03/reverse_ex2.jpg" style="zoom: 80%;" />

**思路**

结合逆转链表，依次取出k个链表来，逆转，注意需要保存在逆转开始前的那个prev节点，可以用画图的方式更好地辅助理解。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverse_link(self, head):
        prev = None
        curr = head
        while curr:
            tmp = curr.next
            curr.next = prev
            prev = curr
            curr = tmp
        return prev


    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        dummy_head = ListNode(-1)
        dummy_head.next = head
        prev = dummy_head
        curr = head
        count = 0
        while curr:
            count += 1
            if count == k:
                count = 0
                tmp = curr.next
                curr.next = None
                prev.next = self.reverse_link(prev.next)
                while prev.next:
                    prev = prev.next
                prev.next = tmp
                curr = tmp
            else:
                curr = curr.next
        return dummy_head.next
```

---



### 1143-最长公共子序列

给定两个字符串 `text1` 和 `text2`，返回这两个字符串的最长 **公共子序列** 的长度。如果不存在 **公共子序列** ，返回 `0` 。

一个字符串的 **子序列** 是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。

- 例如，`"ace"` 是 `"abcde"` 的子序列，但 `"aec"` 不是 `"abcde"` 的子序列。

两个字符串的 **公共子序列** 是这两个字符串所共同拥有的子序列。

**思路**

`dp[i][j] `表示 `text_1 [0:i]` 和 `text_2 [0:j]` 的最长公共子序列的长度。

若该位置两者字符相同，则可以在原先的基础上+1，否则就是取两者各退一位中最大的那个。

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        n = len(text1)
        m = len(text2)

        dp = [[0]*(m+1) for _ in range(n+1)]
        
        for i in range(1, n+1):
            for j in range(1, m+1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1]+1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[n][m]
```

---



### 72-编辑距离

给你两个单词 `word1` 和 `word2`， *请返回将 `word1` 转换成 `word2` 所使用的最少操作数* 。

你可以对一个单词进行如下三种操作：

- 插入一个字符
- 删除一个字符
- 替换一个字符

**思路**

`dp[i][j] `代表 word1 到 i 位置转换成 word2 到 j 位置需要最少步数

所以，

当 `word1[i] == word2[j]`，`dp[i][j] = dp[i-1][j-1]`；

当 `word1[i] != word2[j]`，`dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) + 1`

其中，`dp[i-1][j-1] `表示替换操作，`dp[i-1][j] `表示删除操作，`dp[i][j-1] `表示插入操作。

注意还需要考虑空字符的情况，因此会需要对第0行和第0列进行单独的处理。

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        n = len(word1)
        m = len(word2)
        dp = [[0]*(m+1) for _ in range(n+1)]

        for i in range(1, n+1):
            dp[i][0] = dp[i-1][0] + 1
        
        for j in range(1, m+1):
            dp[0][j] = dp[0][j-1] + 1
        
        for i in range(1, n+1):
            for j in range(1, m+1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i-1][j-1], dp[i][j-1])+1
        
        return dp[n][m]
```

---



### 54-螺旋矩阵

给你一个 `m` 行 `n` 列的矩阵 `matrix` ，请按照 **顺时针螺旋顺序** ，返回矩阵中的所有元素。

<img src="https://assets.leetcode.com/uploads/2020/11/13/spiral.jpg" style="zoom:67%;" />

**思路**

设置四个循环以及四个变量分别代表边界

注意在逆向循环的时候，不要重复读取数字，可以考虑特殊情况，即只有一列或者只有一行的情况，就可以想出逆向情况下`if left < right and top < bottom`这个边界情况。

```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        top = 0
        bottom = len(matrix)-1
        left = 0
        right = len(matrix[0])-1
        ans = []
        while left <= right and top <= bottom:
            for i in range(left, right+1):
                ans.append(matrix[top][i])
            for i in range(top+1, bottom+1):
                ans.append(matrix[i][right])
            if left < right and top < bottom:
                for i in range(right-1, left-1, -1):
                    ans.append(matrix[bottom][i])
                for i in range(bottom-1, top, -1):
                    ans.append(matrix[i][left])
            
            left, right, top, bottom = left+1, right-1, top+1, bottom-1
        return ans


```

---



### 56-合并区间

以数组 `intervals` 表示若干个区间的集合，其中单个区间为 `intervals[i] = [starti, endi]` 。请你合并所有重叠的区间，并返回 *一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间* 。

**思路**

首先对所有的区间进行排序，按照左边界从小到大进行排序。为什么是左边界，因为我们需要左边界来确定合并的范围

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



### 121-买卖股票的最佳时机

给定一个数组 `prices` ，它的第 `i` 个元素 `prices[i]` 表示一支给定股票第 `i` 天的价格。

你只能选择 **某一天** 买入这只股票，并选择在 **未来的某一个不同的日子** 卖出该股票。设计一个算法来计算你所能获取的最大利润。

返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 `0` 。

**思路**

因为只能买卖一次，因此这是一个贪心算法的问题。只需要在最低的时候买进，在之后某一天最高的之后卖出即可。遍历整个数组，如果当天价格是比目前持有的低，那就持有当天的，反之则看如果卖出可以卖多少。最后取最大值即可。

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        hold = prices[0]
        max_profit = 0
        for i in range(1, len(prices)):
            if prices[i] < hold:
                hold = prices[i]
            else:
                max_profit = max(max_profit, prices[i]-hold)
        return max_profit
```

---



### 43-字符串相乘

给定两个以字符串形式表示的非负整数 `num1` 和 `num2`，返回 `num1` 和 `num2` 的乘积，它们的乘积也表示为字符串形式。

**注意：**不能使用任何内置的 BigInteger 库或直接将输入转换为整数。

**思路**

我们可以从列竖式的角度取解决这个问题，首先需要一个辅助函数去计算单个数字与多位数的乘法结果，需要使用进位carry记住当前结果应该进的位数。接着将其转换成依次从左至右计算每个单位数和多位数的结果，注意每次结果要乘以10。

```python
class Solution:
    def multiply(self, num1: str, num2: str) -> str:

        def cal_single(nums, s):
            result = 0
            carry = 0
            times = 0
            s = int(s)
            if s == 0:
                return 0
            for i in range(len(nums)-1, -1, -1):
                digit = int(nums[i])
                mul = (digit*s+carry) % 10
                carry = (digit*s+carry) // 10
                result += (10**times) * mul
                times += 1
            if carry:
                result += (10**times) * carry

            return result

        ans = 0
        for i in range(len(num2)):
            ans = ans*10+cal_single(num1, num2[i])

        return str(ans)
```

---



### 32-最长有效括号

给你一个只包含 `'('` 和 `')'` 的字符串，找出最长有效（格式正确且连续）括号子串的长度。

**思路**

我们始终保持栈底元素为当前已经遍历过的元素中「最后一个没有被匹配的右括号的下标」，这样的做法主要是考虑了边界条件的处理，栈里其他元素维护左括号的下标：

* 对于遇到的每个 ‘(’ ，我们将它的下标放入栈中
* 对于遇到的每个 ‘)’ ，我们先弹出栈顶元素表示匹配了当前右括号：
  * 如果栈为空，说明当前的右括号为没有被匹配的右括号，我们将其下标放入栈中来更新我们之前提到的「最后一个没有被匹配的右括号的下标」
  * 如果栈不为空，当前右括号的下标减去栈顶元素即为「以该右括号为结尾的最长有效括号的长度」
    我们从前往后遍历字符串并更新答案即可。

需要注意的是，如果一开始栈为空，第一个字符为左括号的时候我们会将其放入栈中，这样就不满足提及的「最后一个没有被匹配的右括号的下标」，为了保持统一，我们在一开始的时候往栈中放入一个值为 −1 的元素。



```python
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        stack = [-1]
        max_len = 0
        for i in range(len(s)):
            if s[i] == "(":
                stack.append(i)
            else:
                stack.pop()
                if not stack:
                    stack.append(i)
                else:
                    max_len = max(max_len, i-stack[-1])
        return max_len
        
```

---



### 93-复原IP地址

**有效 IP 地址** 正好由四个整数（每个整数位于 `0` 到 `255` 之间组成，且不能含有前导 `0`），整数之间用 `'.'` 分隔。

- 例如：`"0.1.2.201"` 和` "192.168.1.1"` 是 **有效** IP 地址，但是 `"0.011.255.245"`、`"192.168.1.312"` 和 `"192.168@1.1"` 是 **无效** IP 地址。

给定一个只包含数字的字符串 `s` ，用以表示一个 IP 地址，返回所有可能的**有效 IP 地址**，这些地址可以通过在 `s` 中插入 `'.'` 来形成。你 **不能** 重新排序或删除 `s` 中的任何数字。你可以按 **任何** 顺序返回答案。

 **思路**

使用回溯法，每次从start位置开始选择1-3个长度的字符，如果这个长度是合法的，那么就更新这个start位置。结束的条件便是现在已经有4个ip地址并且开始的start是字符串结尾，说明找到了一个满足条件的地址。

```python
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:

        curr = []
        ans = []
        n = len(s)

        def is_valid(start, end):
            if end > n:
                return False
            if end-start >= 2 and s[start] == "0":
                return False
            if int(s[start:end]) > 255: 
                return False
            return True
            

        def backtracking(start):
            if len(curr) == 4:
                if start == n:
                    ans.append(".".join(curr))
                    return
                else:
                    return
            
            for i in range(1, 4):
                if is_valid(start, start+i):
                    curr.append(s[start:start+i])
                    backtracking(start+i)
                    curr.pop()

        backtracking(0)

        return ans
        
```

---



### 22-括号生成

数字 `n` 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 **有效的** 括号组合。

**思路**

每一步只有两个选择，要么添加左括号，要么添加右括号，每次进入回溯和结束回溯的状态要一致；由于要生成的是有效的，也就是右括号不能比左括号多，要额外多一个条件。

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        curr = []
        ans = []
        left = 0
        right = 0

        def backtracking():
            nonlocal left
            nonlocal right
            if right > left:
                return
            if len(curr) == 2*n:
                if left == right:
                    ans.append("".join(curr))
                return
            curr.append("(")
            left += 1
            backtracking()
            left -= 1
            curr.pop()

            curr.append(")")
            right += 1
            backtracking()
            right -= 1
            curr.pop()

        backtracking()
        return ans
```

---



### 88-合并两个有序数组

给你两个按 **非递减顺序** 排列的整数数组 `nums1` 和 `nums2`，另有两个整数 `m` 和 `n` ，分别表示 `nums1` 和 `nums2` 中的元素数目。

请你 **合并** `nums2` 到 `nums1` 中，使合并后的数组同样按 **非递减顺序** 排列。

**注意：**最终，合并后数组不应由函数返回，而是存储在数组 `nums1` 中。为了应对这种情况，`nums1` 的初始长度为 `m + n`，其中前 `m` 个元素表示应合并的元素，后 `n` 个元素为 `0` ，应忽略。`nums2` 的长度为 `n` 。

**思路**

倒着顺序进行遍历，将大的值放在nums1的末尾。当遍历结束时，观察nums2是否还没有遍历完，如果还没有遍历完，那么就说明这些数字全都比nums1里面的小，因此可以直接全部放在前面。

```python
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        i = m-1
        j = n-1
        k = m+n-1
        while i >= 0 and j >= 0:
            if nums2[j] > nums1[i]:
                nums1[k] = nums2[j]
                j -= 1
            else:
                nums1[k] = nums1[i]
                i -= 1
            k -= 1
        if j >= 0:
            nums1[:j+1] = nums2[:j+1]
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



### 142-环形链表-II

给定一个链表的头节点  `head` ，返回链表开始入环的第一个节点。 *如果链表无环，则返回 `null`。*

如果链表中有某个节点，可以通过连续跟踪 `next` 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 `pos` 来表示链表尾连接到链表中的位置（**索引从 0 开始**）。如果 `pos` 是 `-1`，则在该链表中没有环。**注意：`pos` 不作为参数进行传递**，仅仅是为了标识链表的实际情况。

**不允许修改** 链表。

**思路**

我们使用两个指针，fast 与 slow。它们起始都位于链表的头部。随后，slow 指针每次向后移动一个位置，而 fast 指针向后移动两个位置。如果链表中存在环，则 fast 指针最终将再次与 slow 指针在环中相遇。

如下图所示，设链表中环外部分的长度为 a。slow 指针进入环后，又走了 b 的距离与 fast 相遇。此时，fast 指针已经走完了环的 n 圈，因此它走过的总距离为 a+n(b+c)+b=a+(n+1)b+nc。

<img src="https://assets.leetcode-cn.com/solution-static/142/142_fig1.png" style="zoom: 25%;" />

根据题意，任意时刻，fast 指针走过的距离都为 slow 指针的 2 倍。因此，我们有

$$
a+(n+1)b+nc=2(a+b)⟹a=c+(n−1)(b+c)
$$
有了 
$$
a=c+(n−1)(b+c)
$$
 的等量关系，我们会发现：从相遇点到入环点的距离加上 n−1 圈的环长，恰好等于从链表头部到入环点的距离。

因此，当发现 slow 与 fast 相遇时，我们再额外使用一个指针 ptr。起始，它指向链表头部；随后，它和 slow 每次向后移动一个位置。最终，它们会在入环点相遇。



```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow = fast = head
        while True:
            if not fast or not fast.next:
                return None
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                break

        fast = head
        while fast != slow:
            fast = fast.next
            slow = slow.next
        return fast
        
```

---



### 148-排序链表

给你链表的头结点 `head` ，请将其按 **升序** 排列并返回 **排序后的链表** 。

**思路**

使用归并排序的思想。如果当前没有节点或者只有一个节点，直接返回即可。

否则使用快慢指针将链表分成两个部分，然后分别对前半部分后面部分进行排序，对与排序好的结果，设置一个新的头部，然后就是合并两个链表的操作了。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return head
        slow = fast = head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        half = slow.next
        slow.next = None 
        l = self.sortList(head)
        r = self.sortList(half)

        dummy_head = ListNode(-1)
        curr = dummy_head
        while l and r:
            if l.val < r.val:
                curr.next = l 
                curr = curr.next
                l = l.next
            else:
                curr.next = r
                curr = curr.next
                r = r.next
        if l:
            curr.next = l 
        if r:
            curr.next = r 
        return dummy_head.next

```

---



### 300-最长递增子序列

给你一个整数数组 `nums` ，找到其中最长严格递增子序列的长度。

**子序列** 是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，`[3,6,2,7]` 是数组 `[0,3,1,6,2,2,7]` 的子序列。

**思路**

dp[i]代表以第i个元素结尾的最长递增子序列的长度。最后返回的是整个dp数组的最大值，因为LIS很可能不是以最后一个结尾的。

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [1]*n 
        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j]+1)
        
        return max(dp)
```

---



### 19-删除链表的倒数第N个结点

给你一个链表，删除链表的倒数第 `n` 个结点，并且返回链表的头结点。

**思路**

设置快慢指针，让快慢指针实现差距n个位置。接着同时推进快慢指针，当快指针到链表结尾的时候，慢指针的下一个便是要删除的节点。

注意可能要删除的节点便是头节点，因此需要设置一个dummy_head，快慢指针一开始都指向这个头节点前一个的指针。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dummy_head = ListNode()
        dummy_head.next = head
        slow = fast = dummy_head
        
        for i in range(n):
            fast = fast.next
        while fast.next:
            slow = slow.next
            fast = fast.next
        slow.next = slow.next.next
        return dummy_head.next
```

---



### 33-搜索旋转排序数组

整数数组 `nums` 按升序排列，数组中的值 **互不相同** 。

在传递给函数之前，`nums` 在预先未知的某个下标 `k`（`0 <= k < nums.length`）上进行了 **旋转**，使数组变为 `[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]`（下标 **从 0 开始** 计数）。例如， `[0,1,2,4,5,6,7]` 在下标 `3` 处经旋转后可能变为 `[4,5,6,7,0,1,2]` 。

给你 **旋转后** 的数组 `nums` 和一个整数 `target` ，如果 `nums` 中存在这个目标值 `target` ，则返回它的下标，否则返回 `-1` 。

你必须设计一个时间复杂度为 `O(log n)` 的算法解决此问题。

**思路**

本体采用二分搜索法可以解决。注意我们找到中间的位置mid之后，应该进行分类讨论，找到严格递增的是左边还是右边。因为二分搜索只有在递增的序列中生效。判断的条件也很简单，`nums[mid] < nums[right]`。如果target在严格递增的范围内（注意不仅要和mid比，也需要和left或者right比），那么可以照常移动指针，反之反方向移动就可以了。

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left = 0
        right = len(nums)-1
        while left < right:
            mid = (left+right)//2
            if nums[mid] == target:
                return mid
            if nums[mid] < nums[right]:
                if nums[mid] < target and nums[right] >= target:
                    left = mid+1
                else:
                    right = mid
            else:
                if nums[mid] > target and nums[left] <= target:
                    right = mid
                else:
                    left = mid+1
        
        if nums[left] == target:
            return left
        else:
            return -1

```

---



### 69 x的平方根

给你一个非负整数 `x` ，计算并返回 `x` 的 **算术平方根** 。

由于返回类型是整数，结果只保留 **整数部分** ，小数部分将被 **舍去 。**

**注意：**不允许使用任何内置指数函数和算符，例如 `pow(x, 0.5)` 或者 `x ** 0.5` 。

**思路**

这题的思路比较直观，但是会有一个边界条件的判断比较难处理。在这里我们考虑二分的位置要取在靠左的位置，并且为了防止无限循环，我们只需要动left的指针就可以了，但是这样计算的结果会比正确结果大1，因此需要返回l-1。

```python
class Solution:
    def mySqrt(self, x: int) -> int:
        if x <= 1:
            return x
        l = 1
        r = x

        while l < r:
            mid = (l+r) // 2
            if mid * mid == x:
                return mid
            if mid * mid < x:
                l = mid+1
            else:
                r = mid
        return l-1
```

---



### 239-滑动窗口最大值

给你一个整数数组 `nums`，有一个大小为 `k` 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 `k` 个数字。滑动窗口每次只向右移动一位。

返回 *滑动窗口中的最大值* 。

 **思路**

使用一个双端队列。队列中仅需要维持当前窗口的最大值即可。入队的时候，需要将所有小于当前要入队的值的元素全部去掉。因为考虑到可能有重复的元素，因此判断范围是小于而不是小于等于。当然还要考虑从滑动窗口移动的时候，我们要判断现在最大的值是不是刚好需要去掉的值（nums[i-k]），如果是的话就要从前端去掉这一个值。

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        deque = []
        n = len(nums)
        ans = []
        for i in range(n):
            if i < k:
                while deque and deque[-1] < nums[i]:
                    deque.pop()
                deque.append(nums[i])
                if i == k-1:
                    ans.append(deque[0])
            else:
                if deque and deque[0] == nums[i-k]:
                    deque.pop(0)
                while deque and deque[-1] < nums[i]:
                    deque.pop()
                deque.append(nums[i])
                ans.append(deque[0])
        return ans
```

---
