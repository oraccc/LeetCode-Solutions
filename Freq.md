# LeetCode 高频题



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



### 5-最长回文子串

给你一个字符串 `s`，找到 `s` 中最长的 回文子串。

**思路**

由于回文字符串可以通过向外面添加两个相同的字符进行延伸，因此考虑使用dp\[i]\[j]，代表从字符串的i位置到字符串的j位置的字符是不是回文。考虑到递推公式是

`dp[i+1][j-1] and s[i] == s[j]`，注意到在i位置的时候会需要后面的信息，因此我们对于i要倒着遍历，而j则是正常的遍历。可以画图更好地理解这个思路。

注意对于长度是1或者2的字符串要特别处理。

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        dp = [[False] * n for _ in range(n)]
        max_len = 1
        max_start = 0

        for i in range(n-1, -1, -1):
            for j in range(i, n):
                if j-i <= 1 and s[i] == s[j]:
                    dp[i][j] = True
                    if j-i+1 > max_len:
                        max_len = j-i+1
                        max_start = i 
                elif dp[i+1][j-1] and s[i] == s[j]:
                    dp[i][j] = True
                    if j-i+1 > max_len:
                        max_len = j-i+1
                        max_start = i
        
        return s[max_start: max_start+max_len]
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



### 23-合并K个升序链表

给你一个链表数组，每个链表都已经按升序排列。

请你将所有链表合并到一个升序链表中，返回合并后的链表。

**思路**

维持一个最小堆，大小为K，每一次只需要pop堆内的最小的元素就可以了，但是要注意ListNode没有自带的大小比较，因此需要提前写一个规则来比较大小。

`ListNode.__lt__ = lambda a, b: a.val < b.val`



```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        ListNode.__lt__ = lambda a, b: a.val < b.val
        if not lists:
            return None
        heap = [head for head in lists if head]
        heapq.heapify(heap)
        dummy_head = ListNode(-1)
        curr = dummy_head
        while heap:
            node = heapq.heappop(heap)
            if node.next:
                heapq.heappush(heap, node.next)
            curr.next = node
            curr = curr.next
        return dummy_head.next
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



### 31-下一个排列

整数数组的一个 **排列** 就是将其所有成员以序列或线性顺序排列。

- 例如，`arr = [1,2,3]` ，以下这些都可以视作 `arr` 的排列：`[1,2,3]`、`[1,3,2]`、`[3,1,2]`、`[2,3,1]` 。

整数数组的 **下一个排列** 是指其整数的下一个字典序更大的排列。更正式地，如果数组的所有排列根据其字典顺序从小到大排列在一个容器中，那么数组的 **下一个排列** 就是在这个有序容器中排在它后面的那个排列。如果不存在下一个更大的排列，那么这个数组必须重排为字典序最小的排列（即，其元素按升序排列）。

- 例如，`arr = [1,2,3]` 的下一个排列是 `[1,3,2]` 。
- 类似地，`arr = [2,3,1]` 的下一个排列是 `[3,1,2]` 。
- 而 `arr = [3,2,1]` 的下一个排列是 `[1,2,3]` ，因为 `[3,2,1]` 不存在一个字典序更大的排列。

给你一个整数数组 `nums` ，找出 `nums` 的下一个排列。

必须 原地修改，只允许使用额外常数空间。

**思路**

* 我们希望下一个数 比当前数大，这样才满足 “下一个排列” 的定义。因此只需要 将后面的「大数」与前面的「小数」交换，就能得到一个更大的数。比如 123456，将 5 和 6 交换就能得到一个更大的数 123465。
* 我们还希望下一个数 增加的幅度尽可能的小，这样才满足“下一个排列与当前排列紧邻“的要求。为了满足这个要求，我们需要：
  * 在 尽可能靠右的低位 进行交换，需要 从后向前 查找
  * 将一个 尽可能小的「大数」 与前面的「小数」交换。比如 123465，下一个排列应该把 5 和 4 交换而不是把 6 和 4 交换
  * 将「大数」换到前面后，需要将「大数」后面的所有数 重置为升序，升序排列就是最小的排列。以 123465 为例：首先按照上一步，交换 5 和 4，得到 123564；然后需要将 5 之后的数重置为升序，得到 123546。显然 123546 比 123564 更小，123546 就是 123465 的下一个排列

```python
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        if n <= 1:
            return
        
        i = n-2
        j = n-1
        while i >= 0 and nums[i] >= nums[j]:
            i -= 1
            j -= 1
        
        if i == -1:
            left = 0
            right = n-1
            while left < right:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
                right -= 1
            return
        else:
            k = n-1
            while k > i and nums[i] >= nums[k]:
                k -= 1
            nums[i], nums[k] = nums[k], nums[i]
            left = j
            right = n-1
            while left < right:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
                right -= 1
            return
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



### 39-组合总和

给你一个 **无重复元素** 的整数数组 `candidates` 和一个目标整数 `target` ，找出 `candidates` 中可以使数字和为目标数 `target` 的 所有 **不同组合** ，并以列表形式返回。你可以按 **任意顺序** 返回这些组合。

`candidates` 中的 **同一个** 数字可以 **无限制重复被选取** 。如果至少一个数字的被选数量不同，则两种组合是不同的。 

对于给定的输入，保证和为 `target` 的不同组合数少于 `150` 个。

 **思路**

使用回溯算法，每次循环的时候范围取当前位置到最后一个位置。注意当前位置不是必须得取的，因此不要在循环的外面append和pop，不然的话答案一定会包含当前位置的数，不符合题意。

```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        n = len(candidates)
        curr = []
        ans = []
        def backtracking(pos):
            if sum(curr) > target:
                return
            if sum(curr) == target:
                ans.append(curr[:])
            
            for i in range(pos, n):
                curr.append(candidates[i])
                backtracking(i)
                curr.pop()

        backtracking(0)
        return ans
```

---



### 41-缺失的第一个正数

给你一个未排序的整数数组 `nums` ，请你找出其中没有出现的最小的正整数。

请你实现时间复杂度为 `O(n)` 并且只使用常数级别额外空间的解决方案。

**思路**

我们将数组中所有小于等于 0 的数修改为 N+1；

我们遍历数组中的每一个数 x，它可能已经被打了标记，因此原本对应的数为 ∣x∣，其中 ∣∣ 为绝对值符号。如果 ∣x∣∈[1,N]，那么我们给数组中的第 ∣x∣−1 个位置的数添加一个负号。注意如果它已经有负号，不需要重复添加；

在遍历完成之后，如果数组中的每一个数都是负数，那么答案是 N+1，否则答案是第一个正数的位置加 1。

```python
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        n = len(nums)
        for i in range(n):
            if nums[i] <= 0:
                nums[i] = n+1
        for i in range(n):
            original_num = abs(nums[i])
            if original_num <= n:
                if nums[original_num-1] > 0:
                    nums[original_num-1] = -nums[original_num-1]
                    
        for i in range(n):
            if nums[i] > 0:
                return i+1
        
        return n+1
```

---



### 42-接雨水

给定 `n` 个非负整数表示每个宽度为 `1` 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

<img src="https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/22/rainwatertrap.png" style="zoom: 80%;" />

**思路**

对于每一个位置，它可以承接的水的多少取决于左边最高的墙和右边最高的墙之间较小的那个值，因此我们只需要记录，对于每一个位置，它左边最高的位置是多少，和它右边最高的位置是多少即可。用两个数组遍历两次存储即可。

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        n = len(height)
        left_high = [0]*n
        right_high = [0]*n
        left_high[0] = height[0]
        right_high[-1] = height[-1]
        for i in range(1, n):
            if height[i] > left_high[i-1]:
                left_high[i] = height[i]
            else:
                left_high[i] = left_high[i-1]

        for i in range(n-2, -1, -1):
            if height[i] > right_high[i+1]:
                right_high[i] = height[i]
            else:
                right_high[i] = right_high[i+1]
        
        ans = 0
        for i in range(1, n-1):
            ans += (min(left_high[i], right_high[i]) - height[i])
        return ans
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



### 48-旋转图像

给定一个 *n* × *n* 的二维矩阵 `matrix` 表示一个图像。请你将图像顺时针旋转 90 度。

你必须在 原地旋转图像，这意味着你需要直接修改输入的二维矩阵。**请不要** 使用另一个矩阵来旋转图像。

**思路**

先按照正对角线翻折矩阵，接着让矩阵纵向对称翻折

注意这是一个原地的算法，因此如果需要原地交换两个数组，最好的办法就是依次交换两个数组内每一个元素的值

```python
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)
        for i in range(n):
            for j in range(i):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        
        for j in range(n//2):
            for i in range(n):
                matrix[i][j], matrix[i][n-1-j] = matrix[i][n-1-j], matrix[i][j]
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



### 76-最小覆盖子串

给你一个字符串 `s` 、一个字符串 `t` 。返回 `s` 中涵盖 `t` 所有字符的最小子串。如果 `s` 中不存在涵盖 `t` 所有字符的子串，则返回空字符串 `""` 。

**思路**

使用双指针，首先遍历一遍t，记录需要标记的char和这些char出现的次数

接着右指针开始依次遍历s，如果当前的char是t中的，那么就将出现次数-1，只要减了之后还是大于等于0的，那么说明现在就是有效的覆盖，若变成负数了那就是多出了了char。当全部的字母覆盖完毕之后，移动左边的指针，使其对应的标记char+1，如果超过了0，说明有没有被覆盖到的情况，接着需要继续移动右指针。

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        n = len(t)
        unseen_flag = [False]*128
        unseen_char = [0]*128

        for i in range(n):
            unseen_flag[ord(t[i])] = True
            unseen_char[ord(t[i])] += 1
        
        left = 0
        right = 0
        count = 0
        min_start = 0
        min_len = len(s)+1

        while right < len(s):
            char = s[right]
            if unseen_flag[ord(char)] == True:
                unseen_char[ord(char)] -= 1
                if unseen_char[ord(char)] >= 0:
                    count += 1
            while count == n:
                if right-left+1 < min_len:
                    min_start = left
                    min_len = right-left+1
                if unseen_flag[ord(s[left])]:
                    unseen_char[ord(s[left])] += 1
                    if unseen_char[ord(s[left])] > 0:
                        count -= 1
                left += 1

            right += 1
        if min_len > len(s):
            return ""
        return s[min_start:min_start+min_len]
```

---



### 82-删除排序链表中的重复元素II

给定一个已排序的链表的头 `head` ， *删除原始链表中所有重复数字的节点，只留下不同的数字* 。返回 *已排序的链表* 。

**思路**

本题的难点在于如何删除连续的重复的元素，比如1233445要删去3和4，我们可以设立一个flag来作为标志，代表目前遇到了重复的数字，当当前的节点与下一个节点的值一样的时候，将flag设置为True。若不一样，则需要检查flag的值，如果是True说明需要跳过一系列的元素，反之则不需要。注意跳过元素之后不要急着把prev也往前移一位，因为很有可能是连续的重复数字。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return head
        dummy_head = ListNode(-1)
        dummy_head.next = head
        prev = dummy_head
        curr = head
        flag = False

        while curr:
            if not curr.next:
                if not flag:
                    prev.next = curr
                    curr = curr.next
                else:
                    prev.next = None
                    curr = curr.next
            else:
                if curr.val == curr.next.val:
                    curr = curr.next
                    flag = True
                else:
                    if not flag:
                        prev = curr
                        curr = curr.next
                    else:
                        prev.next = curr.next
                        curr = curr.next 
                    flag = False
        return dummy_head.next
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



### 92-反转链表II

给你单链表的头指针 `head` 和两个整数 `left` 和 `right` ，其中 `left <= right` 。请你反转从位置 `left` 到位置 `right` 的链表节点，返回 **反转后的链表** 。

 **思路**

在反转之前提前记录前后的位置即可。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:

        def reverse_link(node):
            prev = None
            curr = node
            while curr:
                tmp = curr.next
                curr.next = prev
                prev = curr
                curr = tmp
            return prev

        dummy_head = ListNode(-1)
        prev = dummy_head
        dummy_head.next = head

        for i in range(left-1):
            prev = prev.next
        
        curr = prev
        for i in range(right-left+1):
            curr = curr.next 

        tail = curr.next
        curr.next = None


        prev.next = reverse_link(prev.next)

        while prev.next:
            prev = prev.next
        prev.next = tail

        return dummy_head.next
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



### 101-对称二叉树

给你一个二叉树的根节点 `root` ， 检查它是否轴对称。

**思路**

使用递归，依次检测两个节点（左和右）是不是相同的

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        def helper(left, right):
            if not left and not right:
                return True
            elif not left or not right:
                return False
            elif left.val != right.val:
                return False
            else:
                return helper(left.left, right.right) and helper(left.right, right.left)
        return helper(root.left, root.right)
        
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



### 103-二叉树的锯齿形层序遍历

给你二叉树的根节点 `root` ，返回其节点值的 **锯齿形层序遍历** 。（即先从左往右，再从右往左进行下一层遍历，以此类推，层与层之间交替进行）。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        flag = True
        ans = []
        if not root:
            return ans 
        queue = [root]
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
            if not flag:
                tmp = tmp[::-1]
            ans.append(tmp)
            flag = not flag
        return ans
            
```

---



### 105-从前序与中序遍历构造二叉树

给定两个整数数组 `preorder` 和 `inorder` ，其中 `preorder` 是二叉树的**先序遍历**， `inorder` 是同一棵树的**中序遍历**，请构造二叉树并返回其根节点。

**思路**

因为当前先序遍历的第一个数就是根节点，再根据根节点到inorder中去确定左子树的长度，这样便可以依次递归下去了。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:

        if not preorder or not inorder:
            return None
        val = preorder[0]
        root = TreeNode(val)
        in_pos = inorder.index(val)
        left_len = in_pos
        root.left = self.buildTree(preorder[1:1+left_len], inorder[0:in_pos])
        root.right = self.buildTree(preorder[1+left_len:], inorder[in_pos+1:])
        return root
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



### 124-二叉树中的最大路径和

二叉树中的 **路径** 被定义为一条节点序列，序列中每对相邻节点之间都存在一条边。同一个节点在一条路径序列中 **至多出现一次** 。该路径 **至少包含一个** 节点，且不一定经过根节点。

**路径和** 是路径中各节点值的总和。

给你一个二叉树的根节点 `root` ，返回其 **最大路径和** 。

**思路**

这道题的思路和543二叉树的直径是一样的，因为最大的路径和不一定会经过根节点，因此我们需要用一个全局变量来记录最大的路径和。注意dfs返回的应该是经过根节点的最长路径和（且以根节点结尾）：如果经过这个根节点的最大路径和大于0，那么可以返回这个值，代表拼接上这个根节点。反之就返回0，代表计算的时候不要带上这个根节点。

更新最长的路径应该是

`max_value = max(max_value, left_value+right_value+node.val)`

代表当前已根节点为中间节点的最大路径和

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        max_value = float("-inf")

        def dfs_helper(node):
            nonlocal max_value
            if not node:
                return 0
            left_value = dfs_helper(node.left)
            right_value = dfs_helper(node.right)
            max_value = max(max_value, left_value+right_value+node.val)
            return max(0, node.val+max(left_value, right_value))
        dfs_helper(root)
        return max_value

```

---



### 129-求根节点到叶节点数字之和

给你一个二叉树的根节点 `root` ，树中每个节点都存放有一个 `0` 到 `9` 之间的数字。

每条从根节点到叶节点的路径都代表一个数字：

- 例如，从根节点到叶节点的路径 `1 -> 2 -> 3` 表示数字 `123` 。

计算从根节点到叶节点生成的 **所有数字之和** 。

**叶节点** 是指没有子节点的节点。

**思路**

dfs递归子节点，如果当前节点没有左子树和右子树了，那就是叶子节点，将当前的curr的值加入答案中。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        ans = 0

        def dfs(curr, node):
            nonlocal ans 
            if not node:
                return
            curr = curr*10 + node.val
            if not node.left and not node.right:
                ans += curr
                return
            dfs(curr, node.left)
            dfs(curr, node.right)
        
        dfs(0, root)
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



### 143-重排链表

给定一个单链表 `L` 的头节点 `head` ，单链表 `L` 表示为：

```
L0 → L1 → … → Ln - 1 → Ln
```

请将其重新排列后变为：

```
L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …
```

不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。

**思路**

首先将链表对半分，然后逆转右边的链表，然后交错拼接两个链表。

注意左边的链表比右边的链表长度要么相等，要么长一个，交错拼接的时候需要条件是`l1 and l2`

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reorderList(self, head: Optional[ListNode]) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        if not head or not head.next:
            return head
        slow = fast = head
        while fast.next and fast.next.next:
            fast = fast.next.next
            slow = slow.next

        half = slow.next
        slow.next = None

        def reverse_link(head):
            prev = None
            curr = head
            while curr:
                tmp = curr.next
                curr.next = prev
                prev = curr
                curr = tmp
            return prev
        
        half = reverse_link(half)

        l1 = head
        l2 = half 
        while l1 and l2:
            tmp1 = l1.next
            tmp2 = l2.next
            l1.next = l2
            l2.next = tmp1
            l1 = tmp1
            l2 = tmp2
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



### 155-最小栈

设计一个支持 `push` ，`pop` ，`top` 操作，并能在常数时间内检索到最小元素的栈。

实现 `MinStack` 类:

- `MinStack()` 初始化堆栈对象。
- `void push(int val)` 将元素val推入堆栈。
- `void pop()` 删除堆栈顶部的元素。
- `int top()` 获取堆栈顶部的元素。
- `int getMin()` 获取堆栈中的最小元素。

**思路**

设计一个最小的栈，只有当前输入小于等于之前的最小值时，才可以将其append进去，pop的时候也只能将等于目前最小值的数pop出去。

```python
class MinStack:

    def __init__(self):
        self.s = []
        self.min_s = []

    def push(self, val: int) -> None:
        self.s.append(val)
        if not self.min_s or self.min_s[-1] >= val:
            self.min_s.append(val)

    def pop(self) -> None:
        val = self.s.pop()
        if self.min_s and self.min_s[-1] == val:
            self.min_s.pop()

    def top(self) -> int:
        return self.s[-1]

    def getMin(self) -> int:
        return self.min_s[-1]



# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(val)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()
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



### 165-比较版本号

给你两个 **版本号字符串** `version1` 和 `version2` ，请你比较它们。版本号由被点 `'.'` 分开的修订号组成。**修订号的值** 是它 **转换为整数** 并忽略前导零。

比较版本号时，请按 **从左到右的顺序** 依次比较它们的修订号。如果其中一个版本字符串的修订号较少，则将缺失的修订号视为 `0`。

返回规则如下：

- 如果 `*version1* < *version2*` 返回 `-1`，
- 如果 `*version1* > *version2*` 返回 `1`，
- 除此之外返回 `0`。

**思路**

字符串分割，转int比较

```python
class Solution:
    def compareVersion(self, version1: str, version2: str) -> int:
        l1 = version1.split(".")
        l2 = version2.split(".")

        for i in range(max(len(l1), len(l2))):
            num1 = int(l1[i]) if i < len(l1) else 0
            num2 = int(l2[i]) if i < len(l2) else 0
            if num1 > num2:
                return 1
            elif num1 < num2:
                return -1
        
        return 0
```

---



### 169-多数元素

给定一个大小为 `n` 的数组 `nums` ，返回其中的多数元素。多数元素是指在数组中出现次数 **大于** `⌊ n/2 ⌋` 的元素。你可以假设数组是非空的，并且给定的数组总是存在多数元素。

**思路**

Boyer-Moore 投票算法，如果我们把众数记为 +1，把其他数记为 −1，将它们全部加起来，显然和大于 `0`，从结果本身我们可以看出众数比其他数多。

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        ans = nums[0]
        count = 1
        for i in range(1, len(nums)):
            if nums[i] == ans:
                count += 1
            else:
                count -= 1
                if count == 0:
                    count = 1
                    ans = nums[i]
        return ans
```

---



### 199-二叉树的右视图

给定一个二叉树的 **根节点** `root`，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。

**思路1**

层次遍历，每次返回层次遍历中的最后一个值

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        ans = []
        if not root:
            return ans
        queue = []
        queue.append(root)
        while queue:
            n = len(queue)
            curr = []
            for _ in range(n):
                node = queue.pop(0)
                curr.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            ans.append(curr[-1])
        return ans
```

**思路2**

递归，先递归右子树，再递归左子树，当某个深度首次到达时，对应的节点就在右视图中。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        ans = []
        def dfs(node, depth):
            if not node:
                return
            if len(ans) == depth:
                ans.append(node.val)
            dfs(node.right, depth+1)
            dfs(node.left, depth+1)

        dfs(root, 0)
        return ans
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



### 212-单词搜索II

给定一个 `m x n` 二维字符网格 `board` 和一个单词（字符串）列表 `words`， *返回所有二维网格上的单词* 。

单词必须按照字母顺序，通过 **相邻的单元格** 内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母在一个单词中不允许被重复使用。

**思路**

此题因为需要匹配的目标有多个单词，因此我们可以使用前缀树去进行一个记录这些单词。这里的前缀树可以稍微修改一下，树的节点可以多一个属性word，记录以当前节点结尾的单词是什么。在dfs的时候，需要传入当前的树节点。如果该树节点的下一个单词节点是None，那么就说明现在没有任何匹配的，也就可以直接返回了。如果当前下一个树节点的word不是空的，说明找到了完整的单词，因此可以返回了。最后注意用set去重。

```python
class TrieNode:
    def __init__(self):
        self.children = [None]*26
        self.word = ""


class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        curr = self.root
        for c in word:
            pos = ord(c)-ord('a')
            if not curr.children[pos]:
                curr.children[pos] = TrieNode()
            curr = curr.children[pos]
        curr.word = word


class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        trie = Trie()
        for word in words:
            trie.insert(word)
        
        ans = []
        n = len(board)
        m = len(board[0])
        direction = [-1, 0, 1, 0, -1]
        visited = [[False]*m for _ in range(n)]

        def backtracking(curr, i, j):
            char = board[i][j]
            curr = curr.children[ord(char)-ord('a')]

            if not curr:
                return
            if curr.word != "":
                ans.append(curr.word)
            visited[i][j] = True
            for k in range(4):
                row = i + direction[k]
                col = j + direction[k+1]
                if row >= 0 and row < n and col >= 0 and col < m and not visited[row][col]:
                    backtracking(curr, row, col)
            visited[i][j] = False


        for i in range(n):
            for j in range(m):
                backtracking(trie.root, i, j)
        
        return list(set(ans))
        
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



### 221-最大正方形

在一个由 `'0'` 和 `'1'` 组成的二维矩阵内，找到只包含 `'1'` 的最大正方形，并返回其面积。

**思路**

<img src="https://pic.leetcode-cn.com/8c4bf78cf6396c40291e40c25d34ef56bd524313c2aa863f3a20c1f004f32ab0-image.png" style="zoom: 80%;" />

min(上, 左, 左上) + 1

```python
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        n = len(matrix)
        m = len(matrix[0])

        dp = [[0]*m for _ in range(n)]
        max_len = 0
        for i in range(n):
            if matrix[i][0] == "1":
                dp[i][0] = 1
                max_len = 1
        
        for j in range(m):
            if matrix[0][j] == "1":
                dp[0][j] = 1
                max_len = 1
        
        for i in range(1,n):
            for j in range(1,m):
                if matrix[i][j] == "1":
                    dp[i][j] = min(dp[i-1][j], dp[i-1][j-1], dp[i][j-1])+1
                    max_len = max(dp[i][j], max_len)

        return max_len * max_len
```

---



### 236-二叉树的最近公共祖先

给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个节点 p、q，最近公共祖先表示为一个节点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（**一个节点也可以是它自己的祖先**）。”

**思路**

如果当前节点就是p或者q，由于我们是从上往下遍历的，所以可以直接返回p或者q，这就是当前分支上祖先。对于每一个节点，检查其左边的分支和右边的分支是否都有返回，如果都有的话，那就是说明p，q正好在其一左一右，因此当前节点是公共祖先。如果只有一边的话，说明公共祖先在其中的一侧，返回那一侧的公共祖先。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root:
            return None
        if root == p or root == q:
            return root
        
        left_return = self.lowestCommonAncestor(root.left, p, q)
        right_return = self.lowestCommonAncestor(root.right, p, q)
        if left_return and right_return:
            return root
        else:
            return left_return if left_return else right_return
        
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



### 322-零钱兑换

给你一个整数数组 `coins` ，表示不同面额的硬币；以及一个整数 `amount` ，表示总金额。

计算并返回可以凑成总金额所需的 **最少的硬币个数** 。如果没有任何一种硬币组合能组成总金额，返回 `-1` 。

你可以认为每种硬币的数量是无限的。

**思路**

该题的思路也是动态规划，可以参考“279-完全平方数”的写法。注意有可能不能凑成总金额。

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [amount+1] * (amount+1)
        dp[0] = 0
        for i in range(1, amount+1):
            for coin in coins:
                if coin <= i:
                    dp[i] = min(dp[i], dp[i-coin]+1)
        if dp[amount] == amount+1:
            return -1
        return dp[amount]
```

---



### 415-字符串相加

给定两个字符串形式的非负整数 `num1` 和`num2` ，计算它们的和并同样以字符串形式返回。

你不能使用任何內建的用于处理大整数的库（比如 `BigInteger`）， 也不能直接将输入的字符串转换为整数形式。

**思路**

把字符串倒过来处理

```python
class Solution:
    def addStrings(self, num1: str, num2: str) -> str:
        num1 = num1[::-1]
        num2 = num2[::-1]

        n = len(num1)
        m = len(num2)

        carry = 0
        result = ""
        i = 0
        while i < max(n, m) or carry:
            digit1 = int(num1[i]) if i < n else 0
            digit2 = int(num2[i]) if i < m else 0
            curr_sum = (digit1+digit2+carry) % 10
            carry = (digit1+digit2+carry) // 10
            result += str(curr_sum)
            i += 1
        
        return result[::-1]
        
        
```

---



### 426-将二叉搜索树转化为排序的双向链表

将一个 **二叉搜索树** 就地转化为一个 **已排序的双向循环链表** 。

对于双向循环列表，你可以将左右孩子指针作为双向循环链表的前驱和后继指针，第一个节点的前驱是最后一个节点，最后一个节点的后继是第一个节点。

特别地，我们希望可以 **就地** 完成转换操作。当转化完成以后，树中节点的左指针需要指向前驱，树中节点的右指针需要指向后继。还需要返回链表中最小元素的指针。

**思路**

类似于中序遍历这个二叉搜索树，需要一个前驱节点prev来记录当前节点的前面，这样便可以链接。如果一个节点没有前驱节点，那就是二叉搜索树的最左边的节点。每次遍历的时候都需要把当前节点变成前驱节点。遍历结束之后prev就在最右边的位置。

注意最后还需要将prev和head两个点再串起来。

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
"""
class Solution:
    def treeToDoublyList(self, root: 'Node') -> 'Node':
        if not root:
            return None
        self.prev = None

        def helper(node):
            if not node:
                return
            helper(node.left)
            if self.prev:
                self.prev.right = node
                node.left = self.prev
            else:
                self.head = node
            self.prev = node
            helper(node.right)
        
        helper(root)

        self.head.left = self.prev
        self.prev.right = self.head

        return self.head

            
        
```

---



### 540-有序数组中的单一元素

给你一个仅由整数组成的有序数组，其中每个元素都会出现两次，唯有一个数只会出现一次。

请你找出并返回只出现一次的那个数。

你设计的解决方案必须满足 `O(log n)` 时间复杂度和 `O(1)` 空间复杂度。

**思路**

因为需要满足时间复杂度，所以需要二分。根据mid位置是在偶数还是在奇数，我们可以判断出mid应该和mid+1相等还是应该和mid-1位置相同。如果满足相同的位置，说明在mid位置之前都是有序的，反之就是在mid位置之前没有序。不断二分逼近找到位置。

```python
class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        left = 0
        right = len(nums)-1

        while left < right:
            mid = (left+right)//2
            if mid % 2 == 1:
                if nums[mid] == nums[mid-1]:
                    left = mid+1
                else:
                    right = mid 
            else:
                if nums[mid] == nums[mid+1]:
                    left = mid+1
                else:
                    right = mid
        return nums[left]
```

---



### 543-二叉树的直径

给你一棵二叉树的根节点，返回该树的 **直径** 。

二叉树的 **直径** 是指树中任意两个节点之间最长路径的 **长度** 。这条路径可能经过也可能不经过根节点 `root` 。

两节点之间路径的 **长度** 由它们之间边数表示。

**思路**

本质上时记录**左右子树深度之和的最大值**。helper函数就是用来求当前根节点的最大深度。

由于最长的路径不一定会经过当前的root节点，因此递归函数返回的值不一定就是最长路径。因此需要设置一个全局的变量来记录这一个值。而递归函数只是返回以root节点为路径的一个边的最长路径，即`1+max(left_max, right_max)`，（因为左边的路径到root距离要再加一个，右边也是）。然而实际判断最长路径的操作实在每次递归结束后，判断当前左路径加上右路径是否是最长的。

注意nonlocal修饰符，`nonlocal diameter` 声明告诉 Python，`diameter` 变量不是 `helper` 函数的局部变量，而是外层 `diameterOfBinaryTree` 函数的局部变量。这样，`helper` 函数就可以正确地修改 `diameter` 变量的值了。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        diameter = 0

        def helper(node):
            nonlocal diameter
            if not node:
                return 0
            left_max = helper(node.left)
            right_max = helper(node.right)
            diameter = max(diameter, left_max+right_max)
            return 1+max(left_max, right_max)
        
        helper(root)
        return diameter
```

---



### 628-三个数的最大乘积

给你一个整型数组 `nums` ，在数组中找出由三个数组成的最大乘积，并输出这个乘积。

**思路**

首先将数组排序。

如果数组中全是非负数，则排序后最大的三个数相乘即为最大乘积；如果全是非正数，则最大的三个数相乘同样也为最大乘积。

如果数组中有正数有负数，则最大乘积既可能是三个最大正数的乘积，也可能是两个最小负数（即绝对值最大）与最大正数的乘积。

```python
class Solution:
    def maximumProduct(self, nums: List[int]) -> int:
        nums.sort()
        if nums[0] >= 0:
            return nums[-1]*nums[-2]*nums[-3]
        elif nums[-1] <= 0:
            return nums[-1]*nums[-2]*nums[-3]
        else:
            max_value = max(nums[-1]*nums[-2]*nums[-3], nums[0]*nums[1]*nums[-1])
            return max_value
```

---



### 662-二叉树最大宽度

给你一棵二叉树的根节点 `root` ，返回树的 **最大宽度** 。

树的 **最大宽度** 是所有层中最大的 **宽度** 。

每一层的 **宽度** 被定义为该层最左和最右的非空节点（即，两个端点）之间的长度。将这个二叉树视作与满二叉树结构相同，两端点间会出现一些延伸到这一层的 `null` 节点，这些 `null` 节点也计入长度。

题目数据保证答案将会在 **32 位** 带符号整数范围内。

**思路**

当层序遍历的时候，也放入当前节点在这一层的编号，若上一层的编号为i，那么这一层左节点编号是`2*i`，右节点编号是`2*i+1`，计算最大宽度的时候最左边和最右边的编号相减即可。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def widthOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        max_width = 1
        if not root:
            return 0
        queue = [(root, 0)]

        while queue:
            n = len(queue)
            for _ in range(n):
                node, idx = queue.pop(0)
                if node.left:
                    queue.append((node.left, 2*idx))
                if node.right:
                    queue.append((node.right, 2*idx+1))
            
            if queue:
                max_width = max(max_width, queue[-1][1]-queue[0][1]+1)

        return max_width
```

---



### 695-岛屿的最大面积

给你一个大小为 `m x n` 的二进制矩阵 `grid` 。

**岛屿** 是由一些相邻的 `1` (代表土地) 构成的组合，这里的「相邻」要求两个 `1` 必须在 **水平或者竖直的四个方向上** 相邻。你可以假设 `grid` 的四个边缘都被 `0`（代表水）包围着。

岛屿的面积是岛上值为 `1` 的单元格的数目。

计算并返回 `grid` 中最大的岛屿面积。如果没有岛屿，则返回面积为 `0` 。

**思路**

与第200题的思路一致，只是dfs返回的时候返回的是当前岛屿的面积。

```python
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        max_area = 0
        direction = [-1, 0, 1, 0, -1]
        n = len(grid)
        m = len(grid[0])

        def dfs_helper(i, j):
            if grid[i][j] == 0:
                return 0
            grid[i][j] = 0
            count = 1
            for k in range(4):
                row = i+direction[k]
                col = j+direction[k+1]
                if row >= 0 and row < n and col >= 0 and col < m:
                    count += dfs_helper(row, col)
            return count
        
        for i in range(n):
            for j in range(m):
                if grid[i][j]:
                    area = dfs_helper(i, j)
                    max_area = max(area, max_area)
        
        return max_area
            
```

---



### 752-打开转盘锁

你有一个带有四个圆形拨轮的转盘锁。每个拨轮都有10个数字： `'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'` 。每个拨轮可以自由旋转：例如把 `'9'` 变为 `'0'`，`'0'` 变为 `'9'` 。每次旋转都只能旋转一个拨轮的一位数字。

锁的初始数字为 `'0000'` ，一个代表四个拨轮的数字的字符串。

列表 `deadends` 包含了一组死亡数字，一旦拨轮的数字和列表里的任何一个元素相同，这个锁将会被永久锁定，无法再被旋转。

字符串 `target` 代表可以解锁的数字，你需要给出解锁需要的最小旋转次数，如果无论如何不能解锁，返回 `-1` 。

**思路**

使用广度优先搜索queue来找到最短的路径，每个str有8个邻居。同时也需要设立visited的集合，防止重复访问某些位置。注意如果0000就在deadends或者在target中，就可以直接返回结果了。

```python
class Solution:
    def openLock(self, deadends: List[str], target: str) -> int:

        if "0000" in deadends:
            return -1
        if target == "0000":
            return 0
        
        def change(s):
            new_s = []
            for i in range(4):
                digit = int(s[i])
                for j in [-1, 1]:
                    new_d = (digit+j) % 10
                    new_s.append(s[:i] + str(new_d) + s[i+1:])
            return new_s

        steps = 0
        queue = ["0000"]
        deadends = set(deadends)
        visited = set()
        visited.add("0000")

        while queue:
            steps += 1
            n = len(queue)
            for _ in range(n):
                curr = queue.pop(0)
                nxt = change(curr)
                for each in nxt:
                    if each == target:
                        return steps
                    if each in visited or each in deadends:
                        continue
                    visited.add(each)
                    queue.append(each)
        
        return -1

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



### Extra-最大子矩阵

给定一个正整数、负整数和 0 组成的 N × M 矩阵，编写代码找出元素总和最大的子矩阵。

返回一个数组 `[r1, c1, r2, c2]`，其中 `r1`, `c1` 分别代表子矩阵左上角的行号和列号，`r2`, `c2` 分别代表右下角的行号和列号。若有多个满足条件的子矩阵，返回任意一个均可。

**思路**

首先计算出全部的前缀和，方便后续计算

然后固定一个bottom和一个top，从左往右计算当前的矩阵和，如果当前矩阵和小于零，那么就抛弃之前计算的全部结果，直接另起炉灶从下一个位置重新开始计算。

```python
class Solution:
    def getMaxMatrix(self, matrix: List[List[int]]) -> List[int]:
        ans = []
        n = len(matrix)
        m = len(matrix[0])
        pre_sum = [[0]*(m+1) for _ in range(n+1)]

        gloal_max = float("-inf")
        for i in range(1, n+1):
            for j in range(1, m+1):
                pre_sum[i][j] = matrix[i-1][j-1]+pre_sum[i-1][j]+pre_sum[i][j-1]-pre_sum[i-1][j-1]

        for top in range(n):
            for bottom in range(top,n):
                local_max = 0
                left = 0
                for right in range(m):
                    local_max = pre_sum[bottom+1][right+1]-pre_sum[bottom+1][left]-pre_sum[top][right+1]+pre_sum[top][left]
                    if local_max > gloal_max:
                        gloal_max = local_max
                        ans = [top, left, bottom, right]
                    if local_max < 0:
                        left = right+1
        return ans

```

---

