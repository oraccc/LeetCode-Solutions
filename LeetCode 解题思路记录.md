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



### 2-两数相加

给你两个 **非空** 的链表，表示两个非负的整数。它们每位数字都是按照 **逆序** 的方式存储的，并且每个节点只能存储 **一位** 数字。

请你将两个数相加，并以相同形式返回一个表示和的链表。

你可以假设除了数字 0 之外，这两个数都不会以 0 开头。

<img src="https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2021/01/02/addtwonumber1.jpg" style="zoom: 67%;" />

**思路**

用做加法的思路，注意进位，直到两个链表都走完了并且也不需要进位了，才停止遍历。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dummy_head = ListNode(-1)
        curr = dummy_head
        res = carry = 0
        while l1 or l2 or carry:
            num1 = l1.val if l1 else 0
            num2 = l2.val if l2 else 0
            res = (num1+num2+carry) % 10
            carry = (num1+num2+carry) // 10
            curr.next = ListNode(res)
            curr = curr.next
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None
        return dummy_head.next
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



### 17-电话号码的字母组合

给定一个仅包含数字 `2-9` 的字符串，返回所有它能表示的字母组合。答案可以按 **任意顺序** 返回。

给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。

![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2021/11/09/200px-telephone-keypad2svg.png)

**思路**

使用回溯进行处理，注意进入回溯和离开回溯的前后状态应该是一致的。

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        letter_dict = {
            "2": ["a", "b", "c"],
            "3": ["d", "e", "f"],
            "4": ["g", "h", "i"],
            "5": ["j", "k", "l"],
            "6": ["m", "n", "o"],
            "7": ["p", "q", "r", "s"],
            "8": ["t", "u", "v"],
            "9": ["w", "x", "y", "z"]
        }
        curr = []
        ans = []
        if len(digits) == 0:
            return ans

        def backtracking(i):
            if i == len(digits):
                ans.append("".join(curr[:]))
                return
            letter_list = letter_dict[digits[i]]
            for letter in letter_list:
                curr.append(letter)
                backtracking(i+1)
                curr.pop()
        backtracking(0)
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



### 24-两两交换链表中的节点

给你一个链表，两两交换其中相邻的节点，并返回交换后链表的头节点。你必须在不修改节点内部的值的情况下完成本题（即，只能进行节点交换）。

**思路**

注意需要设置一个prev节点，它和curr节点在需要交换位置的时候，相差两个位置，这样便可完成指针的交换，可以画图来帮助指针的下一位交换。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy_head = ListNode(-1)
        dummy_head.next = head
        prev = dummy_head
        curr = head
        k = 0
        while curr:
            k += 1
            if k == 2:
                k = 0
                tmp = curr.next
                curr.next = prev.next
                prev.next.next = tmp
                prev.next = curr
                prev = curr.next
                curr = tmp
            else:
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



### 34-在排序数组中查找元素的第一个和最后一个位置

给你一个按照非递减顺序排列的整数数组 `nums`，和一个目标值 `target`。请你找出给定目标值在数组中的开始位置和结束位置。

如果数组中不存在目标值 `target`，返回 `[-1, -1]`。

你必须设计并实现时间复杂度为 `O(log n)` 的算法解决此问题。

**思路**

这也是二分法的一个体现，注意由于`mid = (left+right)//2`的设定，因此当临界情况时，总是会偏向左边那个。因此判断情况大于等于就是指向**第一个大于或等于目标元素**的位置，而大于则会指向**第一个大于目标元素**的位置，将后者减一即是最后一个位置。

最后需要先判断first是不是已经位于末端了，这样的话就说明没有找到。

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        def search_first(nums, target):
            left = 0
            right = len(nums)
            while left < right:
                mid = (left+right)//2
                if nums[mid] >= target:
                    right = mid
                else:
                    left = mid+1
            return left
        
        def search_last(nums, target):
            left = 0
            right = len(nums)
            while left < right:
                mid = (left+right)//2
                if nums[mid] > target:
                    right = mid
                else:
                    left = mid+1
            return left-1
        
        left = search_first(nums, target)
        right = search_last(nums, target)
        if left == len(nums) or nums[left] != target:
            return [-1,-1]
        else:
            return [left, right]

```

---



### 35-搜索插入位置

给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。

请必须使用时间复杂度为 `O(log n)` 的算法。

**思路**

二分搜索，这里设置了左闭右开的区间。

关于`nums[mid] >= target`此处应该使用大于还是大于等于，应该由我们的mid是怎么计算的来决定。不妨考虑left和right相差一个位置的情况（即循环结束的前一步），此时mid一定是落在left的位置（`mid = (left+right)//2`）。若target本身就在left位置，此时只有`nums[mid] >= target`，可以将区间收敛到left的位置，否则便会收敛到比target大的那一个位置。

```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        left = 0
        right = len(nums)
        while left < right:
            mid = (left+right)//2
            if nums[mid] >= target:
                right = mid
            else:
                left = mid+1
        return left
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



### 51-N皇后

按照国际象棋的规则，皇后可以攻击与之处在同一行或同一列或同一斜线上的棋子。

**n 皇后问题** 研究的是如何将 `n` 个皇后放置在 `n×n` 的棋盘上，并且使皇后彼此之间不能相互攻击。

给你一个整数 `n` ，返回所有不同的 **n 皇后问题** 的解决方案。

每一种解法包含一个不同的 **n 皇后问题** 的棋子放置方案，该方案中 `'Q'` 和 `'.'` 分别代表了皇后和空位。

**思路**

可以使用回溯法，尝试每一种解。我们对行的元素进行遍历

每次考虑在这一行的每一列，我们能不能放下这一棋子。如果可以放下就往下移动一行，直到结束。

对于能不能放下的判断，只需要维持一个列，左斜和右斜的布尔数组即可。

```python
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        col = [False]*n 
        left_diag = [False]*(2*n+1)
        right_diag = [False]*(2*n+1)
        board = [["."]*n for _ in range(n)]
        ans = []

        def backtracking(row):
            if row == n:
                tmp = ["".join(each_row) for each_row in board]
                ans.append(tmp)
                return

            for i in range(n):
                if not col[i] and not left_diag[n-1-i+row] and not right_diag[i+row]:
                    board[row][i] = "Q"
                    col[i] = left_diag[n-1-i+row] = right_diag[i+row] = True
                    backtracking(row+1)
                    col[i] = left_diag[n-1-i+row] = right_diag[i+row] = False
                    board[row][i] = "."
        backtracking(0)
        return ans


```

---



### 52-N皇后II

**n 皇后问题** 研究的是如何将 `n` 个皇后放置在 `n × n` 的棋盘上，并且使皇后彼此之间不能相互攻击。

给你一个整数 `n` ，返回 **n 皇后问题** 不同的解决方案的数量。

**思路**

思路和51题十分相同，只是换了一个返回值。

```python
class Solution:
    def totalNQueens(self, n: int) -> int:
        col = [False]*n 
        left_diag = [False]*(2*n+1)
        right_diag = [False]*(2*n+1)
        ans = 0

        def backtracking(row):
            nonlocal ans
            if row == n:
                ans += 1
                return
            for i in range(n):
                if not col[i] and not left_diag[n-1-i+row] and not right_diag[i+row]:
                    col[i] = left_diag[n-1-i+row] = right_diag[i+row] = True
                    backtracking(row+1)
                    col[i] = left_diag[n-1-i+row] = right_diag[i+row] = False
        backtracking(0)
        return ans

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



### 55-跳跃游戏

给你一个非负整数数组 `nums` ，你最初位于数组的 **第一个下标** 。数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个下标，如果可以，返回 `true` ；否则，返回 `false` 。

**思路**

使用动态规划，检查之前的格子能不能到达，如果可以的话再考虑能不能从那个格子出发。

为了防止超时，循环的时候可以从靠近目前i的地方开始逆向遍历。

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        if len(nums) == 1:
            return True
        dp = [False]*len(nums)
        dp[0] = True
        for i in range(1, len(nums)):
            for k in range(1, i+1):
                if dp[i-k] and nums[i-k] >= k:
                    dp[i] = True
                    break
        return dp[len(nums)-1]
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



### 62-不同路径

一个机器人位于一个 `m x n` 网格的左上角 （起始点在下图中标记为 “Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。

问总共有多少条不同的路径？

 **思路**

使用二维dp，递推公式为`dp[i][j] = dp[i-1][j] + dp[i][j-1]`

可以额外多设置一行和一列，这样方便遍历，不用考虑特殊情况。

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[0]*(n+1) for _ in range(m+1)]
        dp[1][1] = 1
        for i in range(1, m+1):
            for j in range(1, n+1):
                if i == 1 and j == 1:
                    continue
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        
        return dp[m][n]
```

---



### 64-最小路劲和

给定一个包含非负整数的 `*m* x *n*` 网格 `grid` ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。

**说明：**每次只能向下或者向右移动一步。

**思路**

二维dp，推导公式为`dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]`

```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        n = len(grid)
        m = len(grid[0])
        dp = [[0]*m for _ in range(n)]
        dp[0][0] = grid[0][0]
        for i in range(1,n):
            dp[i][0] = dp[i-1][0] + grid[i][0]
        for j in range(1,m):
            dp[0][j] = dp[0][j-1] + grid[0][j]
        for i in range(1,n):
            for j in range(1,m):
                dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
        return dp[n-1][m-1]
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



### 74-搜索二维矩阵

给你一个满足下述两条属性的 `m x n` 整数矩阵：

- 每行中的整数从左到右按非严格递增顺序排列。
- 每行的第一个整数大于前一行的最后一个整数。

给你一个整数 `target` ，如果 `target` 在矩阵中，返回 `true` ；否则，返回 `false` 。

 **思路1**

进行两次二分法搜索

这个时候要注意，第一次搜索的时候，直接对每一行最后一个元素进行二分搜索，因为我们需要的情况是最后停止搜索的位置总是小于等于target。第二次搜索的时候，也是一样的，因此两次都是`mid >= target`来移动右指针。注意每次搜索完需要检查指针有没有越界，如果越界了就说明没有符合的（太大了），不用判断直接返回False。

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        n = len(matrix)
        m = len(matrix[0])
        left = 0
        right = n
        while left < right:
            mid = (left+right)//2
            if matrix[mid][-1] >= target:
                right = mid
            else:
                left = mid+1
        row = left
        if row == n:
            return False
        left = 0
        right = m
        while left < right:
            mid = (left+right)//2
            if matrix[row][mid] >= target:
                right = mid
            else:
                left = mid+1
        
        col = left
        if col == m:
            return False
        return matrix[row][col] == target
```

**思路2**

也可以搜索每一行的头元素，但是要注意此时判断条件为`mid > target`，同时也需要判断边界条件。是应该用大于还是大于等于应该用临界情况来进行判断即可。

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        n = len(matrix)
        m = len(matrix[0])
        left = 0
        right = n
        while left < right:
            mid = (left+right)//2
            if matrix[mid][0] > target:
                right = mid
            else:
                left = mid+1
        row = left-1
        if row == -1:
            return False
        left = 0
        right = m
        while left < right:
            mid = (left+right)//2
            if matrix[row][mid] >= target:
                right = mid
            else:
                left = mid+1
        
        col = left
        if col == m:
            return False
        return matrix[row][col] == target
```

---



### 75-颜色分类

给定一个包含红色、白色和蓝色、共 `n` 个元素的数组 `nums` ，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。

我们使用整数 `0`、 `1` 和 `2` 分别表示红色、白色和蓝色。

必须在不使用库内置的 sort 函数的情况下解决这个问题。

**思路**

设置两个指针，一左一右分别代表应该插入0和插入2的位置，接着使用curr指针遍历整个数组。

如果当前位置为0，那么就和左指针互换，并且左指针和curr都要向前移动，这是因为左指针现在的值只可能是1或者0，而不会是2，因为2已经被换到后面去了。

如果当前位置为2，与右指针互换，这是只需要移动右指针，不需要移动左边的，因为在curr位置换过来的值仍有可能是2。

否则当前位置是1，那么移动curr向右。注意循环的边界条件是curr<=right。

```python
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        zero = 0
        two = len(nums)-1
        curr = 0
        while curr <= two:
            if nums[curr] == 0:
                nums[curr], nums[zero] = nums[zero], nums[curr]
                curr += 1
                zero += 1
            elif nums[curr] == 2:
                nums[curr], nums[two] = nums[two], nums[curr]
                two -= 1
            else:
                curr += 1
        
```

---



### 78-子集

给你一个整数数组 `nums` ，数组中的元素 **互不相同** 。返回该数组所有可能的

子集（幂集）。

解集 **不能** 包含重复的子集。你可以按 **任意顺序** 返回解集。

**思路**

我们需要用回溯法来解决这一类问题。首先考虑因为回答中不能有重复的元素，因此我们需要用下标来限制取数的范围，backtracking(i)则代表目前从i这个下表开始取数。每次取数的时候都需要将当前的结果放进答案中。而下一个数的取值范围就从当前取值位的下一位开始，能保证不重复。

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        ans = []
        curr = []
        def backtracking(i):
            ans.append(curr[:])
            for j in range(i, n):
                curr.append(nums[j])
                backtracking(j+1)
                curr.pop()
        backtracking(0)
        return ans
```

---



### 79-单词搜索

给定一个 `m x n` 二维字符网格 `board` 和一个字符串单词 `word` 。如果 `word` 存在于网格中，返回 `true` ；否则，返回 `false` 。

单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

**思路**

建立一个二维数组visited，来确认有没有访问过。如果当前还没有找到，并且当前的位置值并不是word的位置，就应该继续递归。

注意建立二维数组的时候，不要用`visited = [[False]*m]*n`，而是`[[False]*m for _ in range(n)]`，前者建立的是引用。

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        directions = [-1, 0, 1, 0, -1]
        found = False
        n = len(board)
        m = len(board[0])
        visited = [[False]*m for _ in range(n)]
        curr = 0

        def backtracking(i, j, curr):
            nonlocal found
            if found or board[i][j] != word[curr]:
                return
            if curr == len(word)-1:
                found = True
                return
            visited[i][j] = True
            for k in range(4):
                row = i + directions[k]
                col = j + directions[k+1]
                if row >= 0 and row < n and col >= 0 and col < m and not visited[row][col]:
                    backtracking(row, col, curr+1)

            visited[i][j] = False
        
        for i in range(n):
            for j in range(m):
                if not found:
                    backtracking(i, j, 0)
        
        return found
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



### 98-验证二叉搜索树

给你一个二叉树的根节点 `root` ，判断其是否是一个有效的二叉搜索树。

**有效** 二叉搜索树定义如下：

- 节点的左子树只包含 **小于** 当前节点的数。
- 节点的右子树只包含 **大于** 当前节点的数。
- 所有左子树和右子树自身必须也是二叉搜索树。

 **思路**

要满足二叉搜索数，即左子树中最大的（最右边）要小于根节点，右子树中最小的（最左边）要大于根节点，然后递归检查左子树和右子树。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        if not root:
            return True
        if root.left:
            curr = root.left
            while curr:
                left_most_right = curr
                curr = curr.right
            if left_most_right.val >= root.val:
                return False
        if root.right:
            curr = root.right
            while curr:
                right_most_left = curr
                curr = curr.left
            if right_most_left.val <= root.val:
                return False
        return self.isValidBST(root.left) and self.isValidBST(root.right)


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



### 106-从中序与后序遍历序列构造二叉树

给定两个整数数组 `inorder` 和 `postorder` ，其中 `inorder` 是二叉树的中序遍历， `postorder` 是同一棵树的后序遍历，请你构造并返回这颗 *二叉树* 。

**思路**

与105题类似，注意计算即可 

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        if not inorder or not postorder:
            return None
        val = postorder[-1]
        root = TreeNode(val)
        in_pos = inorder.index(val)
        right_len = len(inorder)-in_pos-1
        root.left = self.buildTree(inorder[:in_pos], postorder[:in_pos])
        root.right = self.buildTree(inorder[in_pos+1:], postorder[in_pos:-1])
        return root
```

---



### 108-将有序数组转换为二叉搜索树

给你一个整数数组 `nums` ，其中元素已经按 **升序** 排列，请你将其转换为一棵 平衡 二叉搜索树。

**平衡二叉树** 是指该树所有节点的左右子树的深度相差不超过 1。

**思路**

平衡二叉树由于是左右差不多长的，因此只需要找到中间的节点，然后将左边的一半给左子树，右边的一半给右子树，递归处理即可。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        n = len(nums)
        if n == 0:
            return None
        mid = nums[n//2]
        root = TreeNode(mid)
        root.left = self.sortedArrayToBST(nums[:n//2])
        root.right = self.sortedArrayToBST(nums[n//2+1:])
        return root
```

---



### 112-路径总和

给你二叉树的根节点 `root` 和一个表示目标和的整数 `targetSum` 。判断该树中是否存在 **根节点到叶子节点** 的路径，这条路径上所有节点值相加等于目标和 `targetSum` 。如果存在，返回 `true` ；否则，返回 `false` 。

**叶子节点** 是指没有子节点的节点。

**思路**

如果当前节点是叶子节点，并且减去当前值之后，curr为0，那么说明找到了这样的一条路径，返回True即可，否则是False。但如果不是叶子节点，那么需要进一步递归了。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        if not root:
            return False

        curr = targetSum - root.val
        if not root.left and not root.right:
            if curr == 0:
                return True 
            else:
                return False
        else:
            return self.hasPathSum(root.left, curr) or self.hasPathSum(root.right, curr)
```

---



### 113-路径总和II

给你二叉树的根节点 `root` 和一个整数目标和 `targetSum` ，找出所有 **从根节点到叶子节点** 路径总和等于给定目标和的路径。

**叶子节点** 是指没有子节点的节点。

**思路**

使用回溯法的时候需要注意，在进入回溯和离开回溯的时候，应该要保证前后状态是一样的，不然的话会出错，尤其是到了最后要结束的部分更要小心。可以模拟最后一步的状态来判断会不会出错。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        ans = []
        curr = []

        def dfs_helper(root, target):
            if not root:
                return
            curr.append(root.val)
            target -= root.val
            if not root.left and not root.right:
                if target == 0:
                    ans.append(curr[:])
            dfs_helper(root.left, target)
            dfs_helper(root.right, target)
            curr.pop()
        
        dfs_helper(root, targetSum)
        return ans

```

---



### 114-二叉树展开为链表

给你二叉树的根结点 `root` ，请你将它展开为一个单链表：

- 展开后的单链表应该同样使用 `TreeNode` ，其中 `right` 子指针指向链表中下一个结点，而左子指针始终为 `null` 。
- 展开后的单链表应该与二叉树先序遍历顺序相同。

**思路**

利用递归处理，先拿到处理之后的左子树和右子树，此时两个应该都是链表，将左子树置空，然后将原来的左子树放到右边，然后再拼接右子树，最后然后当前的root。依次递归下去即可得到结果。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        if not root:
            return
        left_chain = self.flatten(root.left)
        right_chain = self.flatten(root.right)
        root.left = None
        root.right = left_chain
        curr = root
        while curr.right:
            curr = curr.right
        curr.right = right_chain
        return root
```

---



### 118-杨辉三角

给定一个非负整数 *`numRows`，*生成「杨辉三角」的前 *`numRows`* 行。

**思路**

将杨辉三角左对齐，看成一个下三角的矩阵，其转移关系为

`dp[i][j] = dp[i-1][j-1] + dp[i-1][j]`

```python
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        dp = [[1]*(i+1) for i in range(numRows)]
        if numRows <= 2:
            return dp
        for i in range(2, numRows):
            for j in range(1, i):
                dp[i][j] = dp[i-1][j-1] + dp[i-1][j]
        return dp
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



### 128-最长连续序列

给定一个未排序的整数数组 `nums` ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。

请你设计并实现时间复杂度为 `O(n)` 的算法解决此问题。

**思路**

如果已知有一个 x,x+1,x+2,⋯,x+y 的连续序列，而我们却重新从 x+1，x+2 或者是 x+y 处开始尝试匹配，那么得到的结果肯定不会优于枚举 x 为起点的答案，因此我们在外层循环的时候碰到这种情况跳过即可。

那么怎么判断是否跳过呢？由于我们要枚举的数 x 一定是在数组中不存在**前驱数 x−1** 的，不然按照上面的分析我们会从 x−1 开始尝试匹配，因此我们每次在哈希表中检查是否存在 x−1 即能判断是否需要跳过了。

```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        if not nums:
            return 0
        max_length = 1
        nums = set(nums)
        for num in nums:
            if num-1 not in nums:
                start = num
                while num+1 in nums:
                    num += 1

                max_length = max(max_length, num-start+1)
        return max_length
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



### 138-随机链表的复制

给你一个长度为 `n` 的链表，每个节点包含一个额外增加的随机指针 `random` ，该指针可以指向链表中的任何节点或空节点。

构造这个链表的 深拷贝**。 深拷贝应该正好由 `n` 个 **全新** 节点组成，其中每个新节点的值都设为其对应的原节点的值。新节点的 `next` 指针和 `random` 指针也都应指向复制链表中的新节点，并使原链表和复制链表中的这些指针能够表示相同的链表状态。**复制链表中的指针都不应指向原链表中的节点** 。

例如，如果原链表中有 `X` 和 `Y` 两个节点，其中 `X.random --> Y` 。那么在复制链表中对应的两个节点 `x` 和 `y` ，同样有 `x.random --> y` 。

返回复制链表的头节点。

用一个由 `n` 个节点组成的链表来表示输入/输出中的链表。每个节点用一个 `[val, random_index]` 表示：

- `val`：一个表示 `Node.val` 的整数。
- `random_index`：随机指针指向的节点索引（范围从 `0` 到 `n-1`）；如果不指向任何节点，则为 `null` 。

你的代码 **只** 接受原链表的头节点 `head` 作为传入参数。

**思路**

该题的难点在于，由于random节点是随机的，因此当你依次拷贝的时候，有可能遇到了指向后面的random节点。因此至少需要两次的遍历才可以解决问题。

我们可以将新的节点放在原来的节点的后面，这样的话就可以快速定位新的节点的位置。第一次遍历创建这个节点，第二次的时候检查随机指针，最后一次将新的链表和原来的进行拆分。注意最后一次的结束条件和前两次不一样。

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""

class Solution:
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        if not head: return None
        curr = head
        while curr:
            tmp = curr.next
            curr.next = Node(curr.val)
            curr.next.next = tmp
            curr = curr.next.next
        
        curr = head
        while curr:
            if curr.random:
                curr.next.random = curr.random.next
            curr = curr.next.next
        
        curr = ans = head.next
        prev = head
        while curr.next:
            tmp = curr.next
            curr.next = tmp.next
            prev.next = tmp
            curr = curr.next
            prev = prev.next

        return ans

        
```

---



### 139-单词拆分

给你一个字符串 `s` 和一个字符串列表 `wordDict` 作为字典。如果可以利用字典中出现的一个或多个单词拼接出 `s` 则返回 `true`。

**注意：**不要求字典中出现的单词全部都使用，并且字典中的单词可以重复使用。

**思路**

使用动态规划，当前位置为True代表到当前位置是可以用dict中的单词进行拆分的。

当前位置为True的条件当且仅当`dp[i-len(word)] and s[i-len(word):i] == word`，即前k个位置也是True，且这k的位置的字符正好构成一个dict中的单词。

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        n = len(s)
        dp = [False]*(n+1)
        dp[0] = True

        for i in range(1, n+1):
            for word in wordDict:
                if i >= len(word):
                    if dp[i-len(word)] and s[i-len(word):i] == word:
                        dp[i] = True
                        break 
        return dp[n]
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



### 144-二叉树的前序遍历

给你二叉树的根节点 `root` ，返回它节点值的 **前序** 遍历。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        ans = []

        def helper(root):
            if not root:
                return
            ans.append(root.val)
            helper(root.left)
            helper(root.right)
        helper(root)
        return ans
```

---



### 145-二叉树的后序遍历

给你一棵二叉树的根节点 `root` ，返回其节点值的 **后序遍历** 。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        ans = []

        def helper(root):
            if not root:
                return
            helper(root.left)
            helper(root.right)
            ans.append(root.val)
        
        helper(root)
        return ans
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



### 152-乘积最大子数组

给你一个整数数组 `nums` ，请你找出数组中乘积最大的非空连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。

测试用例的答案是一个 **32-位** 整数。

**思路**

这题与第53题的思路很像，但是要注意负数乘负数可以变成正数，所以解这题的时候我们需要维护两个变量，当前的最大值，以及最小值，最小值可能为负数，但没准下一步乘以一个负数，当前的最大值就变成最小值，而最小值则变成最大值了。

我们的动态方程可能这样：

`maxDP[i + 1] = max(maxDP[i] * A[i + 1], A[i + 1],minDP[i] * A[i + 1])`
`minDP[i + 1] = min(minDP[i] * A[i + 1], A[i + 1],maxDP[i] * A[i + 1])`




```python
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        max_dp = nums[:]
        min_dp = nums[:]
        for i in range(1, len(nums)):
            max_dp[i] = max(nums[i], max_dp[i-1]*nums[i], min_dp[i-1]*nums[i])
            min_dp[i] = min(nums[i], max_dp[i-1]*nums[i], min_dp[i-1]*nums[i])
        return max(max_dp)
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



### 169-多数元素

给定一个大小为 `n` 的数组 `nums` ，返回其中的多数元素。多数元素是指在数组中出现次数 **大于** `⌊ n/2 ⌋` 的元素。

你可以假设数组是非空的，并且给定的数组总是存在多数元素。

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



### 173-二叉搜索树迭代器

实现一个二叉搜索树迭代器类`BSTIterator` ，表示一个按中序遍历二叉搜索树（BST）的迭代器：

- `BSTIterator(TreeNode root)` 初始化 `BSTIterator` 类的一个对象。BST 的根节点 `root` 会作为构造函数的一部分给出。指针应初始化为一个不存在于 BST 中的数字，且该数字小于 BST 中的任何元素。
- `boolean hasNext()` 如果向指针右侧遍历存在数字，则返回 `true` ；否则返回 `false` 。
- `int next()`将指针向右移动，然后返回指针处的数字。

注意，指针初始化为一个不存在于 BST 中的数字，所以对 `next()` 的首次调用将返回 BST 中的最小元素。

你可以假设 `next()` 调用总是有效的，也就是说，当调用 `next()` 时，BST 的中序遍历中至少存在一个下一个数字。

**思路**

* 把递归转成迭代，基本想法就是用栈。
* 迭代总体思路是：栈中只保留左节点。

思路必须从递归的访问顺序说起：中序遍历的访问顺序是 `左子树 -> 根节点 -> 右子树` 的顺序，并且对 左子树 和 右子树 也进行递归。

结合下图，实际访问节点的顺序是：

* 从 根节点12 开始一路到底遍历到所有左节点，路径保存到栈中；此时栈为 [12, 6, 5]。
* 弹出栈顶节点，即 叶子节点5 ；
* 下一个栈顶元素是 该叶子节点 的 根节点6；
* 然后把 该新的根节点的右子树9 一路到底遍历其所有左节点；栈为 [12, 9, 8, 7]。
* 继续运行下去，直到栈为空。

<img src="https://pic.leetcode-cn.com/1616898885-tLjlOD-173.001.jpeg" style="zoom:33%;" />

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class BSTIterator:

    def __init__(self, root: Optional[TreeNode]):
        self.stack = []
        curr = root
        while curr:
            self.stack.append(curr)
            curr = curr.left
        

    def next(self) -> int:
        node = self.stack.pop()
        curr = node.right
        while curr:
            self.stack.append(curr)
            curr = curr.left
        return node.val

    def hasNext(self) -> bool:
        return len(self.stack) > 0



# Your BSTIterator object will be instantiated and called as such:
# obj = BSTIterator(root)
# param_1 = obj.next()
# param_2 = obj.hasNext()
```

---



### 189-轮转数组

给定一个整数数组 `nums`，将数组中的元素向右轮转 `k` 个位置，其中 `k` 是非负数。

**思路**

首先对k进行取模，接着利用python数组的切片操作，进行拼接

如果直接使用 `nums = nums[-k:] + nums[0:-k]`，实际上创建了一个新的列表，并将其赋值给了局部变量 `nums`。这并不会改变原列表 `nums` 的内容，因为这个赋值操作只是改变了局部变量 `nums` 的引用，指向了一个新的列表对象。因此应该使用`nums[:]` 修改原列表的内容。

```python
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        k = k % len(nums)
        nums[:] = nums[-k:] + nums[0:-k]
```

---



### 198-打家劫舍

你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，**如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警**。

给定一个代表每个房屋存放金额的非负整数数组，计算你 **不触动警报装置的情况下** ，一夜之内能够偷窃到的最高金额。

**思路**

一个动态规划问题，抢劫到当前位置时最大的金额要么是抢劫了前一家，这一家没有抢，要么就是抢劫了上上一家再加上这一家。

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * (n+1)
        dp[1] = nums[0]
        for i in range(2, n+1):
            dp[i] = max(dp[i-1], dp[i-2]+nums[i-1])
        return dp[n]
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



### 207-课程表

你这个学期必须选修 `numCourses` 门课程，记为 `0` 到 `numCourses - 1` 。

在选修某些课程之前需要一些先修课程。 先修课程按数组 `prerequisites` 给出，其中 `prerequisites[i] = [ai, bi]` ，表示如果要学习课程 `ai` 则 **必须** 先学习课程 `bi` 。

- 例如，先修课程对 `[0, 1]` 表示：想要学习课程 `0` ，你需要先完成课程 `1` 。

请你判断是否可能完成所有课程的学习？如果可以，返回 `true` ；否则，返回 `false` 。

**思路**

这是一个图的遍历问题，由于需要计算能否满足所有的点，因此需要使用BFS算法，可以使用队列。

首先需要设置一个in_degrees列表，来记录每一个点的先修课程数，当先修课程为0的时候，即可认为可以修这一门课了。还需要一个out_graph，来记录这个课的后续课程，当这门课修好之后，所有的后续课程的先修课数量减一，若为0的时候，就可以加入列表，表示可以修这一门课了。

```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        in_degrees = [0]*numCourses
        out_graph = collections.defaultdict(list)
        for pair in prerequisites:
            in_node, out_node = pair[0], pair[1]
            out_graph[out_node].append(in_node)
            in_degrees[in_node] += 1
        queue = []
        finished = 0
        for i in range(len(in_degrees)):
            if in_degrees[i] == 0:
                queue.append(i)
                finished += 1
        while queue:
            for _ in range(len(queue)):
                course = queue.pop(0)
                for each_in in out_graph[course]:
                    in_degrees[each_in] -= 1
                    if in_degrees[each_in] == 0:
                        queue.append(each_in)
                        finished += 1
        return finished == numCourses
            
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



### 230-二叉搜索树中第k小的元素

给定一个二叉搜索树的根节点 `root` ，和一个整数 `k` ，请你设计一个算法查找其中第 `k` 小的元素（从 1 开始计数）。

**思路**

计算目前节点的左子树的全部节点数目，如果这个节点数目大于k，说明目标在左边，递归；如果是k+1，说明目标就是当前的根节点；最后，说明目标在右边的树上，递归的时候，记得减去相应的数目（左子树全部的数量+1）

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def count_nodes(self, root):
        if not root:
            return 0
        else:
            return 1+self.count_nodes(root.left)+self.count_nodes(root.right)

    
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        left_count = self.count_nodes(root.left)
        if k == left_count + 1:
            return root.val
        elif k <= left_count:
            return self.kthSmallest(root.left, k)
        else:
            return self.kthSmallest(root.right, k-(left_count+1))
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



### 238-除自身以外数组的乘积

给你一个整数数组 `nums`，返回 数组 `answer` ，其中 `answer[i]` 等于 `nums` 中除 `nums[i]` 之外其余各元素的乘积 。

题目数据 **保证** 数组 `nums`之中任意元素的全部前缀元素和后缀的乘积都在 **32 位** 整数范围内。

请 **不要使用除法，**且在 `O(n)` 时间复杂度内完成此题。

**思路**

记录每一个位置的前缀乘积和后缀乘积，两次遍历即可

```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        prev = [1]*n 
        after = [1]*n 
        for i in range(1,n):
            prev[i] = prev[i-1]*nums[i-1]
        for i in range(n-2, -1, -1):
            after[i] = after[i+1] * nums[i+1]
        ans = [x*y for x, y in zip(prev, after)]
        return ans
```

---



### 279-完全平方数

给你一个整数 `n` ，返回 *和为 `n` 的完全平方数的最少数量* 。

**完全平方数** 是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数自乘的积。例如，`1`、`4`、`9` 和 `16` 都是完全平方数，而 `3` 和 `11` 不是。

 **思路**

动态规划，注意应该从0开始，因为有些数可以直接由0到达。递推公式为

`dp[i] = min(dp[i-j*j]+1, dp[i])`

```python
class Solution:
    def numSquares(self, n: int) -> int:
        dp = [n+1]*(n+1)
        dp[0] = 0
        for i in range(1,n+1):
            j = 1
            while j*j <= i:
                dp[i] = min(dp[i-j*j]+1, dp[i])
                j += 1

        return dp[n]
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



### 347-前K个高频元素

给你一个整数数组 `nums` 和一个整数 `k` ，请你返回其中出现频率前 `k` 高的元素。你可以按 **任意顺序** 返回答案。

 **思路**

首先使用一个dict来统计每一个元素出现的频率，接着使用最小堆来维持k个最大的频率。

注意我们希望最小堆比较的每个元素的频率，而不是每个元素本身的大小！

使用`(freq, key)`这样的元组作为堆的元素是因为`heapq`库默认是最小堆，而我们希望根据频率（即元组的第一个元素）对元素进行排序。

```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        freq = collections.defaultdict(int)
        for num in nums:
            freq[num] += 1
        
        uniques = list(freq.keys())
        heap = []
        for num in uniques:
            if len(heap) < k:
                heapq.heappush(heap, (freq[num], num))
            else:
                if freq[num] > heap[0][0]:
                    heapq.heappush(heap, (freq[num], num))
                    heapq.heappop(heap)
        
        ans = [x[1] for x in heap]
        return ans
```

---



### 394-字符串解码

给定一个经过编码的字符串，返回它解码后的字符串。

编码规则为: `k[encoded_string]`，表示其中方括号内部的 `encoded_string` 正好重复 `k` 次。注意 `k` 保证为正整数。

你可以认为输入字符串总是有效的；输入字符串中没有额外的空格，且输入的方括号总是符合格式要求的。

此外，你可以认为原始数据不包含数字，所有的数字只表示重复的次数 `k` ，例如不会出现像 `3a` 或 `2[4]` 的输入。

**思路**

因为编码后的字符串有可能存在嵌套的情况，这个先入后出的情况就需要考虑栈。

依次遍历这个字符串，如果是数字，则把之后的数组全部找出来，作为乘数入栈，如果是字符或者是左括号，那么也直接入栈。当检查到是右括号的时候，将在左括号之后的字符全部取出，并乘以times，计算结果再次入栈。

遍历结束之后，栈里面只有全部的字符串了，将其拼接即可得到答案。

```python
class Solution:
    def decodeString(self, s: str) -> str:
        stack = []
        i = 0
        ans = ""
        while i < len(s):
            if s[i] >= "0" and s[i] <= "9":
                times = 0
                while s[i] >= "0" and s[i] <= "9":
                    times = times*10 + int(s[i])
                    i += 1
                stack.append(times)
            elif s[i] != "]":
                stack.append(s[i])
                i += 1
            else:
                curr = ""
                while stack[-1] != "[":
                    curr = stack.pop() + curr
                stack.pop()
                times = stack.pop()
                stack.append(curr*times)
                i += 1
        ans = "".join(stack)
        return ans


```

---



### 438-找到字符串中所有字母异位词

给定两个字符串 `s` 和 `p`，找到 `s` 中所有 `p` 的 **异位词** 的子串，返回这些子串的起始索引。不考虑答案输出的顺序。

**异位词** 指由相同字母重排列形成的字符串（包括相同的字符串）。

**思路**

可以使用滑动窗口来找符合长度的词，并使用数组来表示单词中字母出现的次数，每次滑动之后进行判断，看是否符合条件。

```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:

        def check_same(p_count, curr_count):
            for i in range(len(p_count)):
                if p_count[i] != curr_count[i]:
                    return False
            return True
        m = len(s)
        n = len(p)
        ans = []
        p_count = [0]*26
        curr_count = [0]*26
        if n > m:
            return ans
        for i in range(n):
            p_count[ord(p[i])-ord('a')] += 1

        left = right = 0
        for right in range(m):
            if right < n-1:
                curr_count[ord(s[right])-ord('a')] += 1
            else:
                curr_count[ord(s[right])-ord('a')] += 1
                if check_same(p_count, curr_count):
                    ans.append(left)
                curr_count[ord(s[left])-ord('a')] -= 1
                left += 1
                right += 1
        return ans



```

---



### 496-下一个更大元素I

`nums1` 中数字 `x` 的 **下一个更大元素** 是指 `x` 在 `nums2` 中对应位置 **右侧** 的 **第一个** 比 `x` 大的元素。

给你两个 **没有重复元素** 的数组 `nums1` 和 `nums2` ，下标从 **0** 开始计数，其中`nums1` 是 `nums2` 的子集。

对于每个 `0 <= i < nums1.length` ，找出满足 `nums1[i] == nums2[j]` 的下标 `j` ，并且在 `nums2` 确定 `nums2[j]` 的 **下一个更大元素** 。如果不存在下一个更大元素，那么本次查询的答案是 `-1` 。

返回一个长度为 `nums1.length` 的数组 `ans` 作为答案，满足 `ans[i]` 是如上所述的 **下一个更大元素** 。

**思路**

使用单调栈和哈希表即可解决。最后检查nums1中的元素在不在哈希表内，不在即可返回-1

```python
class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        stack = []
        hash_dict = {}
        for num in nums2:
            while stack and stack[-1] < num:
                prev = stack.pop(-1)
                hash_dict[prev] = num
            stack.append(num)
        
        ans = []
        for num in nums1:
            ans.append(hash_dict.get(num, -1))
        return ans

```

---



### 516-最长回文序列

给你一个字符串 `s` ，找出其中最长的回文子序列，并返回该序列的长度。

子序列定义为：不改变剩余字符顺序的情况下，删除某些字符或者不删除任何字符形成的一个序列。

**思路**

状态
`f[i][j]` 表示 s 的第 i 个字符到第 j 个字符组成的子串中，最长的回文序列长度是多少。

转移方程
如果 s 的第 i 个字符和第 j 个字符相同的话

`f[i][j] = f[i + 1][j - 1] + 2`

如果 s 的第 i 个字符和第 j 个字符不同的话

`f[i][j] = max(f[i + 1][j], f[i][j - 1])`

然后注意遍历顺序，i 从最后一个字符开始往前遍历，j 从 i + 1 开始往后遍历，这样可以保证每个子问题都已经算好了。同问题5一样，注意初始化会有些不同，两个字符的情况可能是1或者是2

```python
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        n = len(s)
        dp = [[0]*n for _ in range(n)]

        for i in range(n-1, -1, -1):
            for j in range(i, n):
                if j == i:
                    dp[i][j] = 1
                elif s[i] == s[j]:
                    dp[i][j] = dp[i+1][j-1] + 2
                else:
                    dp[i][j] = max(dp[i+1][j], dp[i][j-1])
        return dp[0][n-1]
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



### 647-回文子串

给你一个字符串 `s` ，请你统计并返回这个字符串中 **回文子串** 的数目。

**回文字符串** 是正着读和倒过来读一样的字符串。

**子字符串** 是字符串中的由连续字符组成的一个序列。

**思路**

和第5题的思路一样，只是返回的结果不一样。

使用dp去从下到上，从左到右进行遍历。

```python
class Solution:
    def countSubstrings(self, s: str) -> int:
        n = len(s)
        dp = [[False]*n for _ in range(n)]
        count = 0

        for i in range(n-1, -1, -1):
            for j in range(i, n):
                if j-i <= 1 and s[i] == s[j]:
                    dp[i][j] = True
                    count += 1
                elif dp[i+1][j-1] and s[i] == s[j]:
                    dp[i][j] = True
                    count += 1
        
        return count
```

---



### 739-每日温度

给定一个整数数组 `temperatures` ，表示每天的温度，返回一个数组 `answer` ，其中 `answer[i]` 是指对于第 `i` 天，下一个更高温度出现在几天后。如果气温在这之后都不会升高，请在该位置用 `0` 来代替。

 **思路**

根据题意，设置一个最小栈即可解决这个问题，当前温度比现在最低的还要低时，那就入栈，否则的话就出栈，直到目前已经是最小的值了。为了方便计算天数的差距，栈里面可以只需要放idx即可。

```python
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        n = len(temperatures)
        ans = [0] * n
        stack = []
        for i in range(n):
            while stack and temperatures[stack[-1]] < temperatures[i]:
                prev = stack.pop(-1)
                duration = i - prev
                ans[prev] = duration
            stack.append(i)

        return ans
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

