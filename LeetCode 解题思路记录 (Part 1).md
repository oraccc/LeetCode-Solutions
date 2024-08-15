# LeetCode 解题思路记录 (Part 1)



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



### 4-寻找两个正序数组的中位数

给定两个大小分别为 `m` 和 `n` 的正序（从小到大）数组 `nums1` 和 `nums2`。请你找出并返回这两个正序数组的 **中位数** 。

算法的时间复杂度应该为 `O(log (m+n))` 。

**思路**

我们把数组 A 和数组 B 分别在 i 和 j 进行切割。

将 i 的左边和 j 的左边组合成「左半部分」，将 i 的右边和 j 的右边组合成「右半部分」。

为了保证 max ( A [ i - 1 ] , B [ j - 1 ]）） <= min ( A [ i ] , B [ j ]）），因为 A 数组和 B 数组是有序的，所以 A [ i - 1 ] <= A [ i ]，B [ i - 1 ] <= B [ i ] 这是天然的，所以我们只需要保证 B [ j - 1 ] < = A [ i ] 和 A [ i - 1 ] <= B [ j ] 所以我们分两种情况讨论：

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



### 27-移除元素

给你一个数组 `nums` 和一个值 `val`，你需要 **[原地](https://baike.baidu.com/item/原地算法)** 移除所有数值等于 `val` 的元素。元素的顺序可能发生改变。然后返回 `nums` 中与 `val` 不同的元素的数量。

假设 `nums` 中不等于 `val` 的元素数量为 `k`，要通过此题，您需要执行以下操作：

- 更改 `nums` 数组，使 `nums` 的前 `k` 个元素包含不等于 `val` 的元素。`nums` 的其余元素和 `nums` 的大小并不重要。
- 返回 `k`。

**思路**

设置一个左右指针，一个是头一个是尾，我们需要确保从0开始到left之间（不包含left）都不是val，最后返回left即可。如果left目前的值是val，那么就和尾部的值交换，同时尾部往前移一位，但是注意此时left不可以动，很有可能left交换过来的值还是val。如果left此前不是val，向后移动left。直到两者交叉。

注意结束条件是left>right的时候，此时才能返回。

```python
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        left = 0
        right = len(nums)-1
        while left <= right:
            if nums[left] != val:
                left += 1
            else:
                nums[left], nums[right] = nums[right], nums[left]
                right -= 1
        
        return left
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



### 45-跳跃游戏II

给定一个长度为 `n` 的 **0 索引**整数数组 `nums`。初始位置为 `nums[0]`。

每个元素 `nums[i]` 表示从索引 `i` 向前跳转的最大长度。换句话说，如果你在 `nums[i]` 处，你可以跳转到任意 `nums[i + j]` 处:

- `0 <= j <= nums[i]` 
- `i + j < n`

返回到达 `nums[n - 1]` 的最小跳跃次数。生成的测试用例可以到达 `nums[n - 1]`。

**思路1**

使用动态规划即可，整体流程和55题类似，要注意有些点可能到不了，所以要保存一个n+1，代表没有办法到达的点。

```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [n+1] * n
        dp[0] = 0
        for i in range(1, n):
            for j in range(i):
                if dp[j] != n+1 and i-j <= nums[j]:
                    dp[i] = min(dp[i], dp[j]+1)

        return dp[n-1]
```

**思路2**

可以使用担心算法，记录一个当前可以到达的最远距离，如果当前最远的距离没有超过n，那么就循环，加上一次步数。每一个走的时候，就记录从现在位置开始走，能走到的最远距离是多少。下次直接从最远的距离开始走，就可以找到最小的步数。

```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        count = 0
        max_distance = 0
        start = 0
        
        while max_distance < len(nums)-1:
            count += 1
            curr_max = max_distance
            while start <= curr_max:
                max_distance = max(max_distance, start+nums[start])
                start += 1
        
        return count
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



### 110-平衡二叉树

给定一个二叉树，判断它是否是 平衡二叉树

 **思路**

如果从上到下遍历的话，每个节点均会遍历多次，因此从下到上遍历。

对于当前节点，先计算它左子树和右子树的深度，如果此时已经不平衡了，那么就返回-1，代表着子树已经不平衡了。或者如果此时左子树或者右子树的深度变成了-1，那么也是一个不平衡的子树。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:

        def helper(node):
            if not node:
                return 0
            left_height = helper(node.left)
            right_height = helper(node.right)

            if abs(left_height-right_height) > 1 or left_height == -1 or right_height == -1:
                return -1
            else:
                return 1+max(left_height, right_height)
        
        return helper(root) != -1
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



### 125-验证回文串

如果在将所有大写字符转换为小写字符、并移除所有非字母数字字符之后，短语正着读和反着读都一样。则可以认为该短语是一个 **回文串** 。

字母和数字都属于字母数字字符。

给你一个字符串 `s`，如果它是 **回文串** ，返回 `true` ；否则，返回 `false` 。

**思路**

使用双指针，注意跳过非字母数字的字符以及大小写转换

```python
class Solution:
    def isPalindrome(self, s: str) -> bool:
        left = 0
        right = len(s)-1
        while left < right:
            if not s[left].isalnum():
                left += 1
            elif not s[right].isalnum():
                right -= 1
            elif s[left].lower() != s[right].lower():
                return False
            else:
                left += 1
                right -= 1
        return True
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



### 131-分割回文串

给你一个字符串 `s`，请你将 `s` 分割成一些子串，使每个子串都是 **回文串**。返回 `s` 所有可能的分割方案。

**思路**

使用回溯方法解决，每次dfs的时候，检查当前从start开始能否有子串满足要求，即`for end in range(start+1, n+1)`，如何有的话，就继续dfs。

同时也需要建立一个函数来检查是否是有效的回文字符串。

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        n = len(s)

        ans = []
        curr = []

        def backtracking(start):
            if start == n:
                ans.append(curr[:])
                return
            for end in range(start+1, n+1):
                substr = s[start:end]
                if is_valid(substr):
                    curr.append(substr)
                    backtracking(end)
                    curr.pop()
        
        def is_valid(substr):
            left = 0
            right = len(substr)-1
            while left < right:
                if substr[left] != substr[right]:
                    return False
                else:
                    left += 1
                    right -= 1
            return True

        backtracking(0)

        return ans
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

