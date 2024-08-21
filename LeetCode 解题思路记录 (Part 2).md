# LeetCode 解题思路记录 (Part 2)



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



### 153-寻找旋转排序数组中的最小值

已知一个长度为 `n` 的数组，预先按照升序排列，经由 `1` 到 `n` 次 **旋转** 后，得到输入数组。例如，原数组 `nums = [0,1,2,4,5,6,7]` 在变化后可能得到：

- 若旋转 `4` 次，则可以得到 `[4,5,6,7,0,1,2]`
- 若旋转 `7` 次，则可以得到 `[0,1,2,4,5,6,7]`

注意，数组 `[a[0], a[1], a[2], ..., a[n-1]]` **旋转一次** 的结果为数组 `[a[n-1], a[0], a[1], a[2], ..., a[n-2]]` 。

给你一个元素值 **互不相同** 的数组 `nums` ，它原来是一个升序排列的数组，并按上述情形进行了多次旋转。请你找出并返回数组中的 **最小元素** 。

你必须设计一个时间复杂度为 `O(log n)` 的算法解决此问题。

**思路**

其实根绝题意就是找一个nums的转折点，根据中点判断，如果当前mid到right之间是连续的，那么断点一定在mid的左边，反之一定在右边，根据情况移动指针即可。注意临界情况，如果mid正好是最小的怎么办，那么可以在mid和right比较的时候加上等号，这样right=mid的时候，也不会将mid排除在搜素的范围内。

```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        left = 0
        right = len(nums)-1
        while left < right:
            mid = (left+right)//2
            if nums[mid] <= nums[right]:
                right = mid
            else:
                left = mid+1
        
        return nums[left]
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



### 208-实现Trie（前缀树）

Trie（发音类似 "try"）或者说 **前缀树** 是一种树形数据结构，用于高效地存储和检索字符串数据集中的键。这一数据结构有相当多的应用情景，例如自动补全和拼写检查。

请你实现 Trie 类：

- `Trie()` 初始化前缀树对象。
- `void insert(String word)` 向前缀树中插入字符串 `word` 。
- `boolean search(String word)` 如果字符串 `word` 在前缀树中，返回 `true`（即，在检索之前已经插入）；否则，返回 `false` 。
- `boolean startsWith(String prefix)` 如果之前已经插入的字符串 `word` 的前缀之一为 `prefix` ，返回 `true` ；否则，返回 `false` 。

**思路**

为了实现一棵前缀树，我们需要一个树节点的定义，因此我们自己设定一个类叫做TrieNode，然后TrieNode本身的子节点是26个字母的TrieNode，同时也需要is_word这个flag标准这是不是词语的终点，用于search函数使用。在判断是否是单词的时候，需要递归这个单词。

```python
class TrieNode:
    def __init__(self, is_word=False):
        self.is_word = False
        self.children = [None for _ in range(26)]

class Trie:

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        curr = self.root
        for c in word:
            pos = ord(c)-ord("a")
            if not curr.children[pos]:
                curr.children[pos] = TrieNode()
            curr = curr.children[pos]
        curr.is_word = True

    def search(self, word: str) -> bool:
        curr = self.root
        for c in word:
            pos = ord(c)-ord("a")
            if not curr.children[pos]:
                return False
            curr = curr.children[pos]
        return curr.is_word

    def startsWith(self, prefix: str) -> bool:
        curr = self.root
        for c in prefix:
            pos = ord(c)-ord("a")
            if not curr.children[pos]:
                return False
            curr = curr.children[pos]
        return True



# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)
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



### 240-搜索二维矩阵II

编写一个高效的算法来搜索 `*m* x *n*` 矩阵 `matrix` 中的一个目标值 `target` 。该矩阵具有以下特性：

- 每行的元素从左到右升序排列。
- 每列的元素从上到下升序排列。

<img src="https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/11/25/searchgrid2.jpg" style="zoom:67%;" />

**思路**

从右上角往左下方看，就可以发现是一个类似于二叉搜索树的结构，对于每一个元素，比它大的元素在它的下面，比它小的元素在它的左边。因此起点是右上方，一旦搜索过程中出界了，就说明没有找到。

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        n = len(matrix)
        m = len(matrix[0])
        curr_i = 0
        curr_j = m-1
        while curr_i >= 0 and curr_i < n and curr_j >= 0 and curr_j < m:
            if matrix[curr_i][curr_j] == target:
                return True
            elif matrix[curr_i][curr_j] > target:
                curr_j -= 1
            else:
                curr_i += 1
        return False
        
```

---



### 268-丢失的数字

给定一个包含 `[0, n]` 中 `n` 个数的数组 `nums` ，找出 `[0, n]` 这个范围内没有出现在数组中的那个数。

**思路**

将0至n和nums[0]至nums[n-1]一一异或，那么最后的答案就是只出现过一次的那个数字。

```python
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        ans = len(nums)
        for i in range(len(nums)):
            ans ^= nums[i]
            ans ^= i
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



### 287-寻找重复数

给定一个包含 `n + 1` 个整数的数组 `nums` ，其数字都在 `[1, n]` 范围内（包括 `1` 和 `n`），可知至少存在一个重复的整数。

假设 `nums` 只有 **一个重复的整数** ，返回 **这个重复的数** 。

你设计的解决方案必须 **不修改** 数组 `nums` 且只用常量级 `O(1)` 的额外空间。

**思路**

利用抽屉原理和二分搜索进行查找。因为在1至n这个范围内一定有存在重复的数字，那么不妨在1至mid之间找，看有多少数落在这个范围内，如果超过了能承受的容量（mid），说明重复的数一定在这个范围内。反之就在另一边。

```python
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        n = len(nums)
        left = 1
        right = n
        while left < right:
            count = 0
            mid = (left+right)//2
            for num in nums:
                if num <= mid:
                    count += 1
            if count > mid:
                right = mid
            else:
                left = mid+1
        return left

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



### 328-奇偶链表

给定单链表的头节点 `head` ，将所有索引为奇数的节点和索引为偶数的节点分别组合在一起，然后返回重新排序的列表。

**第一个**节点的索引被认为是 **奇数** ， **第二个**节点的索引为 **偶数** ，以此类推。

请注意，偶数组和奇数组内部的相对顺序应该与输入时保持一致。

你必须在 `O(1)` 的额外空间复杂度和 `O(n)` 的时间复杂度下解决这个问题。

 **思路**

分别设置奇偶链表的头， 对于每一个偶节点，可以将它之后的奇数节点放到前面，然后更改链表的指向。有图会更加清晰。最后将奇偶链表头尾相连即可。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def oddEvenList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return None
        odd_head = odd = head
        even_head = even = head.next

        while even and even.next:
            odd.next = even.next
            even.next = even.next.next
            odd.next.next = even

            odd = odd.next
            even = even.next
        
        odd.next = even_head
        return odd_head


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



### 416-分割等和子集

给你一个 **只包含正整数** 的 **非空** 数组 `nums` 。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。

**思路**

使用动态规划来解决这个问题，我们使用二维dp，其中`dp[i][j]`代表到下标为i的这个数位置，能不能组合成j这个目标数组，是一个0-1的背包问题，对于每一个数，我们可以选择放入或者不放入，如果放入，那么就是`dp[i-1][j-nums[i]]`，如果不放入，那么就是`dp[i][j-1]`。注意有些数可能太大了，所以不一定能放。最后只需要看二维数组的最后一个值就可以了。

```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        n = len(nums)
        if sum(nums) % 2 == 1:
            return False
        target = sum(nums) // 2

        dp = [[False]*(target+1) for _ in range(n)]
        dp[0][0] = True
        if nums[0] <= target:
            dp[0][nums[0]] = True
        
        for i in range(1, n):
            for j in range(target+1):
                if nums[i] <= j:
                    dp[i][j] = dp[i-1][j-nums[i]] or dp[i-1][j]
                else:
                    dp[i][j] = dp[i-1][j]
        
        return dp[n-1][target]
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

### 437-路径总和III

给定一个二叉树的根节点 `root` ，和一个整数 `targetSum` ，求该二叉树里节点值之和等于 `targetSum` 的 **路径** 的数目。

**路径** 不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。

**思路**

对于每一个节点，考虑两种情况，以这个节点为根节点，和以这个节点的左子树和右子树分为为根的情况。因此把需要计算根节点为起点的代码总结出来，然后递归左右子树。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rootSum(self, root, targetSum):
        if not root:
            return 0
        count = 0
        if root.val == targetSum:
            count += 1
        count += self.rootSum(root.left, targetSum-root.val)
        count += self.rootSum(root.right, targetSum-root.val)
        return count


    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        if not root:
            return 0
        return self.rootSum(root, targetSum) + self.pathSum(root.left, targetSum) + self.pathSum(root.right, targetSum)
        
        
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



### 538-将二叉搜索树转换为累加树

给出二叉 **搜索** 树的根节点，该树的节点值各不相同，请你将其转换为累加树（Greater Sum Tree），使每个节点 `node` 的新值等于原树中大于或等于 `node.val` 的值之和。

提醒一下，二叉搜索树满足下列约束条件：

- 节点的左子树仅包含键 **小于** 节点键的节点。
- 节点的右子树仅包含键 **大于** 节点键的节点。
- 左右子树也必须是二叉搜索树。

**思路**

逆向遍历二叉搜素树，并记录前序节点，当前节点的值就需要加上前序节点，因为前序节点一定比当前节点的值要大。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def convertBST(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return None
        
        self.prev = None

        def helper(node):
            if not node:
                return

            helper(node.right)

            if self.prev:
                node.val += self.prev.val
            self.prev = node

            helper(node.left)

        helper(root)

        return root

                
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



### 763-划分字母区间

给你一个字符串 `s` 。我们要把这个字符串划分为尽可能多的片段，同一字母最多出现在一个片段中。

注意，划分结果需要满足：将所有划分结果按顺序连接，得到的字符串仍然是 `s` 。

返回一个表示每个字符串片段的长度的列表。

**思路**

此题的思路与合并区间的思路很像，因此我们需要记录每一个字母出现的最后一个位置。接着再次遍历整个字符串，记录当前可以合并的最大范围，如果在此范围内有可以达到更远距离的字母出现了，那么更新这个范围即可。如果到达了该范围，那么就可以将当前片段的长度输入了，为了方便我们还需要记录之前那个片段的结束位置，相减即可。

```python
class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        last = collections.defaultdict(int)
        for i in range(len(s)):
            last[s[i]] = i 
        
        prev_last = -1
        curr_last = last[s[0]]
        ans = []
        for i in range(len(s)):
            curr_last = max(curr_last, last[s[i]])
            if i == curr_last:
                ans.append(curr_last-prev_last)
                prev_last = curr_last
            
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



### 1008-前序遍历构造二叉搜索树

给定一个整数数组，它表示BST(即 **二叉搜索树** )的 先序遍历 ，构造树并返回其根。

**保证** 对于给定的测试用例，总是有可能找到具有给定需求的二叉搜索树。

**思路**

对于前序遍历，第一个数就是他的根节点，接着对于BST来说，左子树的值全都小于右子树，因此只需要找到第一个大于根节点值的位置就是右子树开始的地方。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def bstFromPreorder(self, preorder: List[int]) -> Optional[TreeNode]:

        if not preorder:
            return
        n = len(preorder)
        val = preorder[0]
        root = TreeNode(val)
        start = 1
        while start < n:
            if preorder[start] < val:
                start += 1
            else:
                break
        root.left = self.bstFromPreorder(preorder[1:start])
        root.right = self.bstFromPreorder(preorder[start:])
        return root
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

