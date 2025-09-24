二路归并法：





















我来为你详细讲解Python中dict的常用方法，以及在算法面试中常用的数据结构。

## Python dict() 常用方法

### 基础操作方法

```python
# 创建字典
d = {'a': 1, 'b': 2, 'c': 3}

# 1. get() - 安全获取值，避免KeyError
print(d.get('a'))      # 1
print(d.get('d', 0))   # 0 (默认值)

# 2. setdefault() - 如果key不存在则设置默认值
d.setdefault('d', 4)   # {'a': 1, 'b': 2, 'c': 3, 'd': 4}

# 3. update() - 更新字典
d.update({'e': 5, 'f': 6})

# 4. pop() - 删除并返回值
val = d.pop('a', None)  # 返回1，删除key 'a'

# 5. keys(), values(), items() - 获取视图
for k, v in d.items():
    print(f"{k}: {v}")
```

### 在算法题中的实际应用

```python
# LeetCode 1. Two Sum - 使用字典存储值和索引的映射
def twoSum(nums, target):
    num_map = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_map:
            return [num_map[complement], i]
        num_map[num] = i
    return []

# LeetCode 242. Valid Anagram - 使用字典统计字符频率
def isAnagram(s, t):
    if len(s) != len(t):
        return False
    
    char_count = {}
    # 统计s中字符频率
    for char in s:
        char_count[char] = char_count.get(char, 0) + 1
    
    # 检查t中字符频率
    for char in t:
        if char not in char_count:
            return False
        char_count[char] -= 1
        if char_count[char] == 0:
            del char_count[char]
    
    return len(char_count) == 0
```

## 算法面试常用数据结构

### 1. collections.defaultdict

**原理**: 当访问不存在的key时，自动创建默认值

```python
from collections import defaultdict

# 分组问题 - LeetCode 49. Group Anagrams
def groupAnagrams(strs):
    groups = defaultdict(list)
    for s in strs:
        # 排序后作为key
        key = ''.join(sorted(s))
        groups[key].append(s)
    return list(groups.values())

# 图的邻接表表示
graph = defaultdict(list)
graph['A'].append('B')
graph['A'].append('C')
# 即使'D'不存在，也会自动创建空列表
graph['D'].append('E')
```

### 2. collections.Counter

**原理**: 继承自dict，专门用于计数

```python
from collections import Counter

# LeetCode 347. Top K Frequent Elements
def topKFrequent(nums, k):
    counter = Counter(nums)
    # 返回出现频率最高的k个元素
    return [num for num, freq in counter.most_common(k)]

# 字符串中的字符统计
s = "hello world"
char_count = Counter(s)
print(char_count)  # Counter({'l': 3, 'o': 2, 'h': 1, 'e': 1, ' ': 1, 'w': 1, 'r': 1, 'd': 1})

# 找出现次数为1的字符
unique_chars = [char for char, count in char_count.items() if count == 1]
```

### 3. collections.deque (双端队列)

**原理**: 基于双向链表实现，两端操作都是O(1)

```python
from collections import deque

# BFS遍历 - LeetCode 102. Binary Tree Level Order Traversal
def levelOrder(root):
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        level_nodes = []
        
        for _ in range(level_size):
            node = queue.popleft()  # O(1) 左端弹出
            level_nodes.append(node.val)
            
            if node.left:
                queue.append(node.left)   # O(1) 右端添加
            if node.right:
                queue.append(node.right)
        
        result.append(level_nodes)
    
    return result

# 滑动窗口最大值 - 单调队列
def maxSlidingWindow(nums, k):
    dq = deque()  # 存储索引，保持单调递减
    result = []
    
    for i, num in enumerate(nums):
        # 移除窗口外的元素
        while dq and dq[0] <= i - k:
            dq.popleft()
        
        # 维护单调性：移除比当前元素小的元素
        while dq and nums[dq[-1]] < num:
            dq.pop()
        
        dq.append(i)
        
        # 窗口形成后开始记录结果
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result
```

### 4. heapq (优先队列/堆)

**原理**: 基于完全二叉树实现的最小堆

```python
import heapq

# LeetCode 215. Kth Largest Element in an Array
def findKthLargest(nums, k):
    # 维护大小为k的最小堆
    heap = []
    for num in nums:
        if len(heap) < k:
            heapq.heappush(heap, num)
        elif num > heap[0]:
            heapq.heapreplace(heap, num)  # 弹出最小值，推入新值
    
    return heap[0]

# LeetCode 23. Merge k Sorted Lists
def mergeKLists(lists):
    heap = []
    # 初始化堆：(value, list_index, node)
    for i, head in enumerate(lists):
        if head:
            heapq.heappush(heap, (head.val, i, head))
    
    dummy = ListNode(0)
    current = dummy
    
    while heap:
        val, list_idx, node = heapq.heappop(heap)
        current.next = node
        current = current.next
        
        # 如果还有下一个节点，加入堆
        if node.next:
            heapq.heappush(heap, (node.next.val, list_idx, node.next))
    
    return dummy.next

# 自定义比较的堆
class Task:
    def __init__(self, priority, description):
        self.priority = priority
        self.description = description
    
    def __lt__(self, other):
        return self.priority < other.priority

task_heap = []
heapq.heappush(task_heap, Task(3, "Low priority"))
heapq.heappush(task_heap, Task(1, "High priority"))
```

### 5. SortedList (需要安装sortedcontainers)

**原理**: 使用分块技术，将排序列表分成多个小块，每个块内部有序### 6. Trie (前缀树)

**原理**: 树形结构，用于高效存储和查找字符串

```python
class TrieNode:
    def __init__(self):
        self.children = {}  # 或者用 [None] * 26 for lowercase letters
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
    
    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end
    
    def startsWith(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

# LeetCode 212. Word Search II
def findWords(board, words):
    trie = Trie()
    for word in words:
        trie.insert(word)
    
    result = set()
    
    def dfs(i, j, node, path):
        if node.is_end:
            result.add(path)
        
        if 0 <= i < len(board) and 0 <= j < len(board[0]) and board[i][j] in node.children:
            char = board[i][j]
            board[i][j] = '#'  # 标记已访问
            
            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                dfs(i + di, j + dj, node.children[char], path + char)
            
            board[i][j] = char  # 恢复
    
    for i in range(len(board)):
        for j in range(len(board[0])):
            dfs(i, j, trie.root, "")
    
    return list(result)
```

### 7. Union-Find (并查集)

**原理**: 用于处理不相交集合的合并和查询问题

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = n  # 连通分量个数
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 路径压缩
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        
        # 按秩合并
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        
        self.count -= 1
        return True
    
    def connected(self, x, y):
        return self.find(x) == self.find(y)

# LeetCode 200. Number of Islands
def numIslands(grid):
    if not grid:
        return 0
    
    m, n = len(grid), len(grid[0])
    uf = UnionFind(m * n)
    water_count = 0
    
    for i in range(m):
        for j in range(n):
            if grid[i][j] == '0':
                water_count += 1
            else:
                # 向右和向下连接
                for di, dj in [(0, 1), (1, 0)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < m and 0 <= nj < n and grid[ni][nj] == '1':
                        uf.union(i * n + j, ni * n + nj)
    
    return uf.count - water_count
```

## 总结

在算法面试中，这些数据结构的选择原则：

1. **字典操作频繁** → dict, defaultdict, Counter
2. **需要维护有序性** → SortedList, heapq
3. **队列操作** → deque
4. **字符串前缀匹配** → Trie
5. **图连通性问题** → Union-Find
6. **优先级处理** → heapq

掌握这些数据结构的时间复杂度和使用场景，能让你在面试中更好地选择合适的解法。每种数据结构都有其特定的优势，关键是要理解其内部原理和适用场景。



# SortedList的简化实现原理
class SimpleSortedList:
    def __init__(self):
        self.data = []
    
    def add(self, val):
        # 二分查找插入位置
        left, right = 0, len(self.data)
        while left < right:
            mid = (left + right) // 2
            if self.data[mid] < val:
                left = mid + 1
            else:
                right = mid
        self.data.insert(left, val)
    
    def remove(self, val):
        idx = self.bisect_left(val)
        if idx < len(self.data) and self.data[idx] == val:
            self.data.pop(idx)
            return True
        return False
    
    def bisect_left(self, val):
        left, right = 0, len(self.data)
        while left < right:
            mid = (left + right) // 2
            if self.data[mid] < val:
                left = mid + 1
            else:
                right = mid
        return left
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# 实际使用sortedcontainers.SortedList的例子
# pip install sortedcontainers

from sortedcontainers import SortedList

# LeetCode 315. Count of Smaller Numbers After Self
def countSmaller(nums):
    result = []
    sorted_list = SortedList()
    
    # 从右往左遍历
    for i in range(len(nums) - 1, -1, -1):
        # 找到第一个大于等于nums[i]的位置
        pos = sorted_list.bisect_left(nums[i])
        result.append(pos)  # pos就是比nums[i]小的元素个数
        sorted_list.add(nums[i])  # 添加当前元素
    
    return result[::-1]  # 反转结果

# LeetCode 493. Reverse Pairs
def reversePairs(nums):
    def mergeSort(start, end):
        if start >= end:
            return 0
        
        mid = (start + end) // 2
        count = mergeSort(start, mid) + mergeSort(mid + 1, end)
        
        # 计算逆序对
        j = mid + 1
        for i in range(start, mid + 1):
            while j <= end and nums[i] > 2 * nums[j]:
                j += 1
            count += j - (mid + 1)
        
        # 归并
        nums[start:end+1] = sorted(nums[start:end+1])
        return count
    
    return mergeSort(0, len(nums) - 1)

# 使用SortedList的优雅解法
def reversePairs_sorted(nums):
    sorted_list = SortedList()
    count = 0
    
    for num in reversed(nums):
        # 找到满足 sorted_num < num/2 的元素个数
        pos = sorted_list.bisect_left(num / 2.0)
        count += pos
        sorted_list.add(num)
    
    return count

# LeetCode 327. Count of Range Sum
def countRangeSum(nums, lower, upper):
    prefix_sums = [0]
    for num in nums:
        prefix_sums.append(prefix_sums[-1] + num)
    
    sorted_list = SortedList()
    count = 0
    
    for prefix_sum in prefix_sums:
        # 查找满足 lower <= prefix_sum - prev_sum <= upper 的prev_sum
        # 即 prefix_sum - upper <= prev_sum <= prefix_sum - lower
        left_bound = prefix_sum - upper
        right_bound = prefix_sum - lower
        
        left_idx = sorted_list.bisect_left(left_bound)
        right_idx = sorted_list.bisect_right(right_bound)
        
        count += right_idx - left_idx
        sorted_list.add(prefix_sum)
    
    return count

# 滑动窗口中位数 - LeetCode 480
def medianSlidingWindow(nums, k):
    sorted_list = SortedList()
    result = []
    
    for i, num in enumerate(nums):
        sorted_list.add(num)
        
        if i >= k:
            sorted_list.remove(nums[i - k])
        
        if i >= k - 1:
            if k % 2 == 1:
                median = float(sorted_list[k // 2])
            else:
                median = (sorted_list[k // 2 - 1] + sorted_list[k // 2]) / 2.0
            result.append(median)
    
    return result

# 区间调度问题 - 会议室II变种
def maxConcurrentMeetings(intervals):
    """返回任意时刻最多有多少个会议在进行"""
    events = []
    for start, end in intervals:
        events.append((start, 1))   # 会议开始
        events.append((end, -1))    # 会议结束
    
    sorted_events = SortedList(events)
    max_concurrent = 0
    current_concurrent = 0
    
    for time, delta in sorted_events:
        current_concurrent += delta
        max_concurrent = max(max_concurrent, current_concurrent)
    
    return max_concurrent

# 测试示例
if __name__ == "__main__":
    # 测试SimpleSortedList
    simple_sl = SimpleSortedList()
    for val in [3, 1, 4, 1, 5, 9, 2, 6]:
        simple_sl.add(val)
    print("SimpleSortedList:", [simple_sl[i] for i in range(len(simple_sl))])
    
    # 测试countSmaller
    print("Count Smaller:", countSmaller([5, 2, 6, 1]))  # [2, 1, 1, 0]
    
    # 测试区间调度
    meetings = [(1, 3), (2, 4), (3, 6), (5, 7)]
    print("Max concurrent meetings:", maxConcurrentMeetings(meetings))  # 2









# Python数据结构详解教程 - 从零基础到算法面试

## 一、Python dict（字典）详解

### 1.1 原理

字典是基于**哈希表（Hash Table）**实现的数据结构：

- 内部使用哈希函数将key转换为数组索引
- 通过key可以直接定位到value的存储位置
- 平均时间复杂度：查找O(1)，插入O(1)，删除O(1)

### 1.2 作用

- 建立key-value映射关系
- 快速查找、插入、删除数据
- 统计频率、去重、缓存等场景

### 1.3 基础用法

```python
# 创建字典
d = {'name': '张三', 'age': 25, 'city': '北京'}
d2 = dict(name='李四', age=30)
d3 = dict([('a', 1), ('b', 2)])

# 基本操作
print(d['name'])        # 张三
d['job'] = '程序员'      # 添加
d['age'] = 26          # 修改
del d['city']          # 删除
```

### 1.4 常用方法详解

```python
d = {'a': 1, 'b': 2, 'c': 3}

# 1. get() - 安全获取，避免KeyError
value = d.get('d')          # None（key不存在返回None）
value = d.get('d', 0)       # 0（指定默认值）
value = d.get('a')          # 1（key存在返回对应值）

# 2. setdefault() - 如果key不存在就设置默认值
d.setdefault('d', 4)        # 如果'd'不存在，设置为4
print(d)                    # {'a': 1, 'b': 2, 'c': 3, 'd': 4}

# 3. update() - 批量更新
d.update({'e': 5, 'f': 6})  # 添加多个键值对
d.update([('g', 7), ('h', 8)])  # 也可以用列表

# 4. pop() - 删除并返回值
value = d.pop('a')          # 删除'a'并返回值1
value = d.pop('z', 0)       # key不存在返回默认值0

# 5. keys(), values(), items()
keys = list(d.keys())       # 获取所有键
values = list(d.values())   # 获取所有值
items = list(d.items())     # 获取所有键值对

# 6. 遍历字典
for key in d:
    print(key, d[key])

for key, value in d.items():
    print(key, value)
```

------

## 二、collections.defaultdict

### 2.1 原理

defaultdict是dict的子类，重写了`__missing__`方法：

- 当访问不存在的key时，不会抛出KeyError
- 而是调用default_factory函数创建默认值
- 本质上是给字典添加了"自动初始化"功能

### 2.2 作用

- 避免KeyError异常
- 简化分组、统计等操作的代码
- 自动创建嵌套结构

### 2.3 用法详解

```python
from collections import defaultdict

# 基础用法
dd = defaultdict(int)       # 默认值为0
dd['a'] += 1               # 等价于 dd['a'] = dd.get('a', 0) + 1
print(dd['a'])             # 1
print(dd['b'])             # 0（自动创建）

dd2 = defaultdict(list)    # 默认值为空列表
dd2['fruits'].append('apple')
print(dd2['fruits'])       # ['apple']

dd3 = defaultdict(set)     # 默认值为空集合
dd3['tags'].add('python')

# 实际应用：分组
students = [
    ('张三', '数学'),
    ('李四', '英语'),
    ('张三', '物理'),
    ('王五', '数学')
]

groups = defaultdict(list)
for name, subject in students:
    groups[subject].append(name)

print(dict(groups))
# {'数学': ['张三', '王五'], '英语': ['李四'], '物理': ['张三']}

# 嵌套defaultdict
nested = defaultdict(lambda: defaultdict(int))
nested['A']['x'] = 1
nested['A']['y'] += 1
print(nested['B']['z'])    # 0（自动创建两层）
```

------

## 三、collections.Counter

### 3.1 原理

Counter是dict的子类，专门用于计数：

- 内部还是哈希表，但优化了计数操作
- 提供了most_common、subtract等专门的计数方法
- 可以进行数学运算（加减交并等）

### 3.2 作用

- 统计元素出现频率
- 找出现最多/最少的元素
- 计数器之间的数学运算

### 3.3 用法详解

```python
from collections import Counter

# 基础创建
c1 = Counter('hello world')
print(c1)  # Counter({'l': 3, 'o': 2, 'h': 1, 'e': 1, ' ': 1, 'w': 1, 'r': 1, 'd': 1})

c2 = Counter([1, 2, 3, 1, 2, 1])
print(c2)  # Counter({1: 3, 2: 2, 3: 1})

c3 = Counter({'red': 4, 'blue': 2})

# 常用方法
print(c1.most_common(2))    # [('l', 3), ('o', 2)] 最常见的2个
print(c1.most_common())     # 所有元素按频率排序

# 计数器运算
c4 = Counter([1, 2, 3])
c5 = Counter([1, 1, 2])
print(c4 + c5)             # Counter({1: 3, 2: 2, 3: 1}) 相加
print(c4 - c5)             # Counter({3: 1}) 相减（只保留正数）
print(c4 & c5)             # Counter({1: 1, 2: 1}) 交集（取最小）
print(c4 | c5)             # Counter({1: 2, 2: 1, 3: 1}) 并集（取最大）

# 实际应用：找频率最高的元素
text = "the quick brown fox jumps over the lazy dog"
words = text.split()
word_count = Counter(words)
print(word_count.most_common(3))  # [('the', 2), ('quick', 1), ('brown', 1)]

# 更新计数
c = Counter()
c.update('hello')
c.update(['h', 'e', 'l'])
print(c)  # Counter({'l': 3, 'h': 2, 'e': 2, 'o': 1})
```

------

## 四、collections.deque（双端队列）

### 4.1 原理

deque是"double-ended queue"的缩写：

- 内部使用双向链表实现
- 两端的添加和删除操作都是O(1)时间复杂度
- 中间位置的访问是O(n)（不像list的O(1)随机访问）

### 4.2 作用

- 需要频繁在两端操作时比list更高效
- 实现队列（FIFO）和栈（LIFO）
- 滑动窗口算法
- BFS广度优先搜索

### 4.3 用法详解

```python
from collections import deque

# 基础创建和操作
dq = deque([1, 2, 3])
print(dq)                   # deque([1, 2, 3])

# 右端操作
dq.append(4)               # 右端添加
dq.appendleft(0)           # 左端添加
print(dq)                  # deque([0, 1, 2, 3, 4])

right_item = dq.pop()      # 右端弹出，返回4
left_item = dq.popleft()   # 左端弹出，返回0
print(dq)                  # deque([1, 2, 3])

# 其他操作
dq.extend([4, 5])          # 右端批量添加
dq.extendleft([0, -1])     # 左端批量添加（注意顺序）
print(dq)                  # deque([-1, 0, 1, 2, 3, 4, 5])

dq.rotate(2)               # 向右旋转2位
print(dq)                  # deque([4, 5, -1, 0, 1, 2, 3])

dq.rotate(-2)              # 向左旋转2位（回到原来状态）

# 队列实现（FIFO - 先进先出）
queue = deque()
queue.append('first')      # 入队
queue.append('second')
item = queue.popleft()     # 出队，得到'first'

# 栈实现（LIFO - 后进先出）
stack = deque()
stack.append('first')      # 入栈
stack.append('second')
item = stack.pop()         # 出栈，得到'second'

# 限制大小的deque
limited_dq = deque(maxlen=3)
for i in range(5):
    limited_dq.append(i)
print(limited_dq)          # deque([2, 3, 4], maxlen=3) 自动弹出旧元素
```

------

## 五、heapq（堆/优先队列）

### 5.1 原理

堆是一种特殊的完全二叉树：

- **最小堆**：父节点总是小于等于子节点
- Python的heapq实现的是最小堆
- 用普通列表存储，索引关系：父节点i，左子节点2i+1，右子节点2i+2
- 插入和删除操作都是O(log n)

### 5.2 作用

- 快速找到最小值（或最大值）
- 实现优先队列
- Top K问题
- 合并多个有序序列

### 5.3 用法详解

```python
import heapq

# 基础操作
heap = []
heapq.heappush(heap, 3)    # 插入元素
heapq.heappush(heap, 1)
heapq.heappush(heap, 4)
print(heap)                # [1, 3, 4] 最小元素在索引0

min_item = heapq.heappop(heap)  # 弹出最小元素1
print(min_item, heap)      # 1 [3, 4]

# 从列表创建堆
nums = [3, 1, 4, 1, 5, 9, 2, 6]
heapq.heapify(nums)        # 原地转换为堆
print(nums)                # [1, 1, 2, 3, 5, 9, 4, 6]

# 获取最小的k个元素
print(heapq.nsmallest(3, nums))  # [1, 1, 2]
print(heapq.nlargest(3, nums))   # [9, 6, 5]

# heappushpop 和 heapreplace
heapq.heappushpop(heap, 2)       # 先push再pop，返回最小值
heapq.heapreplace(heap, 7)       # 先pop再push，返回之前的最小值

# 实现最大堆（取负数）
max_heap = []
for num in [3, 1, 4, 1, 5]:
    heapq.heappush(max_heap, -num)

max_value = -heapq.heappop(max_heap)  # 得到5

# 实际应用：Top K 最大元素
def find_k_largest(nums, k):
    # 维护大小为k的最小堆
    heap = []
    for num in nums:
        if len(heap) < k:
            heapq.heappush(heap, num)
        elif num > heap[0]:  # 如果当前数比堆顶大
            heapq.heapreplace(heap, num)
    return heap

result = find_k_largest([3, 1, 4, 1, 5, 9, 2, 6], 3)
print(sorted(result))  # [4, 5, 9]

# 合并多个有序列表
def merge_sorted_lists(lists):
    heap = []
    result = []
    
    # 初始化堆
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))  # (值, 列表索引, 元素索引)
    
    while heap:
        val, list_idx, elem_idx = heapq.heappop(heap)
        result.append(val)
        
        # 如果该列表还有下一个元素
        if elem_idx + 1 < len(lists[list_idx]):
            next_val = lists[list_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))
    
    return result

lists = [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
merged = merge_sorted_lists(lists)
print(merged)  # [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

------

## 六、SortedList（有序列表）

### 6.1 原理

SortedList使用**分块技术**：

- 将大的有序列表分成多个小块（每块几百个元素）
- 每个块内部保持有序
- 块之间也保持有序
- 查找：O(log n)，插入/删除：O(√n)到O(log n)

### 6.2 作用

- 维护动态有序序列
- 支持二分查找
- 区间查询和统计
- 需要频繁插入删除且保持有序的场景

### 6.3 安装和用法

```bash
# 首先需要安装
pip install sortedcontainers
from sortedcontainers import SortedList

# 基础操作
sl = SortedList([3, 1, 4, 1, 5])
print(sl)                  # SortedList([1, 1, 3, 4, 5])

sl.add(2)                  # 插入元素，自动保持有序
print(sl)                  # SortedList([1, 1, 2, 3, 4, 5])

sl.remove(1)               # 删除第一个匹配的元素
print(sl)                  # SortedList([1, 2, 3, 4, 5])

# 索引访问
print(sl[0])               # 1 (最小元素)
print(sl[-1])              # 5 (最大元素)

# 二分查找
index = sl.bisect_left(3)  # 找到第一个大于等于3的位置
print(index)               # 2

index = sl.bisect_right(3) # 找到第一个大于3的位置
print(index)               # 3

# 计数
count = sl.count(1)        # 统计1的个数
print(count)               # 1

# 实际应用1：动态维护中位数
class MedianFinder:
    def __init__(self):
        self.nums = SortedList()
    
    def addNum(self, num):
        self.nums.add(num)
    
    def findMedian(self):
        n = len(self.nums)
        if n % 2 == 1:
            return float(self.nums[n // 2])
        else:
            return (self.nums[n // 2 - 1] + self.nums[n // 2]) / 2.0

# 实际应用2：区间内元素统计
def count_in_range(sorted_list, left, right):
    """统计[left, right]区间内的元素个数"""
    left_idx = sorted_list.bisect_left(left)
    right_idx = sorted_list.bisect_right(right)
    return right_idx - left_idx

sl = SortedList([1, 3, 5, 7, 9, 11, 13])
count = count_in_range(sl, 5, 10)  # 统计[5,10]区间的元素
print(count)  # 3 (包含5, 7, 9)

# 实际应用3：逆序对计算
def count_inversions(nums):
    """计算逆序对数量"""
    sorted_list = SortedList()
    count = 0
    
    for num in nums:
        # 找到比当前数小的元素个数
        pos = sorted_list.bisect_left(num)
        count += pos
        sorted_list.add(num)
    
    return count

nums = [5, 2, 6, 1]
inversions = count_inversions(nums)
print(inversions)  # 5个逆序对：(5,2), (5,1), (2,1), (6,1), (6,2)
```

------

## 七、算法面试中的应用场景总结

### 7.1 什么时候用什么数据结构？

```python
# 1. 需要快速查找、统计频率 → dict, Counter
def most_frequent_char(s):
    from collections import Counter
    counter = Counter(s)
    return counter.most_common(1)[0]

# 2. 需要分组、避免KeyError → defaultdict
def group_words_by_length(words):
    from collections import defaultdict
    groups = defaultdict(list)
    for word in words:
        groups[len(word)].append(word)
    return dict(groups)

# 3. 需要队列操作（BFS）→ deque
def bfs_level_order(root):
    from collections import deque
    if not root:
        return []
    
    queue = deque([root])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node.val)
        
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    
    return result

# 4. 需要找最大/最小值 → heapq
def find_k_largest_elements(nums, k):
    import heapq
    return heapq.nlargest(k, nums)

# 5. 需要维护有序且频繁插入删除 → SortedList
def sliding_window_median(nums, k):
    from sortedcontainers import SortedList
    window = SortedList()
    result = []
    
    for i, num in enumerate(nums):
        window.add(num)
        if i >= k:
            window.remove(nums[i - k])
        if i >= k - 1:
            if k % 2 == 1:
                result.append(window[k // 2])
            else:
                result.append((window[k // 2 - 1] + window[k // 2]) / 2)
    
    return result
```

### 7.2 时间复杂度对比

| 操作 | dict | deque | heapq    | SortedList |
| ---- | ---- | ----- | -------- | ---------- |
| 查找 | O(1) | O(n)  | O(n)     | O(log n)   |
| 插入 | O(1) | O(1)* | O(log n) | O(log n)   |
| 删除 | O(1) | O(1)* | O(log n) | O(log n)   |
| 最值 | O(n) | O(n)  | O(1)     | O(1)       |

*deque的O(1)是指两端操作，中间操作是O(n)

通过这样的系统学习，你现在应该理解了这些数据结构的原理、作用和用法。在实际的算法面试中，选择合适的数据结构往往是解题的关键！



# **DEQUE**

> ---
>
> ## 一、用法（How to use）
>
> `deque`（发音为 "deck"）是 Python 标准库 `collections` 模块中的一个类，表示**双端队列**，支持从两端高效地添加或删除元素。
>
> ### 1. 基本导入与创建
> ```python
> from collections import deque
> 
> dq = deque()  # 创建一个空的双端队列
> dq = deque([1, 2, 3])  # 从列表创建
> ```
>
> ### 2. 常用操作
> | 操作                   | 说明                              | 示例                      |
> | ---------------------- | --------------------------------- | ------------------------- |
> | `append(x)`            | 在**右端**添加元素                | `dq.append(4)`            |
> | `appendleft(x)`        | 在**左端**添加元素                | `dq.appendleft(0)`        |
> | `pop()`                | 从**右端**移除并返回元素          | `dq.pop()`                |
> | `popleft()`            | 从**左端**移除并返回元素          | `dq.popleft()`            |
> | `extend(iterable)`     | 在**右端**批量添加                | `dq.extend([5, 6])`       |
> | `extendleft(iterable)` | 在**左端**批量添加（逆序）        | `dq.extendleft([-1, -2])` |
> | `rotate(n)`            | 向右循环移动 n 步（n 为负则左移） | `dq.rotate(1)`            |
> | `clear()`              | 清空队列                          | `dq.clear()`              |
> | `len(dq)`              | 获取长度                          | `len(dq)`                 |
> | `x in dq`              | 判断是否包含元素                  | `3 in dq`                 |
>
> ---
>
> ## 二、原理（How it works）
>
> ### 1. 底层数据结构：**块状链表（block-linked list）**
> - `deque` 不是简单的 Python 列表，也不是单向链表。
> - 它是一个由**多个固定大小的块（block）**组成的**双向链表**。
> - 每个块是一个数组（通常是 64 个元素的数组），块之间用指针连接。
> - 这种结构使得在两端插入/删除元素都非常高效，且避免了列表的频繁内存重新分配。
>
> ### 2. 内存布局示意
> ```
> [块1] <-> [块2] <-> [块3]
> ```
> 每个块内部是一个数组，支持从两端填充。
>
> ---
>
> ## 三、时间复杂度（Time Complexity）
>
> | 操作                   | 时间复杂度 | 说明                               |
> | ---------------------- | ---------- | ---------------------------------- |
> | `append(x)`            | O(1)       | 右端添加                           |
> | `appendleft(x)`        | O(1)       | 左端添加                           |
> | `pop()`                | O(1)       | 右端删除                           |
> | `popleft()`            | O(1)       | 左端删除                           |
> | `len(dq)`              | O(1)       | 获取长度                           |
> | `rotate(n)`            | O(n)       | 循环移动，最坏情况需移动所有元素   |
> | `x in dq`              | O(n)       | 线性搜索                           |
> | 随机访问（如 `dq[i]`） | O(n)       | **不推荐**，效率低，尤其是中间元素 |
>
> > ⚠️ 注意：`deque` **不支持高效的随机访问**，它的设计目标是两端操作，而不是像列表那样支持快速索引。
>
> ---
>
> ## 四、使用场景总结
>
> | 场景             | 推荐使用                                             |
> | ---------------- | ---------------------------------------------------- |
> | 实现队列（FIFO） | ✅ `append` + `popleft`                               |
> | 实现栈（LIFO）   | ✅ `append` + `pop`                                   |
> | 需要两端高效操作 | ✅ `deque`                                            |
> | 需要频繁随机访问 | ❌ 用 `list` 更合适                                   |
> | 需要线程安全     | ⚠️ `deque` 不是线程安全的，需加锁或使用 `queue.Queue` |
>
> ---
>
> ## 五、小例子：用 deque 实现队列和栈
>
> ### 队列（FIFO）
> ```python
> from collections import deque
> 
> q = deque()
> q.append('a')
> q.append('b')
> print(q.popleft())  # 'a'
> ```
>
> ### 栈（LIFO）
> ```python
> stack = deque()
> stack.append('x')
> stack.append('y')
> print(stack.pop())  # 'y'
> ```
>
> ---
>

> 
>
> ---
>
> ## ✅ 一句话先回答：
> > 每个块是一个**固定大小的数组**，里面可以存多个元素，而不是一个块只存一个元素。  
> > 这样既能利用数组的**连续内存优势**，又能通过**多个块链接**避免大数组扩容的代价。
>
> ---
>
> ## 🔍 举个例子（简化版）
>
> 假设每个块最多存 **4 个元素**：
>
> ### 初始状态
> ```python
> dq = deque()
> ```
>
> 内部是空的，没有块。
>
> ### 执行
> ```python
> dq.append(1)
> dq.append(2)
> dq.append(3)
> dq.append(4)
> ```
>
> 此时内部有一个块（数组）：
> ```
> 块0: [1, 2, 3, 4]
> ```
>
> ### 再执行
> ```python
> dq.append(5)
> ```
>
> 块0满了，就新建一个块1：
> ```
> 块0: [1, 2, 3, 4]  <->  块1: [5, None, None, None]
> ```
>
> ### 再执行
> ```python
> dq.appendleft(0)
> ```
>
> 左边也要新建块（或复用空位）：
> ```
> 块-1: [0, None, None, None]  <->  块0: [1, 2, 3, 4]  <->  块1: [5, None, None, None]
> ```
>
> ---
>
> ## ✅ 关键理解点
>
> | 概念                         | 说明                                                         |
> | ---------------------------- | ------------------------------------------------------------ |
> | **块是数组**                 | 每个块是一个定长数组（比如 64 个槽位），不是链表节点只存一个元素。 |
> | **块之间是链表**             | 块与块之间用指针连接，形成双向链表。                         |
> | **元素在块内是紧凑的**       | 比如块内可能只使用了前 3 个位置，后面是空的。                |
> | **从两端扩展时，优先用空位** | 如果最左/最右块有空位，就直接填，不用新建块。                |
>
> ---
>
> ## 🧠 类比理解
>
> 你可以把 `deque` 想象成：
>
> > **一列火车（deque）由多个车厢（块）组成，每个车厢有 64 个座位（数组），乘客（元素）坐在车厢里。  
> > 车厢之间用门（指针）连接，允许你从车头或车尾上下车。**
>
> ---
>
> ## ✅ 为什么要这样设计？
>
> | 设计目标               | 实现方式                                     |
> | ---------------------- | -------------------------------------------- |
> | 两端快速插入/删除      | 块内数组 + 块间链表                          |
> | 避免列表的频繁内存搬迁 | 不需要像 `list` 那样一次性扩容整个数组       |
> | 节省内存               | 只分配需要的块，不像链表那样每个节点都带指针 |
>
> ---
>
> ## ✅ 小结一句话：
> > 每个块是一个**小数组**，里面可以存多个元素；  
> > 多个块用链表串起来，形成 `deque` 的底层结构。  
> > 这样既保留了数组的**局部性优势**，又避免了**大数组搬迁**的代价。
>
> ---
>



# SortedList

> `sortedcontainers.SortedList`  
> （Python 社区里最常用、没有之一的有序列表实现）  
> 给你拆成三句话：  
>
> 1. 用法（怎么写代码）  
> 2. 原理（底层到底长什么样）  
> 3. 时间复杂度（每个常用操作的精确代价）  
>
> ---
>
> ## 一、用法（How to use）
>
> `SortedList` 不是标准库，先装后导：
>
> ```bash
> pip install sortedcontainers
> ```
>
> ```python
> from sortedcontainers import SortedList
> sl = SortedList()          # 空的有序列表
> sl = SortedList([3, 1, 2]) # 初始化完就是 [1, 2,3]
> ```
>
> | 操作                                    | 说明                         | 示例                     |
> | --------------------------------------- | ---------------------------- | ------------------------ |
> | `sl.add(x)`                             | 插入 **一个** 元素，保持有序 | `sl.add(5)`              |
> | `sl.update(iterable)`                   | 批量插入                     | `sl.update([7,6])`       |
> | `sl.pop(index=-1)`                      | 按 **下标** 删除并返回       | `sl.pop(0)` 最小值       |
> | `sl.remove(x)`                          | 按 **值** 删除               | `sl.remove(5)`           |
> | `sl[index]`                             | 随机访问                     | `sl[100]`                |
> | `sl.bisect_left(x)` / `bisect_right(x)` | 二分查找位置                 | 与 `bisect` 模块语义一致 |
> | `sl.count(x)`                           | 统计出现次数                 | `sl.count(3)`            |
> | `sl.irange(lo, hi)`                     | 返回迭代器：≥lo 且 ≤hi       | 区间扫描，O(k)           |
> | `sl.islice(start, stop)`                | 按 **排名** 切片             | 不拷贝，O(log n + k)     |
> | `len(sl)` / `x in sl`                   | 长度 / 成员测试              | `O(1)` / `O(log n)`      |
>
> **一句话记忆**：它就是一个 **始终排好序的 list**，但所有“写”操作自动保序，所有“读”操作都带二分加速。
>
> ---
>
> ## 二、原理（How it works）
>
> 作者 Grant Jenks 把它叫做 **“B+ 列表”**（B+List）——  
> **“一块可变的有序数组” + “一棵简单 B+ 树”** 的混血结构。
>
> ### 1. 宏观布局  
> ```
> ┌---------┐        ┌---------┐
> │  load=8 │   ...  │ load=8  │   ← 一共 M 个 **叶子数组**
> └---------┘        └---------┘
>      ↑                  ↑
>      └-------- 内部索引数组（只存“每叶最小值”指针）
> ```
>
> - **叶子层**：真正存数据，是 **定长有序 Python list**（默认 `load=1000`）。  
> - **索引层**：只存 **每片叶子的最小值 + 叶子对象指针**，本身也是一个 **有序数组**。  
> - 索引再往上可以有多层，直到一层能装下为止（高度 ≈ log₁₀₀₀ n）。
>
> ### 2. 关键实现细节  
> - **叶子可分裂 / 合并**：当某叶插入后长度 > 2×load 就分裂；删除后 < load/2 就与邻叶合并。  
> - **所有数组都用 Python 内置 list**：利用其 **C 级连续内存 + 二分** 的暴力速度。  
> - **永不拷贝大数据**：切片/区间遍历只返回迭代器，不复制元素。  
> - **缓存友好**：一次二分定位到叶子后，后续扫描都在 **连续内存** 里完成。
>
> ### 3. 类比记忆  
> 把 `SortedList` 想成  
> > **一摞排好序的“名片盒”（叶子数组），盒外再贴一张“目录卡”（索引数组）。**  
> > 找名片先翻目录卡（二分），再打开对应盒子（二分），盒内还是二分——  
> > 三层二分下来，总共 log n 级别。
>
> ---
>
> ## 三、时间复杂度（Time Complexity）
>
> | 操作                     | 复杂度          | 备注                                    |
> | ------------------------ | --------------- | --------------------------------------- |
> | `add(x)`                 | O(log n)        | 二分 + 插入到叶子数组（均摊）           |
> | `update(iterable)`       | O(k log(n + k)) | 批量先排序再合并，最坏与单次插 k 次相同 |
> | `pop(i)` / `__getitem__` | O(log n)        | 先定位叶子，再数组随机访问              |
> | `remove(x)`              | O(log n)        | 先二分找位置，再叶子内删除              |
> | `x in sl` / `count(x)`   | O(log n)        | 两次二分确定左右边界                    |
> | `bisect_left/right`      | O(log n)        | 纯二分                                  |
> | `irange(lo, hi)`         | O(log n + k)    | k 是返回元素个数                        |
> | `islice(start, stop)`    | O(log n + k)    | 同上                                    |
> | 随机索引赋值 `sl[i]=x`   | ❌ **不支持**    | 会破坏有序性，主动禁止                  |
>
> > 内存：约 **n × (元素大小 + 2 指针/元素)**，比原生 list 多一层索引开销，但常数极小。
>
> ---
>
> ## 四、一句话总结
>
> `SortedList` = **“用 B+ 树思路拼起来的有序 list”**  
> - 对外像 list 一样支持 **随机访问、切片、排序输出**；  
> - 对内靠 **多层有序数组 + 二分 + 分裂合并** 保证所有写操作  
>   在 **O(log n)** 内完成，且 **内存连续、缓存友好**。  
>
> 当你需要 **频繁插入/删除/查询的同时还要保持有序**，  
> 而标准库 `bisect` 手动维护列表太慢时，  
> **直接换 `SortedList` 就行，代码量 −90%，速度 ×10～100。**





## 深拷贝和浅拷贝



> import copy
>
> print("=== arr.copy() 始终是浅拷贝 ===")
>
> # 一维数组 - 看起来像深拷贝，实际是浅拷贝
> print("1. 一维数组：")
> arr1d = [1, 2, 3, 4, 5]
> copy1d = arr1d.copy()
>
> print(f"原数组: {arr1d}")
> print(f"拷贝数组: {copy1d}")
> print(f"是否为同一对象: {arr1d is copy1d}")  # False
>
> # 修改拷贝数组
> copy1d[0] = 999
> print(f"修改拷贝后，原数组: {arr1d}")      # [1, 2, 3, 4, 5] - 不受影响
> print(f"修改拷贝后，拷贝数组: {copy1d}")    # [999, 2, 3, 4, 5]
> print("✅ 一维数组：浅拷贝效果等同于深拷贝\n")
>
> # 二维数组 - 浅拷贝的真面目
> print("2. 二维数组：")
> arr2d = [[1, 2], [3, 4], [5, 6]]
> copy2d = arr2d.copy()  # 浅拷贝！
>
> print(f"原数组: {arr2d}")
> print(f"拷贝数组: {copy2d}")
> print(f"外层是否为同一对象: {arr2d is copy2d}")        # False
> print(f"内层是否为同一对象: {arr2d[0] is copy2d[0]}")  # True！！！
>
> # 修改内层元素
> copy2d[0][0] = 999
> print(f"修改拷贝后，原数组: {arr2d}")      # [[999, 2], [3, 4], [5, 6]] - 被影响了！
> print(f"修改拷贝后，拷贝数组: {copy2d}")    # [[999, 2], [3, 4], [5, 6]]
> print("❌ 二维数组：浅拷贝，内层共享引用\n")
>
> print("=== 深拷贝 vs 浅拷贝对比 ===")
>
> # 重置数组
> arr2d = [[1, 2], [3, 4], [5, 6]]
>
> # 浅拷贝
> shallow = arr2d.copy()
> # 深拷贝  
> deep = copy.deepcopy(arr2d)
>
> print("原数组:", arr2d)
> print("浅拷贝:", shallow)
> print("深拷贝:", deep)
>
> print("\n修改原数组的内层元素：")
> arr2d[0][0] = 888
>
> print("修改后 - 原数组:", arr2d)     # [[888, 2], [3, 4], [5, 6]]
> print("修改后 - 浅拷贝:", shallow)   # [[888, 2], [3, 4], [5, 6]] - 受影响
> print("修改后 - 深拷贝:", deep)     # [[1, 2], [3, 4], [5, 6]]   - 不受影响
>
> print("\n=== 内存引用分析 ===")
>
> # 重置
> original = [[1, 2], [3, 4]]
> shallow_copy = original.copy()
> deep_copy = copy.deepcopy(original)
>
> print("内存地址分析：")
> print(f"原数组地址:     {id(original)}")
> print(f"浅拷贝地址:     {id(shallow_copy)}")
> print(f"深拷贝地址:     {id(deep_copy)}")
>
> print(f"\n内层元素地址：")
> print(f"原数组[0]地址:   {id(original[0])}")
> print(f"浅拷贝[0]地址:   {id(shallow_copy[0])}")  # 相同！
> print(f"深拷贝[0]地址:   {id(deep_copy[0])}")    # 不同！
>
> print("\n=== 实际应用场景 ===")
>
> def leetcode_scenario():
>     print("LeetCode 刷题场景：")
>     nums = [1, 2, 3, 4]  # 一维数组
>     backup = nums.copy()  # 浅拷贝足够了
>     
>     # 修改原数组
>     nums[0] = 999
>     print(f"原数组: {nums}")      # [999, 2, 3, 4]
>     print(f"备份数组: {backup}")   # [1, 2, 3, 4] - 不受影响
>     print("✅ 一维数组用 .copy() 完全没问题")
>
> def matrix_scenario():
>     print("\n矩阵操作场景：")
>     matrix = [[1, 2], [3, 4]]
>     
>     # 错误方式：浅拷贝
>     wrong_backup = matrix.copy()
>     matrix[0][0] = 999
>     print(f"原矩阵: {matrix}")           # [[999, 2], [3, 4]]
>     print(f"错误备份: {wrong_backup}")    # [[999, 2], [3, 4]] - 被污染
>     
>     # 正确方式：深拷贝
>     matrix = [[1, 2], [3, 4]]  # 重置
>     correct_backup = copy.deepcopy(matrix)
>     matrix[0][0] = 999
>     print(f"原矩阵: {matrix}")           # [[999, 2], [3, 4]]
>     print(f"正确备份: {correct_backup}")  # [[1, 2], [3, 4]] - 不受影响
>
> leetcode_scenario()
> matrix_scenario()
>
> print("\n=== 记忆要点 ===")
> print("🔑 关键理解：")
> print("   • .copy() 永远是浅拷贝")
> print("   • 一维数组：浅拷贝 = 深拷贝效果")
> print("   • 多维数组：浅拷贝 ≠ 深拷贝效果")
> print("   • 需要真正独立副本：用 copy.deepcopy()")
>
> print("\n📝 使用建议：")
> print("   • LeetCode一维数组：用 .copy()")
> print("   • 矩阵/嵌套结构：用 copy.deepcopy()")
> print("   • 性能敏感场景：优先考虑浅拷贝")





> 我大概理解了，.copy就是浅拷贝，对于一维数组，他把地址拷贝了一份，当对于数字这种不可变的，当改变了之后会新创建个对象，新地址存到第二个数组里，第一个数组的第一个地址还是指向原来的那个元素
>
> """
> 浅拷贝的本质：只复制容器（外层），不复制内容（内层）
>
> 关键理解：Python中的数据类型分为可变和不可变
> """
>
> print("=== 核心概念：可变 vs 不可变对象 ===")
>
> # Python中的不可变对象
> immutable_types = [1, 2.5, "hello", (1, 2), True]
> print("不可变对象（修改时会创建新对象）:")
> for obj in immutable_types:
>     print(f"  {obj} -> {type(obj).__name__}")
>
> # Python中的可变对象  
> mutable_types = [[1, 2], {"a": 1}, {1, 2}]
> print("\n可变对象（修改时在原地修改）:")
> for obj in mutable_types:
>     print(f"  {obj} -> {type(obj).__name__}")
>
> print("\n=== 一维数组：为什么浅拷贝表现得像深拷贝 ===")
>
> arr1 = [1, 2, 3]  # 列表容器，包含不可变的整数
> arr2 = arr1.copy()
>
> print("内存分析:")
> print(f"arr1 容器地址: {id(arr1)}")
> print(f"arr2 容器地址: {id(arr2)}")  # 不同地址 - 这是浅拷贝做的
>
> print(f"arr1[0] 的地址: {id(arr1[0])}")
> print(f"arr2[0] 的地址: {id(arr2[0])}")  # 相同地址 - 共享引用！
>
> print("\n关键时刻：修改元素")
> print("执行: arr2[0] = 999")
>
> arr2[0] = 999  # 这里发生了什么？
>
> print(f"\n修改后:")
> print(f"arr1: {arr1}")  # [1, 2, 3]
> print(f"arr2: {arr2}")  # [999, 2, 3]
>
> print(f"arr1[0] 新地址: {id(arr1[0])}")  # 还是指向 1
> print(f"arr2[0] 新地址: {id(arr2[0])}")  # 现在指向 999
>
> print("\n💡 解释：")
> print("   1. arr1[0] 和 arr2[0] 最初都指向同一个整数对象 1")
> print("   2. 当执行 arr2[0] = 999 时，我们是在给 arr2 的第0个位置赋新值")
> print("   3. 整数1是不可变的，所以创建了新的整数对象999")
> print("   4. arr2[0] 现在指向999，但arr1[0] 还是指向原来的1")
> print("   5. 所以看起来互不影响，但本质上仍是浅拷贝！")
>
> print("\n=== 二维数组：浅拷贝的真面目 ===")
>
> matrix1 = [[1, 2], [3, 4]]  # 列表容器，包含可变的列表对象
> matrix2 = matrix1.copy()
>
> print("内存分析:")
> print(f"matrix1 容器地址: {id(matrix1)}")
> print(f"matrix2 容器地址: {id(matrix2)}")  # 不同 - 浅拷贝复制了外层
>
> print(f"matrix1[0] 地址: {id(matrix1[0])}")
> print(f"matrix2[0] 地址: {id(matrix2[0])}")  # 相同！- 内层共享
>
> print("\n关键时刻：修改内层元素")
> print("执行: matrix2[0][0] = 999")
>
> matrix2[0][0] = 999  # 这里发生了什么？
>
> print(f"\n修改后:")
> print(f"matrix1: {matrix1}")  # [[999, 2], [3, 4]] - 被影响了！
> print(f"matrix2: {matrix2}")  # [[999, 2], [3, 4]]
>
> print(f"matrix1[0] 地址: {id(matrix1[0])}")  # 还是同一个列表对象
> print(f"matrix2[0] 地址: {id(matrix2[0])}")  # 还是同一个列表对象
>
> print("\n💡 解释：")
> print("   1. matrix1[0] 和 matrix2[0] 指向同一个列表对象 [1, 2]")
> print("   2. 当执行 matrix2[0][0] = 999 时，我们是在修改共享的列表对象")
> print("   3. 列表是可变的，所以在原地修改")
> print("   4. 由于 matrix1[0] 和 matrix2[0] 指向同一对象，两者都受影响")
>
> print("\n=== 图解浅拷贝 ===")
>
> def visualize_shallow_copy():
>     print("一维数组浅拷贝：")
>     print("原数组:  [1] -> [2] -> [3]")
>     print("        ↗           ↗")
>     print("拷贝: [容器2]")
>     print("修改拷贝[0]=999后:")
>     print("原数组:  [1] -> [2] -> [3]")  
>     print("拷贝: [999] -> [2] -> [3]")
>     print("看起来独立，实际上是不可变对象的特性造成的")
>     
>     print("\n二维数组浅拷贝：")
>     print("原数组:  [[1,2]] -> [[3,4]]")
>     print("          ↗         ↗")  
>     print("拷贝: [容器2]")
>     print("修改拷贝[0][0]=999后:")
>     print("原数组:  [[999,2]] -> [[3,4]]")
>     print("拷贝:    [[999,2]] -> [[3,4]]")
>     print("共享内层对象，所以相互影响")
>
> visualize_shallow_copy()
>
> print("\n=== 验证实验 ===")
>
> # 实验1：替换整个内层对象（一维数组也会这样）
> print("实验1：替换整个子列表")
> matrix = [[1, 2], [3, 4]]
> matrix_copy = matrix.copy()
>
> # 替换整个子列表
> matrix_copy[0] = [999, 888]  # 注意：这是替换，不是修改
> print(f"原矩阵: {matrix}")       # [[1, 2], [3, 4]] - 不受影响
> print(f"拷贝矩阵: {matrix_copy}") # [[999, 888], [3, 4]]
> print("✅ 替换整个对象时，浅拷贝表现得像深拷贝")
>
> # 实验2：修改内层对象
> print("\n实验2：修改子列表内容")
> matrix = [[1, 2], [3, 4]]
> matrix_copy = matrix.copy()
>
> matrix_copy[0][0] = 777  # 修改共享的内层对象
> print(f"原矩阵: {matrix}")       # [[777, 2], [3, 4]] - 受影响
> print(f"拷贝矩阵: {matrix_copy}") # [[777, 2], [3, 4]]
> print("❌ 修改内层对象时，浅拷贝的问题就暴露了")
>
> print("\n=== 总结 ===")
> print("🔑 浅拷贝的本质：")
> print("   • 总是只复制外层容器")
> print("   • 内层对象始终是共享引用")
> print("   • 表现的差异来自于内层对象的可变性")
>
> print("\n📋 记忆技巧：")
> print("   • 不可变对象（int, str, tuple）：修改=替换，所以看起来独立")
> print("   • 可变对象（list, dict, set）：修改=原地改变，所以相互影响")
> print("   • 浅拷贝永远是浅拷贝，只是表现不同而已")





# dict的常见用法

> # 创建字典
> d = {'name': 'Alice', 'age': 25, 'city': 'Beijing'}
> d2 = dict(name='Bob', age=30)
>
> # 访问和修改
> print(d['name'])  # Alice
> d['age'] = 26
> d['email'] = 'alice@example.com'  # 添加新键值对
>
> # 删除
> del d['city']
> removed_value = d.pop('email', '默认值')

keys = d.keys() values = d.values() items = d.items()



> # 同时遍历键和值
> for key, value in d.items():
>     print(f"{key}: {value}")
>
> # 如果不需要解包
> for item in d.items():
>     key, value = item
>     print(f"{key}: {value}")

*# 使用enumerate获取索引* for index, (key, value) in enumerate(d.items()):    print(f"{index}: {key} = {value}")



# Python的内置排序算法

*# sorted() - 返回新的排序列表* arr = [64, 34, 25, 12, 22, 11, 90] sorted_arr = sorted(arr)  *# 升序* sorted_desc = sorted(arr, reverse=True)  *# 降序* *# list.sort() - 原地排序* arr.sort() arr.sort(reverse=True)

> Python的内置排序函数使用**Timsort算法**：
>
> ## 时间复杂度
>
> - **最好情况**: O(n) - 数据已排序
> - **平均情况**: O(n log n) - 随机数据
> - **最坏情况**: O(n log n) - 严格保证
>
> ## 核心特点
>
> - **混合算法**: 结合归并排序 + 插入排序
> - **稳定排序**: 相等元素顺序不变
> - **自适应**: 对部分有序数据性能极佳
> - **空间复杂度**: O(n)
>
> ## 基本原理
>
> ```python
> # Timsort策略：
> # 1. 找到数据中已排序的片段(run)
> # 2. 短片段用插入排序扩展
> # 3. 用归并排序合并这些片段
> # 4. 针对真实数据模式优化
> ```
>
> 这就是为什么Python的`sorted()`和`list.sort()`在实际应用中表现如此优秀的原因 - 它专门为真实世界的数据设计，而不仅仅是理论上的随机数据。

# PYTHON的sort和sorted

> ### 1. `sort()` 和 `sorted()` 的区别
>
> #### `list.sort()`
>
> - **作用**：对列表本身进行**原地排序**（直接修改原列表），不会创建新列表。
> - **返回值**：返回 `None`（所以不能用 `new_list = list.sort()` 这种写法）。
> - **适用对象**：仅适用于列表（`list`）类型。
>
> 
>
> 示例：
>
> 
>
> python
>
> 
>
> 运行
>
> 
>
> 
>
> 
>
> 
>
> ```python
> arr = [3, 1, 2]
> result = arr.sort()  # 原列表被修改，result 是 None
> print(arr)  # 输出：[1, 2, 3]
> print(result)  # 输出：None
> ```
>
> #### `sorted()`
>
> - **作用**：对可迭代对象（列表、元组、字符串等）进行排序，**创建并返回一个新的排序后的列表**，不修改原对象。
> - **返回值**：返回排序后的新列表。
> - **适用对象**：所有可迭代对象（如 `list`、`tuple`、`str` 等）。
>
> 
>
> 示例：
>
> 
>
> python
>
> 
>
> 运行
>
> 
>
> 
>
> 
>
> 
>
> ```python
> arr = [3, 1, 2]
> new_arr = sorted(arr)  # 原列表不变，new_arr 是排序后的新列表
> print(arr)  # 输出：[3, 1, 2]
> print(new_arr)  # 输出：[1, 2, 3]
> 
> # 对字符串排序（返回字符列表）
> s = "cab"
> print(sorted(s))  # 输出：['a', 'b', 'c']
> ```
>
> ### 2. Python 中 `=` 赋值是浅拷贝还是深拷贝？
>
> **`=` 赋值不是拷贝，而是引用传递**。
> 它既不是浅拷贝也不是深拷贝，而是让新变量指向原对象的内存地址，两者共享同一份数据。修改其中一个，另一个也会随之变化。
>
> 
>
> 示例：
>
> 
>
> python
>
> 
>
> 运行
>
> 
>
> 
>
> 
>
> 
>
> ```python
> a = [1, 2, [3, 4]]
> b = a  # b 指向 a 引用的对象
> 
> b[0] = 100
> print(a)  # 输出：[100, 2, [3, 4]]（a 也被修改）
> 
> b[2][0] = 300
> print(a)  # 输出：[100, 2, [300, 4]]（嵌套对象也被修改）
> ```
>
> ### 3. 浅拷贝（Shallow Copy）和深拷贝（Deep Copy）的区别与用法
>
> #### 浅拷贝（Shallow Copy）
>
> - **定义**：创建一个新对象，但新对象中的元素仍然是原对象元素的引用（对于嵌套对象，只拷贝外层，不拷贝内层）。
>
> - **适用场景**：当对象中**没有嵌套的可变元素**（如列表、字典）时，浅拷贝足够用。
>
> - 实现方式
>
>   ：
>
>   - `list.copy()` 方法（仅列表）
>   - 切片操作 `[:]`（仅列表、元组）
>   - `dict.copy()` 方法（仅字典）
>   - `copy.copy()` 函数（通用）
>
> 
>
> 示例：
>
> 
>
> python
>
> 
>
> 运行
>
> 
>
> 
>
> 
>
> 
>
> ```python
> import copy
> 
> a = [1, 2, [3, 4]]
> b = copy.copy(a)  # 浅拷贝
> 
> # 修改外层元素（互不影响）
> b[0] = 100
> print(a)  # 输出：[1, 2, [3, 4]]
> print(b)  # 输出：[100, 2, [3, 4]]
> 
> # 修改嵌套对象（相互影响，因为共享引用）
> b[2][0] = 300
> print(a)  # 输出：[1, 2, [300, 4]]
> print(b)  # 输出：[100, 2, [300, 4]]
> ```
>
> #### 深拷贝（Deep Copy）
>
> - **定义**：创建一个新对象，并且递归地拷贝原对象中的所有元素（包括嵌套对象），新对象与原对象完全独立，没有任何共享引用。
> - **适用场景**：当对象中**包含嵌套的可变元素**时，需要用深拷贝避免相互影响。
> - **实现方式**：`copy.deepcopy()` 函数（通用）。
>
> 
>
> 示例：
>
> 
>
> python
>
> 
>
> 运行
>
> 
>
> 
>
> 
>
> 
>
> ```python
> import copy
> 
> a = [1, 2, [3, 4]]
> b = copy.deepcopy(a)  # 深拷贝
> 
> # 修改外层元素（互不影响）
> b[0] = 100
> print(a)  # 输出：[1, 2, [3, 4]]
> print(b)  # 输出：[100, 2, [3, 4]]
> 
> # 修改嵌套对象（互不影响，因为完全独立）
> b[2][0] = 300
> print(a)  # 输出：[1, 2, [3, 4]]（a 不变）
> print(b)  # 输出：[100, 2, [300, 4]]
> ```
>
> 
>
> 
>
> ![img](data:image/png;base64,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)
>
> ### 总结
>
> | 操作       | 特点                                       | 适用场景                   |
> | ---------- | ------------------------------------------ | -------------------------- |
> | `=`        | 引用传递，不拷贝                           | 简单赋值，共享数据         |
> | 浅拷贝     | 拷贝外层，嵌套对象仍共享引用               | 无嵌套可变元素的对象       |
> | 深拷贝     | 完全拷贝所有元素（包括嵌套对象）           | 有嵌套可变元素的对象       |
> | `sort()`   | 列表原地排序，返回 `None`                  | 需修改原列表时             |
> | `sorted()` | 对可迭代对象排序，返回新列表，不修改原对象 | 需保留原对象，或非列表对象 |

# python中的lambda表达式

> 在 Python 算法题中，`lambda` 表达式主要用于简化匿名函数的定义，特别适合在需要临时定义简单函数的场景中使用。以下是算法题中常见的使用场景，结合具体例子说明：
>
> ### 1. 排序（`sort()` 或 `sorted()` 中的 `key` 参数）
>
> 这是最常见的场景，用于指定排序的依据。
> 例如你提到的 `intervals.sort(key=lambda x: x[0])`，表示按列表中每个元素的第一个元素（`x[0]`）排序。
>
> 
>
> **更多例子：**
>
> 
>
> - 按元素的第二个值排序：
>
>   python
>
>   
>
>   运行
>
>   
>
>   
>
>   
>
>   
>
>   ```python
>   points = [[3,1], [1,2], [2,3]]
>   points.sort(key=lambda x: x[1])  # 按每个子列表的第二个元素排序
>   # 结果: [[3,1], [1,2], [2,3]]
>   ```
>
> - 按字符串长度排序：
>
>   python
>
>   
>
>   运行
>
>   
>
>   
>
>   
>
>   
>
>   ```python
>   words = ["apple", "banana", "cherry"]
>   words.sort(key=lambda x: len(x))  # 按字符串长度排序
>   # 结果: ["apple", "banana", "cherry"]（长度5,6,6）
>   ```
>
> - 按字典的某个键排序：
>
>   python
>
>   
>
>   运行
>
>   
>
>   
>
>   
>
>   
>
>   ```python
>   people = [{"name": "Alice", "age": 25}, {"name": "Bob", "age": 20}]
>   people.sort(key=lambda x: x["age"])  # 按"age"字段排序
>   # 结果: [{"name": "Bob", "age": 20}, {"name": "Alice", "age": 25}]
>   ```
>
> ### 2. 过滤（`filter()` 函数）
>
> 用于筛选序列中满足条件的元素，`lambda` 定义筛选规则。
>
> 
>
> **例子：**
>
> 
>
> - 筛选出列表中的偶数：
>
>   python
>
>   
>
>   运行
>
>   
>
>   
>
>   
>
>   
>
>   ```python
>   nums = [1, 2, 3, 4, 5, 6]
>   even_nums = list(filter(lambda x: x % 2 == 0, nums))
>   # 结果: [2, 4, 6]
>   ```
>
> - 筛选出长度大于 3 的字符串：
>
>   python
>
>   
>
>   运行
>
>   
>
>   
>
>   
>
>   
>
>   ```python
>   words = ["a", "bb", "ccc", "dddd"]
>   long_words = list(filter(lambda x: len(x) > 3, words))
>   # 结果: ["dddd"]
>   ```
>
> ### 3. 映射（`map()` 函数）
>
> 用于对序列中的每个元素做转换，`lambda` 定义转换规则。
>
> 
>
> **例子：**
>
> 
>
> - 将列表中的每个数平方：
>
>   python
>
>   
>
>   运行
>
>   
>
>   
>
>   
>
>   
>
>   ```python
>   nums = [1, 2, 3, 4]
>   squared = list(map(lambda x: x **2, nums))
>   # 结果: [1, 4, 9, 16]
>   ```
>
> - 将字符串转换为小写：
>
>   python
>
>   
>
>   运行
>
>   
>
>   
>
>   
>
>   
>
>   ```python
>   words = ["Apple", "BANANA", "Cherry"]
>   lower_words = list(map(lambda x: x.lower(), words))
>   # 结果: ["apple", "banana", "cherry"]
>   ```
>
> ### 4. 自定义排序规则（复杂场景）
>
> 在需要自定义排序逻辑时，`lambda` 可以结合多条件排序。
>
> 
>
> **例子：**
>
> 
>
> - 先按元素的第一个值升序，再按第二个值降序：
>
>   python
>
>   
>
>   运行
>
>   
>
>   
>
>   
>
>   
>
>   ```python
>   pairs = [(2, 3), (1, 2), (2, 1)]
>   # 先按x[0]升序，再按x[1]降序（通过-x[1]实现降序）
>   pairs.sort(key=lambda x: (x[0], -x[1]))
>   # 结果: [(1, 2), (2, 3), (2, 1)]
>   ```
>
> ### 5. 优先级队列（`heapq` 模块）
>
> 在使用堆时，`lambda` 可定义元素的优先级。
>
> 
>
> **例子：**
>
> 
>
> - 按元组的第二个元素作为优先级构建小顶堆：
>
>   python
>
>   
>
>   运行
>
>   
>
>   
>
>   
>
>   
>
>   ```python
>   import heapq
>   tasks = [("task1", 3), ("task2", 1), ("task3", 2)]
>   # 按第二个元素（优先级）排序，值越小优先级越高
>   heapq.heapify(tasks)  # 直接堆化默认按第一个元素，需配合lambda时可先处理
>   # 或先转换：
>   sorted_tasks = sorted(tasks, key=lambda x: x[1])
>   ```
>
> ### 总结
>
> `lambda` 的核心优势是**简洁**，适合定义单行、逻辑简单的函数。在算法题中，最常用的场景是**排序时指定 `key`**，其次是结合 `filter()`/`map()` 处理序列。如果逻辑复杂（多行代码），则更适合用 `def` 定义普通函数。

# sort排序以及手撕一个排序算法（快速排序）

> 在 Python 中，`sort()` 方法（以及 `sorted()` 函数）使用的是 **Timsort 算法**，这是一种结合了归并排序和插入排序的混合排序算法。其时间复杂度为：
>
> 
>
> - 最佳情况：`O(n)`（当数据已经接近有序时）
> - 平均情况：`O(n log n)`
> - 最坏情况：`O(n log n)`
>
> 
>
> Timsort 是为实际应用中的真实数据设计的，在处理部分有序的数据时表现尤其出色，这也是 Python 选择它作为内置排序算法的原因。
>
> 
>
> 如果面试官要求手撕一个高效的排序算法，**快速排序（Quicksort）** 是一个很好的选择，它实现相对简洁，平均时间复杂度为 `O(n log n)`，在实际应用中性能优秀。
>
> 
>
> 以下是快速排序的实现代码：
>
> def quick_sort(nums):
>     # 递归终止条件：数组长度小于等于1时无需排序
>     if len(nums) <= 1:
>         return nums
>     
>     # 选择基准值（pivot），这里选择中间元素
>     pivot = nums[len(nums) // 2]
>         
>     # 分治：将数组分为小于、等于、大于基准值的三个部分
>     left = [x for x in nums if x < pivot]
>     middle = [x for x in nums if x == pivot]
>     right = [x for x in nums if x > pivot]
>         
>     # 递归排序左右两部分，并合并结果
>     return quick_sort(left) + middle + quick_sort(right)
>
>
> # 测试示例
> if __name__ == "__main__":
>     test_nums = [3, 6, 8, 10, 1, 2, 1]
>     print("排序前:", test_nums)
>     print("排序后:", quick_sort(test_nums))  # 输出：[1, 1, 2, 3, 6, 8, 10]

> 快速排序的平均时间复杂度为 `O(n log n)`，核心原因在于其**分治策略**和**递归深度**的特性，具体可以从以下角度理解：
>
> ### 1. 分治策略的时间开销
>
> 快速排序的核心是「分区（partition）」操作：
>
> 
>
> - 每次选择一个基准值（pivot），将数组分为两部分（小于基准值 + 大于基准值）。
> - 分区过程需要遍历整个子数组，时间复杂度为 `O(k)`（`k` 是当前子数组的长度）。
>
> ### 2. 递归的层次结构
>
> 假设每次分区都能将数组「均匀地」分成两半（理想情况）：
>
> 
>
> - 第 1 层：处理整个数组（长度 `n`），分区时间 `O(n)`。
> - 第 2 层：处理两个子数组（各长度 `n/2`），总时间 `O(n/2) + O(n/2) = O(n)`。
> - 第 3 层：处理四个子数组（各长度 `n/4`），总时间 `O(n/4)*4 = O(n)`。
> - ...
> - 直到子数组长度为 1（无法再分），递归终止。
>
> ### 3. 递归的总层数
>
> 由于每次都将数组分成两半，递归的总层数为 `log n`（以 2 为底）：
>
> 
>
> - 例如，`n=8` 时，需要 3 层（`8→4→2→1`，共 3 次分割）。
> - 层数 = 「将 `n` 不断除以 2 直到等于 1 的次数」，即 `log2(n)`。
>
> ### 4. 总时间复杂度计算
>
> 每一层的总时间开销都是 `O(n)`（各层子数组长度之和始终为 `n`），而总层数是 `O(log n)`，

# python中的set

在 Python 中，`set` 是基于哈希表实现的无序集合，具有元素唯一性和高效的操作性能。以下是其常见用法及对应的时间复杂度：

### 1. 集合的创建

| 操作              | 示例                | 时间复杂度 | 说明                                                     |
| ----------------- | ------------------- | ---------- | -------------------------------------------------------- |
| 使用 `{}` 创建    | `s = {1, 2, 3}`     | O(n)       | n 为元素个数，初始化时需为每个元素计算哈希值             |
| 使用 `set()` 转换 | `s = set(iterable)` | O(n)       | 遍历可迭代对象（如列表、字符串）并去重，n 为输入元素总数 |



**注意**：空集合必须用 `set()` 创建（`{}` 表示空字典）。

### 2. 元素的添加

| 操作               | 示例               | 时间复杂度 | 说明                                         |
| ------------------ | ------------------ | ---------- | -------------------------------------------- |
| `add(x)`           | `s.add(4)`         | O(1)       | 向集合添加单个元素，哈希表插入操作           |
| `update(iterable)` | `s.update([5, 6])` | O(k)       | 添加可迭代对象中的所有元素，k 为输入元素个数 |

### 3. 元素的删除

| 操作         | 示例           | 时间复杂度 | 说明                                                |
| ------------ | -------------- | ---------- | --------------------------------------------------- |
| `remove(x)`  | `s.remove(3)`  | O(1)       | 删除指定元素，若元素不存在则抛出 `KeyError`         |
| `discard(x)` | `s.discard(3)` | O(1)       | 删除指定元素，若元素不存在则不操作                  |
| `pop()`      | `s.pop()`      | O(1)       | 随机删除并返回一个元素（集合为空时抛出 `KeyError`） |
| `clear()`    | `s.clear()`    | O(n)       | 清空集合，需删除所有元素，n 为集合大小              |

### 4. 成员判断

| 操作         | 示例         | 时间复杂度 | 说明                                 |
| ------------ | ------------ | ---------- | ------------------------------------ |
| `x in s`     | `3 in s`     | O(1)       | 判断元素是否在集合中，哈希表查找操作 |
| `x not in s` | `3 not in s` | O(1)       | 判断元素是否不在集合中               |

### 5. 集合运算

| 操作                                        | 示例                                   | 时间复杂度             | 说明                                     |                    |                            |
| ------------------------------------------- | -------------------------------------- | ---------------------- | ---------------------------------------- | ------------------ | -------------------------- |
| 交集（`&` 或 `intersection()`）             | `a & b` 或 `a.intersection(b)`         | O(min(len(a), len(b))) | 取两集合的共同元素                       |                    |                            |
| 并集（`                                     | `或`union()`）                         | `a                     | b`或`a.union(b)`                         | O(len(a) + len(b)) | 取两集合的所有元素（去重） |
| 差集（`-` 或 `difference()`）               | `a - b` 或 `a.difference(b)`           | O(len(a))              | 取仅在 a 中存在的元素                    |                    |                            |
| 对称差集（`^` 或 `symmetric_difference()`） | `a ^ b` 或 `a.symmetric_difference(b)` | O(len(a) + len(b))     | 取仅在 a 或 b 中存在的元素（不同时存在） |                    |                            |

### 6. 集合关系判断

| 操作                               | 示例                          | 时间复杂度             | 说明                           |
| ---------------------------------- | ----------------------------- | ---------------------- | ------------------------------ |
| 子集判断（`issubset()` 或 `<=`）   | `a.issubset(b)` 或 `a <= b`   | O(len(a))              | 判断 a 的所有元素是否都在 b 中 |
| 超集判断（`issuperset()` 或 `>=`） | `a.issuperset(b)` 或 `a >= b` | O(len(b))              | 判断 b 的所有元素是否都在 a 中 |
| 无交集判断（`isdisjoint()`）       | `a.isdisjoint(b)`             | O(min(len(a), len(b))) | 判断 a 和 b 是否没有共同元素   |

### 7. 其他常用操作

| 操作                 | 示例       | 时间复杂度 | 说明                                           |
| -------------------- | ---------- | ---------- | ---------------------------------------------- |
| 计算长度（`len()`）  | `len(s)`   | O(1)       | 返回集合中元素的个数（哈希表维护了长度计数器） |
| 复制集合（`copy()`） | `s.copy()` | O(n)       | 创建集合的浅拷贝，n 为集合大小                 |

### 总结

- `set` 的核心优势是 **O (1) 时间复杂度的增、删、查操作**，这得益于哈希表的实现。
- 集合运算的复杂度取决于参与运算的集合大小，适合处理需要去重或关系判断的场景（如列表去重、查找共同元素等）。
- 局限性：集合是无序的（无法通过索引访问），且元素必须是可哈希的（如整数、字符串、元组，不能包含列表、字典等）。
