# 数组+哈希表

## **LeetCode27——**移除元素

#### 思路1：创建一个等大的空间，但不知道python怎么创建列表，

#### 题解思路1：双指针法

大致思路：这个题的要求就是左边是不是val的数，所以利用双指针，左指针每次指向可以赋值的地方，右指针遍历

第一次解题代码（通过）：

> class Solution:
>
>   def removeElement(self, nums: List[int], val: int) -> int:
>
> ​    left,right = 0,0
>
> ​    numsLen = len(nums)
>
> ​    while right<numsLen:
>
> ​      if nums[right] == val:
>
> ​        right = right + 1
>
> ​      else:
>
> ​        nums[left] = nums[right]
>
> ​        left = left + 1
>
> ​        right = right + 1
>
> ​    return left

学习提升：1.但凡涉及到数组遍历，就思考一下利用双指针行不行

2. python中没有++。你就用+=         left+=1
3. 如果想要复制一维数组直接.copy(),修改互不影响，如果复制二维数组不能用copy，修改原来数组的值，复制的数组的值也会改变。如果只是单纯的创建一个数组，就直接arr = []

#### 双指针法的延申：

1.双向双指针（对撞指针）

> **适用场景：**
>
> - 有序数组中查找满足条件的元素对
> - 字符串回文判断
> - 反转数组/字符串
> - 三数之和、四数之和问题
>
> **经典题目：**
>
> #### 🔥 基础题
>
> - **LeetCode 167 - 两数之和 II（输入有序数组）**
> - **LeetCode 344 - 反转字符串**
> - **LeetCode 125 - 验证回文串**
> - **LeetCode 11 - 盛最多水的容器**
>
> #### 🔥 进阶题
>
> - **LeetCode 15 - 三数之和**
>
> - **LeetCode 18 - 四数之和**
>
> - **LeetCode 42 - 接雨水**
>
> - # 对撞指针模板
>   def two_sum_sorted(nums, target):
>       left, right = 0, len(nums) - 1
>       while left < right:
>           current_sum = nums[left] + nums[right]
>           if current_sum == target:
>               return [left, right]
>           elif current_sum < target:
>               left += 1
>           else:
>               right -= 1
>       return []

2.2. 快慢指针（同向双指针）

> **特点：** 两个指针同方向移动，速度不同
>
> **适用场景：**
>
> - 数组元素删除/移动
> - 去重操作
> - 链表环检测
> - 寻找链表中点
>
> **经典题目：**
>
> #### 🔥 数组操作
>
> - **LeetCode 27 - 移除元素**（就是你刚才的题目！）
> - **LeetCode 26 - 删除有序数组中的重复项**
> - **LeetCode 80 - 删除有序数组中的重复项 II**
> - **LeetCode 283 - 移动零**
> - **LeetCode 88 - 合并两个有序数组**
>
> #### 🔥 链表问题
>
> - **LeetCode 141 - 环形链表**
> - **LeetCode 142 - 环形链表 II**
> - **LeetCode 876 - 链表的中间结点**
> - **LeetCode 19 - 删除链表的倒数第N个节点**
>
> 
>
> python

```python
# 快慢指针模板（数组去重）
def remove_duplicates(nums):
    slow = 0
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]
    return slow + 1

# 快慢指针模板（链表环检测）
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False
```

3. 滑动窗口（特殊的双指针）

> **特点：** 维护一个窗口，动态调整窗口大小
>
> **适用场景：**
>
> - 子串/子数组问题
> - 最长/最短满足条件的区间
> - 固定长度窗口的最值问题
>
> **经典题目：**
>
> #### 🔥 定长窗口
>
> - **LeetCode 643 - 子数组最大平均数 I**
> - **LeetCode 1456 - 定长子串中元音的最大数目**
>
> #### 🔥 不定长窗口
>
> - **LeetCode 3 - 无重复字符的最长子串**
> - **LeetCode 76 - 最小覆盖子串**
> - **LeetCode 209 - 长度最小的子数组**
> - **LeetCode 424 - 替换后的最长重复字符**
> - **LeetCode 438 - 找到字符串中所有字母异位词**
>
> 
>
> python

```python
# 滑动窗口模板
def sliding_window(s, target):
    left = 0
    window = {}
    result = []
    
    for right in range(len(s)):
        # 扩展窗口
        char = s[right]
        window[char] = window.get(char, 0) + 1
        
        # 收缩窗口
        while condition_met(window, target):
            # 更新结果
            result.append(left)
            
            # 左指针右移
            left_char = s[left]
            window[left_char] -= 1
            if window[left_char] == 0:
                del window[left_char]
            left += 1
    
    return result
```



## LeetCode 1 两数之和

#### 思路一：就是先把数组排序一下然后用双指针，但问题是现在不清楚用什么算法排序，等会chat一下

你真是脑残，这个一看就有问题，你排序之后数的位置不就变了吗？，你这个还得记录原来数据的位置，更麻烦了，错误的代码：

> class Solution:
>
>   def twoSum(self, nums: List[int], target: int) -> List[int]:
>
> ​    nums.sort()
>
> ​    left,right = 0,len(nums)-1
>
> ​    while left<right:
>
> ​      if nums[left]+nums[right] > target:
>
> ​        right -= 1
>
> ​      elif nums[left]+nums[right] < target:
>
> ​        left += 1
>
> ​      else:
>
> ​        return [left,right]

#### 学习思路一：使用哈希表。这个刚开始考虑到了，但想成c++的了（创建一个特别大的数组），后面看题解发现竟然可以用dict作为哈希表，我真服了，等会补充一下dict的知识。不会拼写enumerate

```python
class Solution:

  def twoSum(self, nums: List[int], target: int) -> List[int]:

​    hashTable = dict()

​    for i,num in enumerate(nums):

​      if target-num in hashTable:

​        return [hashTable[target-num],i]

​      else:

​        hashTable[num] = i
```



## LeetCode 49 字母异位词分组

思路1：很麻烦很麻烦，写到一半不知道怎么用哈希表来判断他们俩是不是一个了：

> class Solution:
>
> ​    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
> ​        result = []
> ​        strsLen = len(strs)
> ​        flags = [0 for i in range(strsLen)]  # 修正：range(strsLen)
> ​        
> ​        for i in range(strsLen):  # 修正：range(strsLen)
> ​            if flags[i] == 1:
> ​                continue
> ​            flags[i] = 1
> ​            tempList = [strs[i]]  # 先把当前字符串加入
> ​            
>    ​         # 构建当前字符串的字符频率表
> ​            hashTable = {}
> ​            for j in strs[i]:
> ​                if j in hashTable:
> ​                    hashTable[j] += 1
> ​                else:
> ​                    hashTable[j] = 1
> ​            
>
>             # 检查后续字符串是否为异位词
> ​            k = i + 1
> ​            while k < strsLen:
> ​                if flags[k] == 0:  # 只检查未处理的字符串
>    ​                 # 构建当前待比较字符串的字符频率表
> ​                    tempHashTable = {}
> ​                    for s in strs[k]:
> ​                        if s in tempHashTable:
> ​                            tempHashTable[s] += 1
> ​                        else:
> ​                            tempHashTable[s] = 1
> ​                    
>    ​                 # 比较两个哈希表是否相同
> ​                    if hashTable == tempHashTable:
> ​                        tempList.append(strs[k])
> ​                        flags[k] = 1  # 标记为已处理
> ​                k += 1
> ​            
> ​            result.append(tempList)
> ​        
> ​        return result

官方计数代码太夸张了：

> class Solution:
>
>   def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
>
> ​    mp = collections.defaultdict(list)
>
> 
>
> ​    for st in strs:
>
> ​      counts = [0] * 26
>
> ​      for ch in st:
>
> ​        counts[ord(ch) - ord("a")] += 1
>
> ​      \# 需要将 list 转换成 tuple 才能进行哈希
>
> ​      mp[tuple(counts)].append(st)
>
> ​    
>
> ​    return list(mp.values())



> 
>
> class Solution:
>
>   def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
>
> ​    mp = collections.defaultdict(list)
>
> 
>
> ​    for st in strs:
>
> ​      key = "".join(sorted(st))
>
> ​      mp[key].append(st)
>
> ​    
>
> ​    return list(mp.values())

#### 解法1：（报错了，list.sort()是原地排序，并返回none），自己解决的思路：用字典存储，利用tuple来当key

```python
class Solution:

  def groupAnagrams(self, strs: List[str]) -> List[List[str]]:

​    hashMap = {}

​    result = []

​    for i in range(len(strs)):

​      tempList = [j for j in strs[i]]

​      tempList.sort()

​      if tuple(tempList) in hashMap.keys():

​        hashMap[tuple(tempList)].append(strs[i])

​      else:

​        hashMap[tuple(tempList)] = [strs[i]]

​    for i in hashMap.keys():

​      result.append(hashMap[i])

​    return result
```

#### 官方题解  学习点1：

class Solution:

  def groupAnagrams(self, strs: List[str]) -> List[List[str]]:

​    mp = collections.defaultdict(list)



​    for st in strs:

​      key = "".join(sorted(st))

​      mp[key].append(st)

​    

​    return list(mp.values())

> 在这个字母异位词分组的解法中，使用 `collections.defaultdict` 而非 Python 内置的普通 `dict`，主要是为了**简化代码逻辑并避免键不存在的错误**。
>
> 具体来说，两者的核心差异在于处理「键不存在」的场景：
>
> 1. **普通 `dict` 的问题**
>    当你尝试给一个不存在的键赋值时（如 `mp[key].append(st)`），普通字典会直接抛出 `KeyError`。
>    因此需要先判断键是否存在，代码会更繁琐：
>
>    python
>
>    
>
>    运行
>
>    
>
>    ```python
>    mp = {}
>    for st in strs:
>        key = "".join(sorted(st))
>        if key not in mp:  # 必须先判断，否则可能报错
>            mp[key] = []
>        mp[key].append(st)
>    ```
>
>    **`defaultdict` 的优势**
>    `collections.defaultdict` 允许你指定一个「默认工厂函数」（如 `list`）。
>    当访问不存在的键时，它会自动调用该函数创建一个默认值（如空列表 `[]`），无需手动初始化：
>
>    
>
>    ```python
>    mp = collections.defaultdict(list)  # 指定默认值为列表
>    for st in strs:
>        key = "".join(sorted(st))
>        mp[key].append(st)  # 键不存在时自动创建空列表，直接 append 即可
>    ```
>
>    
>
> 
>
> 简单说，`defaultdict` 在这里的作用是**省去了判断键是否存在并手动初始化的步骤**，让代码更简洁、可读性更高。
>
> 
>
> 在功能上，两种写法最终结果完全一致，但 `defaultdict` 版本的代码更优雅，尤其适合这种「键对应的值是容器类型（列表、集合等）」的场景。



## Leetcode128 最长连续序列

#### 思路1：没考虑到数字重复的情况：（待解决：遍历）

class Solution:

  def longestConsecutive(self, nums: List[int]) -> int:

​    nums.sort()

​    arr = [1]*len(nums)

​    for i in range(len(nums)): #怎么从1开始遍历呢？

​      if i == 0:

​        continue

​      if nums[i]-1 == nums[i-1]:

​        arr[i] = arr[i-1] + 1

​    return max(arr) 

最后通过先转换成set再排序，解决：

class Solution:

  def longestConsecutive(self, nums: List[int]) -> int:

​    if len(nums) == 0:

​      return 0

​    nums = list(set(nums))

​    nums.sort()

​    arr = [1]*len(nums)

​    for i in range(len(nums)): #怎么从1开始遍历呢？

​      if i == 0:

​        continue

​      if nums[i]-1 == nums[i-1]:

​        arr[i] = arr[i-1] + 1

​    return max(arr) 

​      
