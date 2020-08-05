# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 12:08:15 2020

@author: Pawel
"""

"""
1. Find duplicats in a list and remove all but the one that has no duplicates
"""

"""
My Solution
""""

def singleNumber(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    from collections import Counter
    Sol=[]
    Sol1=[]
    Sol1=[k for k,v in Counter(nums).items() if v>1]
    Sol = [x for x in nums if x not in Sol1]
    return Sol[0]

"""
Better Solution: add to set, if it repeats, remove from set since the number only appears twice
""""
class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        s = set()
        for elt in nums:
            if elt in s:
                s.remove(elt)
            else:
                s.add(elt)
        return s.pop()
    
"""
2. Rotate an input array

"""
class Solution(object):
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: None Do not return anything, modify nums in-place instead.
        
        """
        x=len(nums)
        if x>1 and k>0:
            if x<k:
                a=k%x
                nums.extend(nums[:(x-a)])
                for ii in range(x-a):
                    nums.pop(0)
            else:
                nums.extend(nums[:(x-k)])
                for ii in range(x-k):
                    nums.pop(0)

"""
Better Solution: using list slicing
""""                    
                    
class Solution(object):
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        k %= len(nums)
        if k == 0:
            return
    
        nums.reverse()
        nums[:k] = nums[:k][::-1]
        nums[k:] = nums[k:][::-1] 

"""
3. Keep only intersection of elements
"""                   

class Solution(object):
    def intersect(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        A=[]
        for elem in (list(nums1)):
            if elem in nums2:
                A.append(elem)
                nums2.remove(elem)
        return A        
    
"""
Better Solution: using dict
""""      
    
class Solution(object):
def intersect(self, nums1, nums2):
    """
    :type nums1: List[int]
    :type nums2: List[int]
    :rtype: List[int]
    """
    dict = {}
    for i in nums1:
        if i in dict:
            dict[i] += 1
        else:
            dict[i] = 1
    ans = []
    for i in nums2:
        if i in dict and dict[i]:
            dict[i] -= 1
            ans.append(i)
    return ans
    
   

'''
4. Sums of two number to match target - return index of elements
'''

class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        Ans=[]
        Counter=0
        while len(Ans)!=2:
            Ans=[]
            Values=[x+nums[Counter] for x in nums]
            if target in Values:
                Ans.append(Counter)
                Ans.append(Values.index(target))
   
            Counter +=1
            Ans = set(Ans)
        return Ans

''''
Better solution
'''
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        wanted_dict = {}
        
        for i in range(len(nums)):
            if nums[i] in wanted_dict:
                return [wanted_dict[nums[i]], i]
            else:
                wanted_dict[target-nums[i]] = i
        
        return [wanted_dict[nums[i]], i]
    
    
''''
5. Add plus one to an integer in an array that shows all digits [1,2,3,4] is 1234
'''
class Solution(object):
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
                """
        
        
        if digits[-1]==9:
            ii=len(digits)-1
            while digits[ii]==9:
                digits[ii]=0
                ii=ii-1
            if digits[0]==0:
                digits.insert(0,1)
            else:
                digits[ii]=digits[ii]+1
        else:
            digits[-1]=digits[-1]+1
        
        return digits
    
    

'''
6. Move all zeros to end of list, working with one list
'''

class Solution(object):
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        Number = nums.count(0)
        nums[:] = [x for x in nums if x != 0]
        nums.extend([0]*Number)    
        
        
"""
7. Get first unique non-repeating char
"""
 

class Solution(object):
    def firstUniqChar(self, s):
        """
        :type s: str
        :rtype: int
        """
        #char_count = lambda s, char: s.count(char)
        #my_dict = dict( [ ( char, char_count(s, char) ) for char in s ]  )
        #from collections import Counter
        #counts=Counter(s)
        NO_OF_CHARS = 256
        def getCharCountArray(string): 
            count = [0] * NO_OF_CHARS 
            for i in string: 
                count[ord(i)]+=1
            return count 
        
        counts=getCharCountArray(s)
        print(counts)
        index=-1
        k=0
        for ii in s:
            if counts[ord(ii)]==1:
                index=k
                break
            k=k+1
            
        return index
        


"""
Better Solution- Build own dictionary
"""
 class Solution(object):
    def firstUniqChar(self, s):
        """
        :type s: str
        :rtype: int
        """
        d = {}
        for x in s: 
            if x not in d:
                d[x] = 1
            else:
                d[x] += 1 
        for key, value in enumerate(s):
            if d[value] == 1:
                return key
        return -1   
        
    
  """
8. Implement ATOI
"""
 

class Solution(object):
    def myAtoi(self, str):
        """
        :type str: str
        :rtype: int
        """
        
        
        str=str.strip()
        if len(str)==0:
            return 0
        sign=1
        i=0
        if str[0] == "-": #Check sign, record and shift the for loop by 1
            sign=-1
            i=1
        if str[0] == "+":
            sign=1
            i=1
        Final=0
        for j in range(i, len(str)): 
            if str[j].isdigit()==False: 
                break
            else:
                Final = Final * 10 + (int(str[j])) 
        if Final >  2**31-1 and sign==1:
            Final = 2**31 -1
        if Final >  2**31-1 and sign==-1:
            Final = 2**31
        
        return sign * Final 
        
   """
9. Needle in haystack
"""
 
class Solution(object):
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        Len=len(needle)
        Res=-1
        if needle=="":
            Res=0
        else:
            for ii in range(len(haystack)-Len+1):
                if haystack[ii:ii+Len]==needle:
                    Res=ii
                    break
        return Res      
    
    
"""
Better solution
"""""
class Solution(object):
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        return haystack.find(needle)
    
    
 """
10. Reverse singe linked list
""""

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        prev = None
        curr = head

        while curr:
            next = curr.next
            curr.next = prev
            prev = curr
            curr = next

        return prev   
    
"""
11. Linked - check if values are palindromes
"""
# Bad solution, but converting to array, so at least a method;

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        curr=head
        arr=[]
        while curr:
            arr.append(curr.val)
            curr=curr.next
        for ii in range(len(arr)//2):
            if arr[ii]!=arr[-ii-1]:
                return False
        return True
    
"""
Better Solution
"""

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        # val = []
        # while head:
        #     val.append(head.val)
        #     head = head.next
        # return val == val[::-1]
        
        slow = fast = head    #moves faster to find mid-point. Once finds mid-point, reverses list and comapres vals
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
        prev = None
        while slow:
            temp = slow.next
            slow.next = prev
            prev = slow
            slow = temp
        slow = prev
        while slow:
            if slow.val != head.val:
                return False
            slow = slow.next
            head = head.next
        return True
    
    
"""
12. Merge two sorted linked list together
"""

class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        if not l1 and not l2:
            return None
        elif not l1:
            return l2
        elif not l2:
            return l1
        
        if l1.val <= l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        head = dummy = ListNode(0)

        while l1 and l2:
            if l1.val < l2.val:
                head.next = ListNode(l1.val)  # or l1
                l1 = l1.next
            else:
                head.next = ListNode(l2.val)  # or l2
                l2 = l2.next
            head = head.next
        head.next = l1 or l2
        return dummy.next     
            
"""
13. Delete linked list element
"""
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        node.val = node.next.val
        node.next=node.next.next
                
"""
14. Delete nth element from end
"""
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        
        counter=0
        curr=head
        curr2=dummy=ListNode(0)
        while curr:
            curr=curr.next
            counter+=1
        if counter==1:
            return None
        counter2=0
        while counter2!=(counter-n):
            print(curr2)
            counter2+=1
            curr2.next=ListNode(head.val)
            head=head.next
            curr2=curr2.next
        if n!=1:
            curr2.next=head.next
            curr2=curr2.next
            counter2+=2
            head=head.next
            while counter2!=counter:
                counter2+=1
                curr2.next=head.next
                head=head.next
                curr2=curr2.next
        return dummy.next
                   
"""
Better solution - first goes forward n and then they move jointly. 
"""            
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        first, second = head, head
        
        for i in range(n):
            first = first.next
        if not first:
            return head.next
        
        while first.next:
            first = first.next
            second = second.next
        second.next = second.next.next
        return head
    
    
"""
15. Sorting - Finding the version that is false
"""
# The isBadVersion API is already defined for you.
# @param version, an integer
# @return a bool
# def isBadVersion(version):

class Solution(object):
    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        
        Upper=n
        Lower=1
        ResU=isBadVersion(Upper)
        ResL=isBadVersion(Lower)
        Guess_New=n//2
        Guess_Upper=n
        Guess_Lower=1
        if ResU!=ResL:
            while Guess_Upper-Guess_Lower >1:
                New_Guess = isBadVersion(Guess_New)
                if New_Guess==ResU:
                    Guess_Upper=Guess_New
                    Guess_New=Guess_Upper-(Guess_Upper-Guess_Lower)//2
                    Res_U=New_Guess
                else:
                    Guess_Lower= Guess_New
                    Guess_New=Guess_Lower+(Guess_Upper-Guess_Lower)//2
            return Guess_Upper
        else:
            return 1
                   
"""
Better solution - simpler version of bisection method
"""            
class Solution(object):
    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        l, r = 1, n
        
        while l<r:
            mid = l+(r-l)//2
            if isBadVersion(mid):
                r = mid
            else:
                l = mid+1
                
        return r
    
"""
15. Sorting - Merging two sorted lists
"""
# The isBadVersion API is already defined for you.
# @param version, an integer
# @return a bool
# def isBadVersion(version):

class Solution(object):
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: None Do not return anything, modify nums1 in-place instead.
        """
        Counter=0
        if m!=0: 
            if n!=0:
                for ii in range(n):
                    while nums2[ii]>nums1[Counter] and Counter<m:
                        Counter+=1
                    nums1.insert(Counter,nums2[ii])
                    Counter+=1
                    m+=1
                del nums1[-n:]     
        else:
            for ii in range(n):
                nums1.insert(ii,nums2[ii])
            del nums1[-n:]    
                   
"""
Better solution - inserts into list backwards
"""            
class Solution(object):
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: None Do not return anything, modify nums1 in-place instead.
        """
        i = m-1
        j = n-1
        k = len(nums1)-1
        while i>-1 and j>-1:
            if nums1[i]>nums2[j]:
                nums1[k]=nums1[i]
                i-=1
            else:
                nums1[k]=nums2[j]
                j-=1
            k-=1
        while j>-1 and k>-1:
            nums1[k]=nums2[j]
            j-=1
            k-=1
                
        return r    
"""
16. Fibbonaci Seq
"""
# The isBadVersion API is already defined for you.
# @param version, an integer
# @return a bool
# def isBadVersion(version):

class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        FibList=[]
        FibList.append(1)
        FibList.append(1)
        for ii in range(n-1):
            FibList.append(FibList[-1]+FibList[-2])
        
        return FibList[n]
"""
Better solution 
"""            
class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        a,b = 1,1
        for i in range(2,n+1):
            c = a+b
            a,b = b,c
        return b


"""
17. Buy sell stock from list
"""


class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        temp = 0
        ans = 0
        for i in range(len(prices)-1):
            temp += prices[i+1] - prices[i]
            if temp < 0:
                temp = 0
            ans = max(ans,temp)
        return ans
    
"""
My Terrible answer
"""
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        n=len(prices)
        if n==0 or n==1:
            return 0
        Profit=[]
        for ii in range(n-1):
            Profit.append(max(prices[ii+1:])-prices[ii])
        ProfitFinal=max(Profit)
                         
        if ProfitFinal<0:
            return 0
        return ProfitFinal
    

"""
18. Find highest sum in array
"""


class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        Indiv=-1
        n=len(nums)
        if n==1:
            return nums[0]
        Counter=n
        print(Counter)
        while Indiv<0 and Counter>0:
            Indiv=nums[Counter-1]
            Counter=Counter-1
        
        if Counter==0:
            return max(nums)
        Init=Indiv
        Update=Indiv
        print(n)
        for ii in range(Counter):
            Update=Update+nums[Counter-ii-1]
            if Update > Init:
                Init=Update
            elif Update<0:
                Update=0
        return Init  
"""
Better answer
"""
class Solution(object):
    def maxSubArray(self, nums):
        if not nums:
            return 0 
        n=len(nums)
        # dp=[[0] for i in range(n)]
        dp=[0]*n
        dp[0]=nums[0]
        # maxs=dp[0]
        for i in range(1,n):
            dp[i]=max(dp[i-1]+nums[i],nums[i]) #以nums[i]为结束点的最大子段和
            # if dp[i]>maxs:
            #     maxs=dp[i]
            # maxs=max(dp[i],maxs)
        # return maxs
        return max(dp)
    
    
"""
19.House Rob - not adjacent
""""
class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n=len(nums)
        R1=0
        R2=0
        for ii in range(n):
            Val=max(R1+nums[ii],R2)
            R1=R2
            R2=Val    
        return max(R1,R2)
        
                
"""
20.House Rob - not adjacent
""""
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        Profit=0
        Curr=0
        Prev=prices[0]
        n=len(prices)
        for ii in range(1,n):
            Profit = max(Profit,Profit+prices[ii]-prices[ii-1])  #Update profits by moving forward, if negative  you wouldn't add it                                                                       you wouldn't add it
            
        return Profit
    
"""
21. Group Anagrams
"""
class Solution(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        
        NO_OF_CHARS = 256
        def getCharCountArray(string): 
            count = [0] * NO_OF_CHARS 
            for i in string: 
                count[ord(i)]+=1
            return count 
        Unique_Set=[]
        Final_Set=[]
        n1=len(strs)
        Elem=getCharCountArray(strs[0])
        Final_Set.append([strs[0]])
        Unique_Set.append(Elem)
        for ii in range(1,n1):
            Elem=getCharCountArray(strs[ii])
            n2=len(Unique_Set)
            There=False
            for jj in range(n2):
                if Elem==Unique_Set[jj]:
                    Final_Set[jj].append(strs[ii])
                    There=True
                    break
            if There==False:
                Final_Set.append([strs[ii]])
                Unique_Set.append(Elem)
        return Final_Set
    
"""
Sol I understand fully
"""
class Solution(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        
        dict = {}        
        for str in strs:
            key_str = ''.join(sorted(str))  ##Sorts in alph order and creates three elements which join merges
            if key_str in dict:
                dict[key_str].append(str)                
            else:
                dict[key_str] = [str]
        print(dict)
        return list(dict.values())    
    
"""
Better sol
"""
class Solution(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]   
        :rtype: List[List[str]]
        """
        hash_map = collections.defaultdict(list)
        for string in strs:
            hash_map["".join(sorted(string))].append(string)
        return list(hash_map.values())
    
"""
22. Length on longest Substring
"""
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        n=len(s)
        #print(n)
        if n==1:
            return 1
        my_dict={}
        Counter=0
        Final=-1
        
        ii=0
        while ii<(n):
            #print(ii)
            if s[ii] in my_dict:
                if Counter>Final:
                    Final=Counter
                Counter=0
                ii=my_dict[s[ii]]+1
                #print(ii)
                my_dict={}
                #print(my_dict)
            else:
                my_dict[s[ii]]=ii
                Counter=Counter+1
                ii=ii+1
        if Final==0 or Counter>Final:
            Final=Counter
        return Final

"""
Better Solution
""""
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        left, right, longest = 0, 0, 0
        chars = set()
        
        while left < len(s) and right < len(s):       #Moves one pointer, when hits duplicate, removes until right pointer is clear again
            if s[right] not in chars:
                chars.add(s[right])
                right += 1
                longest = max(longest, right - left)
            else:
                chars.remove(s[left])
                left += 1
            
        return longest
            
 """
23. Is there a substring of 3 increasing numbers

"""   
class Solution(object):
    def increasingTriplet(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        n = len(nums)
        first = float("inf")
        second = float("inf")

        for i in range(n):
            if nums[i] < first:
                first = nums[i]
            elif first < nums[i] < second:
                second = nums[i]
            if nums[i] > second:
                return True

        return False
    
""""
24. Sum linked list integer and create new linked list
""""
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        head = dummy = ListNode(0)
        Store=0
        while l1 or l2:
            if l1==None:
                Val1=0
            else:
                Val1=l1.val
                l1=l1.next
            if l2==None:
                Val2=0
            else:
                Val2=l2.val
                l2=l2.next
            Res=Val1+Val2+Store
            Store=0
            if Res>=10:
                Store=1
                Res=Res-10
            head.next=ListNode(Res)
            head=head.next
        if Store!=0:
            head.next=ListNode(1)
            
        return dummy.next
    
"""
25. When do linked list share common node:
"""

"""
Hash table method
"""
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """

        hash_table_A = {}
        while headA != None:
            hash_table_A[headA] = headA.next
            headA = headA.next
            #print(hash_table_A)
        while headB != None:
            if headB in hash_table_A:
                return headB
            headB = headB.next
        return None
            #print(Blank)''

"""
Better Method:
""""
class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """

        if headA is None or headB is None:
            return None

        pa = headA # 2 pointers
        pb = headB

        while pa is not pb:
            print(pa)
            # if either pointer hits the end, switch head and continue the second traversal, 
            # if not hit the end, just move on to next
            pa = headB if pa is None else pa.next
            pb = headA if pb is None else pb.next
            
        return pa
            #print(Blank)
            
"""
26. Sort numbers in list: cheat method - count first then overwrite
""""
class Solution(object):
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        my_dict={}
        for ii in range(3):
            my_dict[ii] = 0
        
        for ii in nums:
            my_dict[ii] += 1 
            
        x=0
        for ii in range(3):
            nums[x:x+my_dict[ii]]=[ii]*my_dict[ii]
            x=x+my_dict[ii]

"""
Better Method:
""""
        Counter=0
        n=len(nums)
        Object=0
        s=0
        e=n
        while Counter!=n:
            if nums[Object]==0:
                s+=1
                Object+=1
                Counter+=1
            elif nums[Object]==2:
                nums[s:e]=nums[s:e][::-1]
                e-=1
                nums[s:e]=nums[s:e][::-1]
                Counter+=1
            elif nums[Object]==1:
                nums[s:e]=nums[s:e][::-1]
                e-=1
                nums[s:e]=nums[s:e][::-1]
                e+=1
                Counter+=1
            
            #print(Blank)
            
            
"""
26. Search for beginning and end of range of numbers
""""
class Solution(object):
    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
 
        n=len(nums)
        Lower=0
        Upper=n
        if n==0:
            return [-1,-1]
        Beginning_Point=-1
        Search_Prev=-1
        Search=n//2
        print(Search)
        while Search!=Search_Prev:
            if nums[Search]==target:
                break
            elif nums[Search]>target:
                Search_Prev=Search
                Upper=Search
                Search=Lower+(Upper-Lower)//2
            else:
                Search_Prev=Search
                Lower=Search
                Search=Lower+(Upper-Lower)//2
        if Search==Search_Prev:
            return [-1,-1]
        SearchLow=Search
        SearchHigh=Search
        while nums[SearchLow]==target:
            SearchLow-=1
            if SearchLow<0:
                break
        while nums[SearchHigh]==target:
            SearchHigh+=1
            if SearchHigh>n-1:
                break
        return [SearchLow+1, SearchHigh-1]
                
        
        
"""
Better Method:
""""
def searchRange(self, nums, target):
    def binarySearchLeft(A, x):
        left, right = 0, len(A) - 1
        while left <= right:
            mid = (left + right) / 2
            if x > A[mid]: left = mid + 1
            else: right = mid - 1
        return left

    def binarySearchRight(A, x):
        left, right = 0, len(A) - 1
        while left <= right:
            mid = (left + right) / 2
            if x >= A[mid]: left = mid + 1
            else: right = mid - 1
        return right
        
    left, right = binarySearchLeft(nums, target), binarySearchRight(nums, target)
    return (left, right) if left <= right else [-1, -1]
            
"""
27. Merge intervals
"""
class Solution(object):
    def merge(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: List[List[int]]
        """
       
        intervals.sort(key=lambda x: x[0])   #Sort elements of interval
        Res=[]
        n=len(intervals)
        if n==0:
            return []
        Res.append(intervals[0])
        Counter=1
        for ii in range(1,n):
            if intervals[ii][0]<=Res[ii-Counter][1]:
                Res[ii-Counter][1]=max(Res[ii-Counter][1],intervals[ii][1])
                Counter+=1
            else:
                Res.append(intervals[ii])
        return Res         
    
    
"""
28. Find element in rotated array
""" 

class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
    def search(self, nums, target):
        if not nums:
            return -1

        low, high = 0, len(nums) - 1

        while low <= high:
            mid = (low + high) / 2
            print(mid)
            if target == nums[mid]:
                return mid

            if nums[low] <= nums[mid]:
                if nums[low] <= target <= nums[mid]:   #
                    high = mid - 1
                else:
                    low = mid + 1
            else:
                if nums[mid] <= target <= nums[high]:
                    low = mid + 1
                else:
                    high = mid - 1

        return -1 
    
    
"""
29. Search 2 Matrix:
"""
class Solution(object):
    def searchMatrix(self, matrix, target):
        if not matrix:
            return False
        row,col,width=len(matrix)-1,0,len(matrix[0])
        while row>=0 and col<width:
            if matrix[row][col]==target:
                return True
            elif matrix[row][col]>target:
                row=row-1
            else:
                col=col+1
        return False
    
"""
30. Find number of unique paths on checker board
"""
class Solution(object):
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        Counter=0
        if m==0 or n==0:
            return 0
        if m==1 or n==1:
            return 1
        A=[1,1]
        Counter=2
        while Counter!=m+n-1:
            Blank=[1]
            for ii in range(Counter-1):
                Blank.append(A[ii]+A[ii+1])
            Blank.append(1)
            A=Blank
            Counter+=1
        B=min(m,n)    
        return A[-B]
    
"""
Factorial solution
"""
class Solution(object):
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        return math.factorial(m+n-2)/math.factorial(m-1)/math.factorial(n-1)
    
"""
DP solution
"""
class Solution(object):
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        dp = [[1]*m for _ in range(n)]
        
        for i in range(1,n):
            for j in range(1,m):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
                
        return dp[-1][-1]
    
    
"""
31. Coins into amount alogorithm:
""""
class Solution(object):
    def coinChange(self, coins, amount):
        dp = [0] + [float('inf')] * amount
        
        for coin in coins:
            for i in range(coin, amount+1):
                dp[i] = min(dp[i], dp[i-coin]+1)   #Builds table which it can then index and check
        
        return dp[-1] if dp[-1] != float('inf') else -1
    
"""
32. Find longest subsequence
"""
class Solution(object):
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n=len(nums)
        if n==0:
            return 0
        Blank=[1]*n
        Blank2=[1]*n
        for ii in range(n):
            Blank2=[1]*n
            for jj in range(ii):
                if nums[ii]>nums[ii-jj-1]:
                    Blank2[ii-jj]=Blank[ii-jj-1]+1
            Blank[ii]=max(Blank2)
        return max(Blank)



"""
Better solutions
"""
# O(n*m) solution. m is the sub[]'s length
def lengthOfLIS(self, nums):
        sub = []
        for val in nums:
            pos , sub_len = 0, len(sub)
            while(pos <= sub_len):    # update the element to the correct position of the sub.
                if pos == sub_len:
                    sub.append(val)
                    break
                elif val <= sub[pos]:
                    sub[pos] = val
                    break
                else:
                    pos += 1
        
        return len(sub)

# O(nlogn) solution with binary search
def lengthOfLIS(self, nums):

        def binarySearch(sub, val):
            lo, hi = 0, len(sub)-1
            while(lo <= hi):
                mid = lo + (hi - lo)//2
                if sub[mid] < val:
                    lo = mid + 1
                elif val < sub[mid]:
                    hi = mid - 1
                else:
                    return mid
            return lo
        
        sub = []
        for val in nums:
            pos = binarySearch(sub, val)
            if pos == len(sub):
                sub.append(val)
            else:
                sub[pos] = val
        return len(sub)
    
"""
33. Can you get to end of list by jumps of length
"""
class Solution(object):
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool     
        """
        n=len(nums)
        if n==0:
            return False
        if n==1:
            return True
        Counter=1
        for ii in range(n-1):
            if Counter>1:
                if nums[n-ii-2]>=Counter:
                    Counter=1
                else:
                    Counter+=1
            elif nums[n-ii-2]==0:
                Counter+=1
            
        if Counter==1:
            return True
        else:
            return False
        
import math
def f(n):
    Const=5**.5
    phi=.5+Const/2
    Small=phi**int(math.log(Const*n,phi))/Const
    Big=[Small,phi*Small]
    Res=round(Big[2*n>sum(Big)])
    fib_index = 2.078087 * math.log(Res) + 1.672276
    return Res,round(fib_index) 

def solution(S):
    # write your code in Python 3.6
    List=["A","B","C","D","E","F","1","0"]
    FullSet=set(List)
    Word=hex(S)
    Word=Word[2:]
    n=len(Word)
    Res=""
    for ii in range(n):
        if Word[ii] not in FullSet:
            return "ERROR"
        elif Word[ii]=="1":
            Res=Res+"I"
        elif Word[ii]=="0":
            Res=Res+"O"
        else:
            Res=Res+Word[ii]
        
    return Res
