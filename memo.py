import math
from math import gcd,pi,sqrt
INF = float("inf")
MOD = 10**9 + 7
import sys
input = sys.stdin.readline
sys.setrecursionlimit(10**6)
import itertools
import bisect
import re
from collections import Counter,deque,defaultdict
def iinput(): return int(input())
def imap(): return map(int, input().split())
def ilist(): return list(imap())
def irow(N): return [iinput() for i in range(N)]
def sinput(): return input().rstrip()
def smap(): return sinput().split()
def slist(): return list(smap())
def srow(N): return [sinput() for i in range(N)]

def main():
    import numpy as np
    n,c = imap()
    l = np.array([ilist() for _ in range(n)])
    right = np.cumsum(l[:,1]) - l[:,0]
    right_max = np.maximum.accumulate(right)
    left_dist = c - l[:,0][::-1]
    left = np.cumsum(l[:,1][::-1]) - left_dist
    left_max = np.maximum.accumulate(left)
    # print(left_dist)
    right_to_left = right[:-1] - l[:,0][:-1] + left_max[:-1][::-1]
    left_to_right = left[:-1] - left_dist[:-1] + right_max[:-1][::-1]
    ans = max(np.max(right), np.max(left))
    if len(right_to_left):
        ans = max(ans, np.max(right_to_left))
        ans = max(ans, np.max(left_to_right))
    print(ans if ans > 0 else 0)
    # print(right,left,right_to_left,left_to_right)


if __name__=="__main__":
    main()