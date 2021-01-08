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
    def fact_prepare(n, MOD):
        """階乗,逆元を何度も使用する時用の前処理

        Args:
            n: 階乗を求める上限
            MOD: 方とする値

        Returns:
            factorials: 階乗の配列
            invs: 逆元の配列
        
        Examples:
            nCr = factorials[n] * invs[n-r] * invs[r]
        """
        f = 1
        factorials = [1]
        for m in range(1, n + 1):
            f *= m
            f %= MOD
            factorials.append(f)
        inv = pow(f, MOD - 2, MOD)
        invs = [1] * (n + 1)
        invs[n] = inv
        for m in range(n, 1, -1):
            inv *= m
            inv %= MOD
            invs[m - 1] = inv
        return factorials, invs

    n, m, k = map(int, input().split())
    MOD = 998244353
    facts, invs = prepare(n, MOD)
    
    ans = 0
    for s in range(k + 1):
        p = n - s
        ans = (ans + m * pow(m - 1, p - 1, MOD) * facts[n - 1] * invs[s] * invs[n - s - 1]) % MOD
    print(ans)


if __name__=="__main__":
    main()