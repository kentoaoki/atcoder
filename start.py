# AtCoder用ライブラリ
"""
初期操作
"""

# 立ち上げ時のコピペ内容

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

if __name__=="__main__":
    main()


"""
組み合わせ
"""

## 二項係数の前処理(nCr)

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

## nCrのmod計算

def fur(n,r):
    p,q = 1, 1
    for i in range(r): # p=(n-r+1)!,q=r!を作る
        p = p*(n-i)%MOD
        q = q*(i+1)%MOD
    return (p * pow(q,MOD-2,MOD)) % MOD # qの逆元を作ってp/qを処理する

## 普通のnCr

def comb(n, r):
    if n-r>=0:
        return math.factorial(n) // (math.factorial(n - r) * math.factorial(r))
    else:
        return 0

"""
整数問題
"""

## 単純な素数判定
def isPrime(n):
    if n == 2:
        return True
    if n%2 == 0 or n == 1:
        return False
    m = math.floor(math.sqrt(n))+1
    for p in range(3,m,2):
        if n%p == 0:
            return False
    return True

## エラトステネスの篩（素数列挙）
def primes(n):
    if n <= 0:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0], is_prime[1] = False, False
    for i in range(2, int(n**0.5) + 1):
        if not is_prime[i]:
            continue
        for j in range(i * 2, n + 1, i):
            is_prime[j] = False
    return [i for i in range(n + 1) if is_prime[i]]

## 最大公約数
gcd(a,b)

## 最小公倍数
a*b//gcd(a,b)

## 素因数分解(一次元配列)
def prime_decomposition(n):
    i = 2
    table = []
    while i * i <= n: # sqrt(n)で計算が済む
        while n % i == 0:
            n //= i
            table.append(i)
        i += 1
    if n > 1:
        table.append(n)
    return table

## 素因数分解（素因数、指数の二次元配列）
def prime_decomposition_2dim(n):
    arr = []
    temp = n
    for i in range(2, int(-(-n**0.5//1))+1):
        if temp%i==0:
            cnt=0
            while temp%i==0:
                cnt+=1
                temp //= i
            arr.append([i, cnt])
    if temp!=1:
        arr.append([temp, 1])
    if arr==[]:
        arr.append([n, 1])
    return arr


## 約数列挙
def make_divisors(n):
    lower_divisors , upper_divisors = [], []
    i = 1
    while i*i <= n:
        if n % i == 0:
            lower_divisors.append(i)
            if i != n // i:
                upper_divisors.append(n//i)
        i += 1
    return lower_divisors + upper_divisors[::-1]

## 公約数列挙
def common_divisors(n, m):
    n = gcd(n,m)
    lower_divisors , upper_divisors = [], []
    i = 1
    while i*i <= n:
        if n % i == 0:
            lower_divisors.append(i)
            if i != n // i:
                upper_divisors.append(n//i)
        i += 1
    return lower_divisors + upper_divisors[::-1]

## 10進数からn進数への変換
def Base_10_to_n(X, n):
    if (int(X/n)):
        return Base_10_to_n(int(X/n), n)+str(X%n)
    return str(X%n)

## n進数から10進数への変換
def Base_n_to_10(X,n):
    X = str(X)
    out = 0
    for i in range(1,len(str(X))+1):
        out += int(X[-i])*(n**(i-1))
    return out#int out

"""
2次元配列操作
"""

## 転置行列

l_2 = [list(x) for x in zip(*l)]

## 8方向への移動
dx=[0,1,0,-1,1,1,-1,-1]
dy=[1,0,-1,0,1,-1,1,-1]

"""
幾何
"""

### 2点を通る直線
def line(x0,y0,x1,y1): # (x0,y0),(x1,y1)を通る直線ax+by+c=0を求める
        x = (d-b)
        y = -c+a
        z = b*(c-a) - a*(d-b)
        return x,y,z
    
### 点と直線の距離
def dist(x0,y0,a,b,c): # (x0,y0)から直線ax+by+c=0までの距離を求める
    up = abs(a*x0 + b*y0 + c)
    down = math.sqrt(a**2 + b**2)
    return up/down

"""
探索
"""

### DFS(デック)

def search(G,s):
    """
    グラフGに対し、頂点sを始点とする探索（デック）
    """
    N = len(G)
    todo = deque() # BFSならキュー、DFSならスタックとして使用
    seen = [False]*N # 発見チェック（seen:True & not in todo => 訪問済み)
    
    # 初期状態
    seen[s] = True
    todo.append(s)
    
    # todoが空になるまで探索を行う
    while todo:
        v = todo.popleft() # DFSならpop()
        for x in G[v]:
            if seen[x]: # 発見済み
                continue
            seen[x] = True
            todo.append(x)

### DFS(再帰)

def dfs(G,v):
    """
    頂点vから辿れる頂点を全て訪問する（再帰）
    """
    # 訪問時の処理
    seen[v] = True
    
    # 再帰的に探索
    for next_v in G[v]:
        if seen[next_v]:
            continue
        dfs(G, next_v) # 再帰的に探索
    
    # 帰りがけの処理
        
N,M = map(int,input().split())
G = [[] for i in range(N)] # 隣接リスト
for i in range(M):
    a,b = map(int, input().split())
    a -= 1
    b -= 1
    G[a].append(b)
print(G)

seen = [False]*N
for v in range(N): # 全頂点から探索
    if seen[v]:
        continue
    dfs(G,v)

### BFS

def BFS(G, s):
    """グラフGに関して,BFSで頂点sからの最短路長を求める

    Args:
        G: グラフ(隣接リスト)
        s: 探索開始頂点
    
    Returns:
        dist: 探索開始頂点からの距離
    """
    # 初期設定
    N = len(G)
    dist = [-1]*N
    que = deque()
    dist[s] = 0
    que.append(s)
    
    # 探索
    while que:
        v = que.popleft()
        for x in G[v]:
            if dist[x] != -1:
                continue
            dist[x] = dist[v] + 1
            que.append(x)
    return dist

N,M = map(int, input().split())
G = [[] for i in range(N)]
for i in range(M):
    a,b = map(int, input().split())
    a -= 1
    b -= 1
    G[a].append(b)

dist = BFS(G,0)

### 二分探索

def is_ok(arg):
    # 条件を満たすかどうか？問題ごとに定義
    pass


def meguru_bisect(ng, ok):
    '''
    初期値のng,okを受け取り,is_okを満たす最小(最大)のokを返す
    まずis_okを定義すべし
    ng ok は  とり得る最小の値-1 とり得る最大の値+1
    最大最小が逆の場合はよしなにひっくり返す
    '''
    while (abs(ok - ng) > 1):
        mid = (ok + ng) // 2
        if is_ok(mid):
            ok = mid
        else:
            ng = mid
    return ok

### ダイクストラ法(線形探索)

def dijkstra(G,s):
    """ダイクストラ法により、辺の重みが非負グラフの最短経路を求める O(|V|^2)
    
    Args:
        G: 重み付きグラフ((頂点,重み)の隣接リスト形式)
        s: 探索開始頂点
    
    Returns:
        dist: s(探索開始頂点)からの最短距離.INFの場合は到達しない
        
    """
    INF = float('inf')
    used = [False]*N
    dist = [INF]*N
    dist[s] = 0
    for iter in range(N):
        # 未使用かつdist値最小の頂点を探す
        min_dist = INF
        min_v = -1
        for v in range(N):
            if not used[v] and dist[v] < min_dist:
                min_dist = dist[v]
                min_v = v
        if min_v == -1:
            break
        for v,w in G[min_v]:
            dist[v] = min(dist[v], dist[min_v] + w)
        used[min_v] = True
    return dist

N,M,s = map(int, input().split())
G = [[] for i in range(N)]
for i in range(M):
    a,b,w = map(int, input().split())
    G[a].append([b,w])
ans = dijkstra(G,s)
ans

### ダイクストラ法(heapq)

def dijkstra(G,s):
    """ヒープ使用のダイクストラ法により,辺の重みが非負グラフの最短経路を求める O(|E|log|V|)
    
    Args:
        G: 重み付きグラフ((頂点,重み)の隣接リスト形式)
        s: 探索開始頂点
    
    Returns:
        dist: s(探索開始頂点)からの最短距離.INFの場合は到達しない
        
    """
    INF = float('inf')
    dist = [INF]*N
    dist[s] = 0

    # ヒープの初期化
    import heapq
    que = [[dist[s], s]]
    heapq.heapify(que)

    while que:
        d, v = heapq.heappop(que) # d最小値ペアの取り出し
        if d > dist[v]: # ゴミ処理
            continue
        for e in G[v]:
            if dist[e[0]] > dist[v] + e[1]:
                dist[e[0]] = dist[v] + e[1]
                heapq.heappush(que, [dist[e[0]], e[0]])
    return dist

### ワーシャルフロイド法

def worshall_floyd(G):
    """ワーシャルフロイド法により、グラフの全点間最短距離を返す
    
    Args:
        G: グラフ(隣接行列,距離不明辺はINFで初期化)
    
    Returns:
        G: 最短距離の記述されたグラフ(INFは到達不能,マイナスは負閉路)
    """
    # 自身への距離をゼロで初期化
    N = len(G)
    for i in range(N):
        G[i][i] = 0
    # 探索
    for k in range(N):
        for i in range(N):
            for j in range(N):
                G[i][j] = min(G[i][j], G[i][k] + G[k][j])
    return G

"""
フィールド操作
"""

def field_to_graph(F):
        """番兵つきのフィールドをグラフに変換 O(HW)

        Args:
            F: 周囲に番兵のあるフィールド(','が通路,'#'が壁を想定)

        Returns:
            G: 通路を辺と見たグラフ(隣接行列)

        Notes:
            フィールドに対して探索をかけるときに使えます
            頂点番号は左上が1,右下がHWの連番です
        """
        # 初期設定
        H = len(F) - 2
        W = len(F[0]) - 2
        G = [[INF]*(W*H) for i in range(W*H)]
        for i in range(W*H):
            G[i][i] = 0
        dy = [1,0,-1,0]
        dx = [0,1,0,-1]
        
        # 探索
        for h in range(1, H + 1):
            for w in range(1, W + 1):
                if F[h][w] == '#':
                    continue
                for i in range(4):
                    Y = h + dy[i]
                    X = w + dx[i]
                    if S[Y][X] == '#':
                        continue
                    frm = w + (h-1)*W - 1
                    to = X + (Y-1)*W - 1
                    G[frm][to] = 1
                    G[to][frm] = 1
        return G


"""
グループ管理
"""

### UnionFind

class UnionFind:
    """Union-Find法を活用し、データのグループ分けを行う
    
    Attributes:
        par: 親のインデクスをさす
        rank: グループの高さ
        size: グループのサイズ(親に集約される)
        
    Args:
        n: グループ分けする要素の個数
    """
    def __init__(self, n):
        self.par = [i for i in range(n+1)]
        self.rank = [0] * (n+1)
        self.size = [1] * (n+1) # 親に集約される

    # 検索
    def find(self, x):
        if self.par[x] == x:
            return x
        else:
            self.par[x] = self.find(self.par[x])
            return self.par[x]

    # 併合
    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)

        if x == y:
            return

        if self.rank[x] < self.rank[y]:
            self.par[x] = y
            self.size[y] += self.size[x]
            self.size[x] = 0
        else:
            self.par[y] = x
            self.size[x] += self.size[y]
            self.size[y] = 0
            if self.rank[x] == self.rank[y]:
                self.rank[x] += 1

    # 同じ集合に属するか判定
    def same(self, x, y):
        return self.find(x) == self.find(y)

    # すべての頂点に対して親を検索する
    def all_find(self):
        for n in range(len(self.par)):
            self.find(n)

### セグメントツリー

class SegmentTree:
    def __init__(self, size, op, e):
        """初期化

        Args:
            size: 配列サイズ(2の累乗でなくても良い)
            op: 集計したい処理
            e: 要素の初期値(単位元)
        
        Note:
            0-indexで生成
        """
        self._op = op
        self._e = e
        self._size = size
        t = 1
        while t < size:
            t *= 2
        self._offset = t - 1
        self._data = [e] * (t * 2 - 1)

    def update(self, index, value):
        op = self._op
        data = self._data
        i = self._offset + index
        data[i] = value
        while i >= 1:
            i = (i - 1) // 2
            data[i] = op(data[i * 2 + 1], data[i * 2 + 2])

    def query(self, start, stop):
        def iter_segments(data, l, r):
            while l < r:
                if l & 1 == 0:
                    yield data[l]
                if r & 1 == 0:
                    yield data[r - 1]
                l = l // 2
                r = (r - 1) // 2
        op = self._op
        it = iter_segments(self._data, start + self._offset,
                        stop + self._offset)
        result = self._e
        for v in it:
            result = op(result, v)
        return result

"""
デバッグ用テストケース生成
"""

def random_int(a,b):
    """閉区間a~bからランダムに整数を生成する

    Args:
        a,b: 区間.a,b含む。
    Returns:
        閉区間a~bのランダム整数
    """
    import random
    return random.randint(a,b)

def random_int_list(a,b,n):
    """閉区間a~bの整数n個からなるリストを生成(重複あり)
    """
    import random
    return [random.randint(a,b) for _ in range(n)]