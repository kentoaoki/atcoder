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

N,M,s = map(int, input().split())
G = [[] for i in range(N)]
for i in range(M):
    a,b,w = map(int, input().split())
    G[a].append([b,w])
ans = dijkstra(G,s)
print(*map(lambda x: x if x != float('inf') else 'INF', ans), sep='\n')