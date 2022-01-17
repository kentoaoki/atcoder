def list_inp(): return list(map(int, input().split()))

D = int(input())
C = list_inp()
S = [list_inp() for i in range(D)]
t = [int(input()) for i in range(D)]

point = 0
last_edit = [0]*26

for i in range(D):
    last_edit[t[i] - 1] = i + 1
    posi = S[i][t[i] - 1]
    nega = 0
    for j in range(26):
        nega += C[j]*(i + 1 - last_edit[j])
    point = point + posi - nega
    # print(posi, nega)
    print(point)