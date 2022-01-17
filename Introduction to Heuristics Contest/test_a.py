def list_inp(): return list(map(int, input().split()))

D = int(input())
C = list_inp()
S = [list_inp() for i in range(D)]

def check_point(last_edit, t, last_t):
    """last_tを追加した時の得点増分
    """
    point = 0

    tmp = last_edit[last_t - 1]
    last_edit[last_t - 1] = i + 1
    posi = S[i][last_t - 1]
    nega = 0
    for j in range(26):
        nega += C[j]*(i + 1 - last_edit[j])
    point = point + posi - nega

    last_edit[last_t - 1] = tmp
    return point

t = []
last_edit = [0]*26
point = 0

for i in range(D):
    contest_type = 0
    point = -10**5

    for j in range(26):
        point_test = check_point(last_edit, t, j + 1)
        if point_test > point:
            point = point_test
            contest_type = j + 1
    t = t + [contest_type]
    last_edit[contest_type - 1] = i
    print(t[-1])
