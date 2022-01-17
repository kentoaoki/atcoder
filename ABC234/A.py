def a(x):
    y = x**2 + 2*x + 3
    return y

test = int(input())
ans = a(a(a(test) + test) + a(a(test)))
print(ans)
