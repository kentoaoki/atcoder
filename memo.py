def main():
    def Base_10_to_n(X, n):
        if (int((X-1)/n)):
            return Base_10_to_n(int(X/n), n)+" "+str(X%n)
        return str((X-1)%n)

    import string
    n = int(input())
    a = Base_10_to_n(n,26)
    a = list(a.split())
    a = [int(i) for i in a][::-1]
    # print(a)
    ans = ""
    st = string.ascii_lowercase
    for i in range(len(a)):
        if a[i] != 0:
            ans += st[a[i]-1]
        else:
            ans += "z"

    print(ans[::-1])


if __name__=="__main__":
    main()