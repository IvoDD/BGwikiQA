import numpy as np

def hungarian(matr):
    inf = 1e9
    n, m = matr.shape
    u = np.zeros(n)
    v = np.zeros(m)
    match = np.ones(n+m, dtype=np.int32)*-1
    best = np.zeros(m)
    bestp = np.zeros(m, dtype=np.int32)
    used = np.zeros(n, dtype=np.int32)
    for row in range(n):
        used[:row+1] = 0
        used[row] = 1
        best[:] = matr[row]-u[row]-v
        bestp[:] = row
        while True:
            delta = inf
            nxt = 0
            for i in range(m):
                if delta > best[i]:
                    delta = best[i]
                    nxt = i
            for i in range(row+1):
                if used[i]==1:
                    u[i]+=delta
            for i in range(m):
                if best[i]==inf:
                    v[i]-=delta
                else:
                    best[i]-=delta
            best[nxt] = inf
            if match[n+nxt]==-1:
                while nxt!=-1:
                    prev = match[bestp[nxt]]
                    match[n+nxt] = bestp[nxt]
                    match[bestp[nxt]] = nxt
                    nxt = prev
                break
            else:
                vr = match[n+nxt]
                used[vr]=1
                for i in range(m):
                    if best[i]!=inf and best[i]>matr[vr, i]-u[vr]-v[i]:
                        best[i] = matr[vr, i]-u[vr]-v[i]
                        bestp[i] = vr

    ans = 0
    for i in range(n):
        # print(i, match[i])
        ans += matr[i, match[i]]
    return ans