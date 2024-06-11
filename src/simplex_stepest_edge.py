import numpy as np

error = 1.0e-10  # 許容誤差

def lp_simplex(A, b, c):
    (m, n) = A.shape  # m:制約条件数, n:変数数
    print("m = {}, n = {}".format(m, n))

    # 初期化
    Ai = np.hstack((A, np.identity(m)))
    c0 = np.r_[c, np.zeros(m)]

    basis = [n + i for i in range(m)]
    nonbasis = [j for j in range(n)]

    print("N = {}" .format(nonbasis))
    print("B = {}" .format(basis))

    iter = 0

    while True:
        # 基本解の計算
        x = np.zeros(n + m)
        x[basis] = np.linalg.solve(Ai[:, basis], b)

        # 双対変数の計算
        y = np.linalg.solve(Ai[:, basis].T, c0[basis])

        # 現在の目的関数の係数を計算
        rc = c0[nonbasis] - y @ Ai[:, nonbasis]

        # 最適性のチェック（双対可能性）
        if np.all(rc <= error):
            print("number of iterations = {}".format(iter))
            print("optimal solution found")
            print("obj.val. = {}".format(c0[basis] @ x[basis]))
            print(x[0:n])
            print(rc)
            break

        # 入る変数の決定(最急辺)
        s = -1
        d = None
        max_ratio = -np.inf
        for i in range(len(rc)):
            if rc[i] > error:
                tmp = np.linalg.solve(Ai[:, basis], Ai[:, nonbasis[i]])
                norm = np.sqrt(1 + np.sum(tmp**2))
                ratio = rc[i] / norm
                if ratio > max_ratio:
                    max_ratio = ratio
                    s = i
                    d = tmp

        if s == -1:
            print("no entering variable found, optimal solution found")
            break

        # 非有界性のチェック
        if np.all(d <= error):
            print("problem is unbounded")
            break

        if iter % 50 == 0:
            print("iter: {}".format(iter))
            print("current obj.val. = {}".format(x[basis] @ c0[basis]))

        ratio = []
        for i in range(len(d)):
            if d[i] > error:
                ratio.append(x[basis][i] / d[i])
            else:
                ratio.append(np.inf)

        # 出る変数の決定
        r = np.argmin(ratio)

        nonbasis[s], basis[r] = basis[r], nonbasis[s]

        #print("N = {}" .format(nonbasis))
        #print("B = {}" .format(basis))

        iter += 1

if __name__ == "__main__":
    A = np.array([[2, 0, 0], [1, 0, 2], [0, 3, 1]])
    b = np.array([4, 8, 6])
    c = np.array([3, 4, 2])

    lp_simplex(A, b, c)
