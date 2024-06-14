import numpy as np
from settings import time_dec

error = 1.0e-10  # 許容誤差

def print_simplex_detail(
        iter=None, 
        m=None, 
        n=None, 
        basis=None, 
        nonbasis=None,
        x_b=None,
        y=None,
        rc=None,
        s=None,
        d=None,
        r=None,
        ):
    if iter is not None:
        print(f"iter = {iter}")
    if m is not None and n is not None:
        print(f"(m, n) = {(m, n)}")
    if basis is not None:
        print(f"basis = {basis}")
    if nonbasis is not None:
        print(f"nonbasis = {nonbasis}")
    if y is not None:
        print(f"y = {y}")
    if rc is not None:
        print(f"rc = {rc}")
    if s is not None:
        print(f"s = {s}")
    if d is not None:
        print(f"d = {d}")
    if x_b is not None:
        print(f"x_basis = {x_b}")
    if r is not None:
        print(f"r = {r}")

@time_dec
def lp_simplex(A, b, c):
    (m, n) = A.shape  # m:制約条件数, n:変数数
    print("m = {}, n = {}".format(m, n))

    # 初期化
    Ai = np.hstack((A, np.identity(m)))
    c0 = np.r_[c, np.zeros(m)]

    basis = [n + i for i in range(m)]
    nonbasis = [j for j in range(n)]

    iter = 0

    while True:
    # 巡回の確認の際に使用
    #while iter < 10:
        # 基本解の計算
        x = np.zeros(n + m)
        x[basis] = np.linalg.solve(Ai[:, basis], b)

        # 双対変数の計算
        y = np.linalg.solve(Ai[:, basis].T, c0[basis])

        # 現在の目的関数の係数を計算
        rc = c0[nonbasis] - y @ Ai[:, nonbasis]

        # 最適性のチェック（双対可能性）
        if np.all(rc <= error):
            print("--optimal solution found---------------------------------------------------")
            print_simplex_detail(iter=iter)
            print("obj.val. = {}".format(c0[basis] @ x[basis]))
            print("x = {}".format(x))
            print("---------------------------------------------------------------------------")
            break
        
        # 入る変数の決定(最大係数)
        s = np.argmax(rc)
        d = np.linalg.solve(
            Ai[:, basis], Ai[:, nonbasis[s]]
        )

        # 非有界性のチェック
        if np.all(d <= error):
            print("problem is unbounded")
            break
            
        # 出る変数の決定
        ratio = []
        for i in range(len(d)):
            if d[i] > error:
                ratio.append(x[basis][i] / d[i])
            else:
                ratio.append(np.inf)

        r = np.argmin(ratio)

        nonbasis[s], basis[r] = basis[r], nonbasis[s]

        if iter % 50 == 0:
                print("iter: {}".format(iter))
                print("current obj.val. = {}".format(x[basis] @ c0[basis]))

        # 巡回の確認の際に使用
        #print_simplex_detail(iter=iter, basis=basis, nonbasis=nonbasis, rc=rc, d=d, s=s, r=r, x_b=x[basis])
        iter += 1

if __name__ == "__main__":
    # 例題
    A = np.array([[2, 0, 0], [1, 0, 2], [0, 3, 1]])
    b = np.array([4, 8, 6])
    c = np.array([3, 4, 2])
    lp_simplex(A, b, c)

    # 巡回する例
    #A = np.array([[1, 12, -2, -12], [0.25, 1, -0.25, -2], [1, -4, 0, -8]])
    #b = np.array([0, 0, 1])
    #c = np.array([1, -4, 0, -8])
    #lp_simplex(A, b, c)
