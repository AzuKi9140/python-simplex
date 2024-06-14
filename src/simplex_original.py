import numpy as np
from scipy.linalg import lu_factor, lu_solve

from .settings import time_dec

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

def check_degeneracy(x_basis):
    return np.where(x_basis == 0)[0]

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
        # 基本解の計算
        x = np.zeros(n + m)
        x[basis] = np.linalg.solve(Ai[:, basis], b)

        # 双対変数の計算
        y = np.linalg.solve(Ai[:, basis].T, c0[basis])

        # 現在の目的関数の係数を計算
        rc = c0[nonbasis] - y @ Ai[:, nonbasis]

        # LU分解
        LU, piv = lu_factor(Ai[:, basis])

        # 最適性のチェック（双対可能性）
        if np.all(rc <= error):
            print("--optimal solution found---------------------------------------------------")
            print_simplex_detail(iter=iter, basis=basis, nonbasis=nonbasis, rc=rc, x_b=x[basis])
            print("obj.val. = {}".format(c0[basis] @ x[basis]))
            print("x = {}".format(x))
            print("---------------------------------------------------------------------------")
            break
        
        degenerate_indices = check_degeneracy(x[basis])

        if degenerate_indices.size > 0:
            # 入る変数の決定(最小添え字)
            s = -1
            candidates_nonbasis_index = []
            for i in range(len(rc)):
                if(rc[i] >= error):
                    candidates_nonbasis_index.append(i)
            s = min(candidates_nonbasis_index)
            if s == -1:
                print("no entering variable found, optimal solution found")
                break

            #d = np.linalg.solve(Ai[:, basis], Ai[:, nonbasis[s]])
            d = lu_solve((LU, piv), Ai[:, nonbasis[s]])

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

            # 最小の比率を持つ変数のインデックスを取得
            min_ratio = min(ratio)
            candidates_basis_index = [i for i, val in enumerate(ratio) if val == min_ratio]

            # 最小添え字を持つインデックスを選択
            r = min(candidates_basis_index)

            print_simplex_detail(iter=iter, basis=basis, nonbasis=nonbasis, rc=rc, d=d, s=s, r=r, x_b=x[basis])

            nonbasis[s], basis[r] = basis[r], nonbasis[s]
        
        else:
            # 入る変数の決定(最大改善、最急降下)
            s_max_improvement = -1
            d_max_improvement = None
            max_improvement = -np.inf
            s_steepest_edge = -1
            d_steepest_edge = None
            max_ratio = -np.inf
            for i in range(len(rc)):
                if rc[i] > error:
                    tmp = lu_solve((LU, piv), Ai[:, nonbasis[i]])
                    improvement = rc[i] / np.max(tmp)
                    norm = np.sqrt(1 + np.sum(tmp**2))
                    ratio = rc[i] / norm
                    if improvement > max_improvement:
                        max_improvement = improvement
                        s_max_improvement = i
                        d_max_improvement = tmp

                    if ratio > max_ratio:
                        max_ratio = ratio
                        s_steepest_edge = i
                        d_steepest_edge = tmp

            if s_max_improvement == -1:
                print("no entering variable found by maximum improvement, optimal solution found")
                break

            if s_steepest_edge == -1:
                print("no entering variable found by steepest edge, optimal solution found")
                break            

            # 非有界性のチェック
            if np.all(d_max_improvement <= error) or np.all(d_steepest_edge <= error):
                print("problem is unbounded")
                break

            # 最大改善規則で選んだ変数の比率を計算
            ratio_max_improvement = []
            for i in range(len(d_max_improvement)):
                if d_max_improvement[i] > error:
                    ratio_max_improvement.append(x[basis][i] / d_max_improvement[i])
                else:
                    ratio_max_improvement.append(np.inf)

            # 最急辺規則で選んだ変数の比率を計算
            ratio_steepest_edge = []
            for i in range(len(d_steepest_edge)):
                if d_steepest_edge[i] > error:
                    ratio_steepest_edge.append(x[basis][i] / d_steepest_edge[i])
                else:
                    ratio_steepest_edge.append(np.inf)

            # 出る変数の決定
            r_max_improvement = np.argmin(ratio_max_improvement)
            r_steepest_edge = np.argmin(ratio_steepest_edge)

            # 同じ変数が選ばれた場合、一回の入れ替えのみを行う
            if s_max_improvement == s_steepest_edge:
                r = min(r_max_improvement, r_steepest_edge)
                nonbasis[s_max_improvement], basis[r] = basis[r], nonbasis[s_max_improvement]
            else:
                nonbasis[s_max_improvement], basis[r_max_improvement] = basis[r_max_improvement], nonbasis[s_max_improvement]
                nonbasis[s_steepest_edge], basis[r_steepest_edge] = basis[r_steepest_edge], nonbasis[s_steepest_edge]

        if iter % 50 == 0:
            print("iter: {}".format(iter))
            print("current obj.val. = {}".format(x[basis] @ c0[basis]))
        iter += 1

if __name__ == "__main__":
    #A = np.array([[2, 0, 0], [1, 0, 2], [0, 3, 1]])
    #b = np.array([4, 8, 6])
    #c = np.array([3, 4, 2])
    #lp_simplex(A, b, c)

    #A = np.array([[1, 12, -2, -12], [0.25, 1, -0.25, -2], [1, -4, 0, -8]])
    #b = np.array([0, 0, 1])
    #c = np.array([1, -4, 0, -8])
    #lp_simplex(A, b, c)

    n = 100
    m = 100
    A = np.random.rand(m, n) - np.random.rand(m, n)
    b = np.ones(m)
    c = np.ones(n)
    lp_simplex(A, b, c)
