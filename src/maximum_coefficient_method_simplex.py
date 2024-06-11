import numpy as np

from src.lib.check_degeneracy import check_degeneracy, find_min_index_basis

error = 1.0e-10  # 許容誤差


def lp_simplex(A, b, c):

    (m, n) = A.shape  # m:制約条件数, n:変数数
    print("m = {}, n = {}".format(m, n))

    # 初期化
    Ai = np.hstack((A, np.identity(m)))
    c0 = np.r_[c, np.zeros(m)]

    basis = [n + i for i in range(m)]
    nonbasis = [j for j in range(n)]

    iter = 0

    degeneration_flag = False

    while True:
        # 基本解の計算
        if iter > 5:
            break
        print("iter = ", iter)
        x = np.zeros(n + m)
        x[basis] = np.linalg.solve(Ai[:, basis], b)
        # 双対変数の計算
        y = np.linalg.solve(Ai[:, basis].T, c0[basis])
        print("Ai[:, basis] = ", Ai[:, basis])
        print("Ai[:, nonbasis] = ", Ai[:, nonbasis])

        # 現在の目的関数の係数を計算
        rc = c0[nonbasis] - y @ Ai[:, nonbasis]

        degenerate_indices = check_degeneracy(x[basis])

        if degeneration_flag:
            if len(degenerate_indices) > 0:
                min_index_basis = find_min_index_basis(basis, degenerate_indices)
            else:
                min_index_basis = np.argmin(basis)
            non_zero_rc_index = [i for i in range(len(rc)) if rc[i] > error]
            print("rc = ", rc)
            print("non_zero_rc_index = ", non_zero_rc_index)
            non_zero_rc_nonbasis = [nonbasis[i] for i in non_zero_rc_index]

            min_index_nonbasis = np.argmin(non_zero_rc_nonbasis)
            print("--------------------------------")
            print("nonbasis = ", nonbasis)
            print("basis = ", basis)
            print("x[basis] = ", x[basis])
            print("check_degeneracy(x[basis]) = ", degenerate_indices)
            print("min_index_nonbasis = ", min_index_nonbasis)
            print("min_index_basis = ", min_index_basis)
            print("--------------------------------")

        if degeneration_flag is False and degenerate_indices.size > 0:
            print("--------------------------------")
            print("nonbasis = ", nonbasis)
            print("basis = ", basis)
            print("x[basis] = ", x[basis])
            print("check_degeneracy(x[basis]) = ", degenerate_indices)
            print("degeneracy detected")

            non_zero_rc = [i for i in range(len(rc)) if rc[i] > error]
            non_zero_rc_nonbasis = [nonbasis[i] for i in non_zero_rc]

            min_index_nonbasis = np.argmin(non_zero_rc_nonbasis)
            print("min_index_nonbasis = ", min_index_nonbasis)
            min_index_basis = find_min_index_basis(basis, degenerate_indices)
            print("min_index_basis = ", min_index_basis)
            print("--------------------------------")
            degeneration_flag = True
        bb = np.linalg.solve(Ai[:, basis], b)  # 非効率な計算

        # 最適性のチェック（双対可能性）
        if np.all(rc <= error):
            print("rc = ", rc)
            print("number of iterations = {}".format(iter))
            print("optimal solution found")
            print("obj.val. = {}".format(c0[basis] @ x[basis]))
            print(x[0:n])
            break
        elif degeneration_flag is False:
            # 入る変数の決定
            s = np.argmax(rc)
        else:
            s = min_index_nonbasis

        d = np.linalg.solve(Ai[:, basis], Ai[:, nonbasis[s]])  # sに対応する列の取得
        print("d = ", d)

        # 非有界性のチェック
        if np.all(d <= error):
            print("problem is unbounded")
            break
        else:
            if iter % 50 == 0 and iter > 0:
                print("iter: {}".format(iter))
                print("current obj.val. = {}".format(x[basis] @ c0[basis]))

            ratio = []
            for i in range(len(d)):
                if d[i] > error:
                    ratio.append(bb[i] / d[i])
                else:
                    ratio.append(np.inf)

            # 出る変数の決定
            r = np.argmin(ratio)

            if degeneration_flag is False:
                nonbasis[s], basis[r] = basis[r], nonbasis[s]
            else:
                nonbasis[min_index_nonbasis], basis[min_index_basis] = (
                    basis[min_index_basis],
                    nonbasis[min_index_nonbasis],
                )

            iter += 1


if __name__ == "__main__":
    # A = np.array([[2, 0, 0], [1, 0, 2], [0, 3, 1]])
    # b = np.array([4, 8, 6])
    # c = np.array([3, 4, 2])
    A = np.array([[1, 12, -2, -12], [0.25, 1, -0.25, -2], [1, -4, 0, -8]])
    b = np.array([0, 0, 1])
    c = np.array([1, -4, 0, -8])

    lp_simplex(A, b, c)
