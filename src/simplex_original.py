import numpy as np

error = 1.0e-10  # 許容誤差

def check_degeneracy(x_basis):
    return np.where(x_basis == 0)[0]

def find_min_index_basis(basis, degenerate_indices, d):
    relevant_basis = [basis[i] for i in degenerate_indices if d[i] >= -error]
    #if not relevant_basis:
    #    return None
    min_value = min(relevant_basis)
    return basis.index(min_value)

def find_min_index_nonbasis(nonbasis, rc):
    candidates = [nonbasis[i] for i in range(len(nonbasis)) if rc[i] > error]
    #if not candidates:
    #    return None
    min_value = min(candidates)
    return nonbasis.index(min_value)

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

        # 最適性のチェック（双対可能性）
        if np.all(rc <= error):
            print("number of iterations = {}".format(iter))
            print("optimal solution found")
            print("obj.val. = {}".format(c0[basis] @ x[basis]))
            print(x[0:n])
            break
        
        degenerate_indices = check_degeneracy(x[basis])

        if degenerate_indices.size > 0:
            degeneration_flag = True
            min_index_nonbasis = find_min_index_nonbasis(nonbasis, c0[nonbasis])
            #if min_index_nonbasis is None:
            #    break
            d_min_index = np.linalg.solve(Ai[:, basis], Ai[:, nonbasis[min_index_nonbasis]])

            # 非有界性のチェック
            if np.all(d_min_index <= error):
                print("problem is unbounded")
                break

            min_index_basis = find_min_index_basis(basis, degenerate_indices, d_min_index)
            #if min_index_basis is None:
            #    break

            nonbasis[min_index_nonbasis], basis[min_index_basis] = basis[min_index_basis], nonbasis[min_index_nonbasis]
        else:
            # 入る変数の決定(最大改善)
            s_max_improvement = -1
            d_max_improvement = None
            max_improvement = -np.inf
            for i in range(len(rc)):
                if rc[i] > error:
                    tmp = np.linalg.solve(Ai[:, basis], Ai[:, nonbasis[i]])
                    improvement = rc[i] / np.max(tmp)
                    if improvement > max_improvement:
                        max_improvement = improvement
                        s_max_improvement = i
                        d_max_improvement = tmp

            if s_max_improvement == -1:
                print("no entering variable found by maximum improvement, optimal solution found")
                break

            # 入る変数の決定(最急辺)
            s_steepest_edge = -1
            d_steepest_edge = None
            max_ratio = -np.inf
            for i in range(len(rc)):
                if rc[i] > error:
                    tmp = np.linalg.solve(Ai[:, basis], Ai[:, nonbasis[i]])
                    norm = np.sqrt(1 + np.sum(tmp**2))
                    ratio = rc[i] / norm
                    if ratio > max_ratio:
                        max_ratio = ratio
                        s_steepest_edge = i
                        d_steepest_edge = tmp

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
    A = np.array([[2, 0, 0], [1, 0, 2], [0, 3, 1]])
    b = np.array([4, 8, 6])
    c = np.array([3, 4, 2])

    lp_simplex(A, b, c)

    A = np.array([[1, 12, -2, -12], [0.25, 1, -0.25, -2], [1, -4, 0, -8]])
    b = np.array([0, 0, 1])
    c = np.array([1, -4, 0, -8])

    lp_simplex(A, b, c)
