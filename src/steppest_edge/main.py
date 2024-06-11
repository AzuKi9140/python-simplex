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

    iter = 0

    while True:
        if iter > 10:
            break
        # print("-"*20)
        # print("iter: {}".format(iter))

        # comupute primal variable (basic solution)
        x = np.zeros(n + m)
        x[basis] = np.linalg.solve(Ai[:, basis], b)
        print("basis = ", basis)
        print("nonbasis = ", nonbasis)
        print("x[basis] = ", x[basis])

        # compute dual-variable
        y = np.linalg.solve(Ai[:, basis].T, c0[basis])

        # compute reduced-cost
        rc = c0[nonbasis] - y @ Ai[:, nonbasis]
        print("reduced-cost = {}".format(rc))

        # check optimality (dual feasibility)
        if np.all(rc <= error):
            print("number of iterations = {}".format(iter))
            print("optimal solution found")
            print("obj.val. = {}".format(c0[basis] @ x[basis]))
            print(x[0:n])
            break

        max_steppest_edge_ration = 0

        for rc, nonbasis_index in zip(rc, nonbasis):
            d = np.linalg.solve(Ai[:, basis], Ai[:, nonbasis_index])
            ration = rc / np.sqrt(1 + np.sum(d**2))
            print("d = {}".format(d))
            print("ration = {}".format(ration))

            if ration > max_steppest_edge_ration:
                max_steppest_edge_ration = ration
                s = nonbasis_index
                max_ration_d = d

        basis_ratio = []
        for i in range(len(max_ration_d)):
            if max_ration_d[i] > error:
                basis_ratio.append(x[basis][i] / max_ration_d[i])
            else:
                basis_ratio.append(np.inf)
        r = np.argmin(basis_ratio)

        nonbasis[s], basis[r] = basis[r], nonbasis[s]

        print("B = {}".format(basis))
        print("N = {}".format(nonbasis))

        iter += 1


if __name__ == "__main__":
    A = np.array([[2, 0, 0], [1, 0, 2], [0, 3, 1]])
    b = np.array([4, 8, 6])
    c = np.array([3, 4, 2])

    lp_simplex(A, b, c)
