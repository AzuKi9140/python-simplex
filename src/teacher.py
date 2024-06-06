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
        print("x[basis] = ", x[basis])
        bb = np.linalg.solve(Ai[:, basis], b)  # 突貫工事したので，非効率

        # compute dual-variable
        y = np.linalg.solve(Ai[:, basis].T, c0[basis])

        # compute reduced-cost
        rc = c0[nonbasis] - y @ Ai[:, nonbasis]
        # print("reduced-cost = {}".format(rc))

        # check optimality (dual feasibility)
        if np.all(rc <= error):
            print("number of iterations = {}".format(iter))
            print("optimal solution found")
            print("obj.val. = {}".format(c0[basis] @ x[basis]))
            print(x[0:n])
            break
        else:
            # decide an entering-variable
            s = np.argmax(rc)
            # print("entering index = {}".format(nonbasis[s]))
        d = np.linalg.solve(
            Ai[:, basis], Ai[:, nonbasis[s]]
        )  # obtain a column correspoding to s
        # print("d = {}".format(d))

        # check unboundedness
        if np.all(d <= error):
            print("problem is unbounded")
            break
        else:
            if iter % 50 == 0:
                print("iter: {}".format(iter))
                # print("current obj.val. = {}".format(bb@c0[basis]))
                print("current obj.val. = {}".format(x[basis] @ c0[basis]))
                # print("current primal sol. =")

            ratio = []
            for i in range(len(d)):
                if d[i] > error:
                    ratio.append(bb[i] / d[i])
                else:
                    ratio.append(np.inf)

            # decide leaving-variable
            r = np.argmin(ratio)
            # print("leaving index = {}".format(basis[r]))

            nonbasis[s], basis[r] = basis[r], nonbasis[s]

            # print("B = {}".format(basis))
            # print("N = {}".format(nonbasis))

            iter += 1


if __name__ == "__main__":
    # A = np.array([[2, 0, 0], [1, 0, 2], [0, 3, 1]])
    # b = np.array([4, 8, 6])
    # c = np.array([3, 4, 2])
    A = np.array([[1, 12, -2, -12], [0.25, 1, -0.25, -2], [1, -4, 0, -8]])
    b = np.array([0, 0, 1])
    c = np.array([1, -4, 0, -8])

    lp_simplex(A, b, c)
