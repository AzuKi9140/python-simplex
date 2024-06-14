import numpy as np
from settings import time_dec
from mip import CONTINUOUS, Model, maximize, xsum

@time_dec
def lp_simplex(A, b, c):
    p = Model()

    (m,n) = A.shape
    print("num. of variables = {}".format(n))
    print("num. of constraints = {}".format(m))

    x = {}
    for j in range(n):
        x[j] = p.add_var(var_type=CONTINUOUS, lb=0.0)

    for i in range(m):
        p.add_constr(xsum(A[i,j]*x[j] for j in range(n)) <= b[i])

    p.objective = maximize(xsum(c[j]*x[j] for j in range(n)))

    p.optimize()

    print(p.status)
    print("obj.val. = {}".format(p.objective_value))

if __name__ == "__main__":
    # 例題
    A = np.array([[2, 0, 0], [1, 0, 2], [0, 3, 1]])
    b = np.array([4, 8, 6])
    c = np.array([3, 4, 2])
    lp_simplex(A, b, c)

    # 計測(karate)
    A = np.loadtxt("./test_data/A_karate.txt", delimiter="\t")
    m = len(A)
    b = np.ones(m)
    c = np.loadtxt("./test_data/c_karate.txt", delimiter="\t")
    lp_simplex(A, b, c)
    
    # 計測(dolphins)
    A = np.loadtxt("./test_data/A_dolphins.txt", delimiter="\t")
    m = len(A)
    b = np.ones(m)
    c = np.loadtxt("./test_data/c_dolphins.txt", delimiter="\t")
    lp_simplex(A, b, c)

    # 計測(ランダム)
    n = 300
    m = 500
    a = 0.1
    b = 0.7
    from matrix_generator import generate_random_lp_problem
    A, b_vec, c_vec = generate_random_lp_problem(n, m, a, b)

    lp_simplex(A, b_vec, c_vec)
