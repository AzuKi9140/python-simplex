import numpy as np

def rand_coef(n, m, a, b):
    neg_rate = int(n * m * a)
    pos_rate = int(n * m * b)

    neg_r = np.random.uniform(-0.5, -10e-10, neg_rate)
    pos_r = np.random.uniform(10e-10, 1.0, pos_rate)
    zero = np.zeros(n * m - len(neg_r) - len(pos_r))

    r = np.concatenate([neg_r, pos_r, zero])
    r_shuffled = np.random.permutation(r)

    mat = np.reshape(r_shuffled, (m, n))

    return mat

def generate_random_lp_problem(n, m, a, b):
    np.random.seed(seed=1)

    A_init = rand_coef(n, m, a, b)
    A = np.r_[A_init, np.ones((1, n))]

    b_init = np.ones(m)
    b_vec = np.r_[b_init, n]

    c = rand_coef(n=n, m=1, a=a, b=b)
    c_vec = c.reshape(-1)

    return A, b_vec, c_vec