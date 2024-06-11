import numpy as np
from mip import Model, minimize, xsum, CONTINUOUS

# 与えられたデータ
A = np.array([[1, 12, -2, -12], [0.25, 1, -0.25, -2], [1, -4, 0, -8]])
b = np.array([0, 0, 1])
c = np.array([1, -4, 0, -8])

# モデルのインスタンスを作成
m = Model(sense=minimize)

# 変数の数
num_vars = A.shape[0]

# 変数を生成 (整数変数)
x = [m.add_var(var_type=CONTINUOUS, lb=0) for i in range(num_vars)]

# 目的関数を設定
m.objective = xsum(b[i] * x[i] for i in range(num_vars))

# 制約を追加
for i in range(A.shape[1]):
    m += xsum(A[j][i] * x[j] for j in range(num_vars)) >= c[i]

# 問題を解く
m.optimize()

# 結果を表示
print("Optimal status:", m.status)
print("Optimal value:", m.objective_value)
for v in x:
    print(f"{v.name} = {v.x}")
