import numpy as np

# 例としての y と Ai[:, nonbasis]
y = np.array([1, 2, 3])
print(y.shape)
x = np.array([[1, 2, 3]])
print(x.shape)
Ai_nonbasis = np.array([[4, 5, 6], [7, 8, 9], [10, 11, 12]])


# 行列積を計算
result = y @ Ai_nonbasis 
# 3*1 * 3*3 = 計算できないが、@だとy.T * Ai_nonbasisの転置が出力される
print(result)  
print(result.shape)

