import numpy as np

error = 1.0e-10

def check_degeneracy(
    x_basis: np.array,
):
    return np.where(x_basis < error)[0]


def find_min_index_basis(basis, degenerate_indices):
    # degenerate_indicesに基づいてbasisの要素を抽出
    relevant_basis = [basis[i] for i in degenerate_indices]
    # 最小値を持つ要素のインデックスを見つける
    min_value = min(relevant_basis)
    # 元のnonbasisリストでのインデックスを返す
    return basis.index(min_value)


if __name__ == "__main__":
    x_basis = np.array([1, 0, 0])
    print(check_degeneracy(x_basis))
    non_zero_index = np.array([1, 2, 3])
    print(check_degeneracy(non_zero_index))
