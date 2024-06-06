import numpy as np
def SampEn(x, m, r):
    N = len(x)  # 1. 信号总长N
    x = np.array(x)
    if x.ndim != 1:
        raise ValueError("x的维度不是一维")
    if N < m + 1:
        raise ValueError("len(x)小于m+1")

    AB = []
    for temp in range(2):  # 7.	将窗 m 增长为 m+1: temp = 0：m ； temp = 1: m+1
        # 3. 以m为窗，将时间序列分为k = n-m+1个序列
        X = []
        m = m + temp
        for i in range(N + 1 - m):
            X.append(x[i:i + m])
        X = np.array(X)

        # 4. 计算每个i与所有j之间的绝对值距离，统计距离d小于r的个数：N_m(i)【count】
        C = []
        for index1, i in enumerate(X):
            count = 0
            for index2, j in enumerate(X):
                if index1 != index2:
                    if np.max(np.abs(i - j)) <= r:
                        count += 1
            # 5. 获取每个i的 C
            C.append(count / (N - m + 1))
        # 处理C为0的值，替换为一个非零的很小的数,以防取对数时报错
        C = np.array(C)
        C = np.where(C == 0, 1e-10, C)

        # 6. 求所有i得平均值
        AB.append(np.sum(C) / (N - m + 1))

    # 8. 获取样本熵
    SE = np.log(AB[0]) - np.log(AB[1])
    return SE
