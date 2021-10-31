import numpy as np
import cmath, math


def bit_product_sum(x, y):
    return sum([item[0] * item[1] for item in zip(x, y)])


def euclid_distance(point1, point2, root=True):
    # print(point1,point2)
    ret = (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2
    if root:
        ret = math.sqrt(ret)
    # print(point1,point2,ret)
    return ret


def cal_velocity_variance(velocity_distribution):
    """
    计算速度方差：用于后续计算群体压力指标
    :param velocity_distribution: 输入速度的分布【v1，v2，v3】，其中v为二维空间中的速度矢量形如（vx，vy）
    :return: 返回速度方差 = var(vx)+var(vy)
    """
    vx = [i[0] for i in velocity_distribution]
    vy = [i[1] for i in velocity_distribution]
    return np.var(vx) + np.var(vy)


def cosine_similarity(x, y, norm=False):
    """ 计算两个向量x和y的余弦相似度 """
    x, y = list(x), list(y)
    assert len(x) == len(y), "len(x) != len(y)"
    zero_list = [0] * len(x)
    if x == zero_list or y == zero_list:
        return float(1) if x == y else float(0)

    # method 1
    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))

    # method 2
    # cos = bit_product_sum(x, y) / (np.sqrt(bit_product_sum(x, x)) * np.sqrt(bit_product_sum(y, y)))

    # method 3
    # dot_product, square_sum_x, square_sum_y = 0, 0, 0
    # for i in range(len(x)):
    #     dot_product += x[i] * y[i]
    #     square_sum_x += x[i] * x[i]
    #     square_sum_y += y[i] * y[i]
    # cos = dot_product / (np.sqrt(square_sum_x) * np.sqrt(square_sum_y))

    return 0.5 * cos + 0.5 if norm else cos  # 归一化到[0, 1]区间内


def constrain_max_velocity(velo=[0, 0], threshold=1000):
    """
    :param velo: 输入速度
    :param threshold: 速度限制
    :return: 如果速度在速度限制内则返回 TRUE
    """
    # velo = list(velo)
    abs_v = 0
    for v in velo:
        abs_v += abs(v)
    if abs_v > threshold:
        return False
    else:
        return True


def calc_ent(x):
    """
        calculate shanno ent of x
    """

    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp

    return ent


def calc_condition_ent(x, y):
    """
        calculate ent H(y|x)
    """

    # calc ent(y|x)
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        sub_y = y[x == x_value]
        temp_ent = calc_ent(sub_y)
        ent += (float(sub_y.shape[0]) / y.shape[0]) * temp_ent

    return ent


def calc_ent_grap(x, y):
    """
        calculate ent grap
    """

    base_ent = calc_ent(y)
    condition_ent = calc_condition_ent(x, y)
    ent_grap = base_ent - condition_ent

    return ent_grap


if __name__ == '__main__':
    # a = [0, 0, 0, 0, 0, 0, 1, 2]
    # print(calc_ent(np.array(a)))
    # b = [1, 1, 1, 1, 1, 1, 2, 2]
    # print(calc_ent_grap(np.array(a), np.array(b)))

    # print(cosine_similarity([1, 2], [0, 0]))

    # cn = complex(-1, 0)
    # mag, ang = cmath.polar(cn)
    # print(ang)
    # vs = [(-1, -0, 5), (0, 1), (1, 0.5), (1, 1)]
    # print(velocity_variance(vs))
    # print(euclid_distance(vs[-1], vs[-2]))
    print('for test')
