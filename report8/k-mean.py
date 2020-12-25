import numpy as np
import math
import pandas
import matplotlib.pyplot as plot
import random

r = lambda: random.uniform(0, 100)
data_size = 100


def main(): # 클러스터 두개로 나눔
    mu1 = [r(), r()]
    mu2 = [r(), r()]
    print(mu1, mu2, "are the first mus")
    data = generate_data()
    counter = 0
    mu1_points, mu2_points = None, None
    while True:
        mu1_points, mu2_points = new_points(data, mu1, mu2)  # 거리기반 데이터 분류
        new_mu1, new_mu2 = converge(mu1_points, mu2_points) # 클러스터 중심을 현 데이터의 중심으로 이동
        if mu1 == new_mu1 and mu2 == new_mu2: # 클러스터 중심점이 아무 변화가 없을때 탈출
            break
        mu1, mu2 = new_mu1, new_mu2 # 클르스터 중심점에 변화가 있으면
        counter += 1 # 센터값 증가
        if counter == 1000:
            print("We are at iteration", counter)
    print("Converged points:", mu1, mu2)
    frame1 = pandas.DataFrame(mu1_points)
    frame2 = pandas.DataFrame(mu2_points)
    plot.scatter(frame1.iloc[0:, 0], frame1.iloc[0:, 1], color='b', s=100)
    plot.scatter(frame2.iloc[0:, 0], frame2.iloc[0:, 1], color='r', s=100)
    plot.scatter(mu1[0], mu1[1], color='y', s=120)
    plot.scatter(mu2[0], mu2[1], color='purple', s=120)
    x = np.array(range(101))
    plot.plot(x, line_of_separation(x, mu1, mu2))
    plot.show()


def line_of_separation(x, mu1, mu2): # 두 클러스터 사이를 구분하기 위해 선으로 표시
    slope = -((mu1[0] - mu2[0]) / (mu1[1] - mu2[1]))
    y_intercept = ((mu1[1] + mu2[1]) / 2) - (slope * ((mu1[0] + mu2[0]) / 2))
    return (slope * x) + y_intercept


def converge(mu1_points, mu2_points):  # 클러스터 중심을 현 데이터의 중심으로 이동
    n = len(mu1_points)
    mu1_x, mu1_y = 0, 0
    for i in range(n):
        mu1_x += mu1_points[i][0]
        mu1_y += mu1_points[i][1]

    m = len(mu2_points)
    mu2_x, mu2_y = 0, 0
    for i in range(m):
        mu2_x += mu2_points[i][0]
        mu2_y += mu2_points[i][1]

    return [mu1_x / n, mu1_y / n], [mu2_x / m, mu2_y / m]


def new_points(data, mu1, mu2): # 거리기반 데이터 분류
    mu1_points = []
    mu2_points = []
    for i in data: # 데이터들은 가장 가까운 클러스들에 할당된다.
        if distance(i, mu1) < distance(i, mu2): #
            mu1_points += [i]
        else:
            mu2_points += [i]
    return mu1_points, mu2_points


def generate_data(): # 랜덤하게 데이터 생성
    data = []
    for i in range(data_size):
        data += [[r(), r()]]
    return data


def distance(p0, p1): # 거리 공식 적용
    return math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)


main()