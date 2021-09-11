import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# 1.熵
def EN(image):
    grayscale_num = np.zeros(256)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            grayscale_num[int(image[i][j])] += 1
    temp = 0
    for i in range(len(grayscale_num)):
        p = grayscale_num[i] / np.sum(grayscale_num)
        if p != 0:
            temp -= p * np.log2(p)
    return temp



# 2.互信息
def MI(image_X, image_F):
    grayscale_X = np.zeros(256)
    grayscale_F = np.zeros(256)
    grayscale_XF = np.zeros((256, 256))
    # 统计直方图
    for i in range(image_X.shape[0]):
        for j in range(image_X.shape[1]):
            grayscale_X[int(image_X[i][j])] += 1
            grayscale_F[int(image_F[i][j])] += 1
            grayscale_XF[int(image_X[i][j]), int(image_F[i][j])] += 1

    # 计算联合信息熵
    EN_XF = 0
    sum_XF = np.sum(grayscale_XF)
    grayscale_XF = grayscale_XF / sum_XF
    for i in range(256):
        for j in range(256):
            if grayscale_XF[i][j] != 0:
                EN_XF -= grayscale_XF[i][j] * np.log2(grayscale_XF[i][j])
    print(EN_XF)

    # 计算X图信息熵
    EN_X = 0
    sum_X = np.sum(grayscale_X)
    grayscale_X = grayscale_X / sum_X
    for i in range(256):
        if grayscale_X[i] != 0:
            EN_X -= grayscale_X[i] * np.log2(grayscale_X[i])
    print(EN_X)

    # 计算F图信息熵
    EN_F = 0
    sum_F = np.sum(grayscale_F)
    grayscale_F = grayscale_F / sum_F
    for i in range(256):
        if grayscale_F[i] != 0:
            EN_F -= grayscale_F[i] * np.log2(grayscale_F[i])
    print(EN_F)

    # 计算X和F的互信息
    mi = EN_F + EN_X - EN_XF
    return mi


# 4.结构相似度
def SSIM(image_X, image_F):
    image_raw_data = tf.gfile.FastGFile(image_X, 'rb').read()
    image_raw_data2 = tf.gfile.FastGFile(image_F, 'rb').read()
    im1 = tf.image.decode_bmp(image_raw_data)
    im2 = tf.image.decode_bmp(image_raw_data2)
    ssim = tf.image.ssim(im1, im2, 255)
    with tf.Session() as sess:
        return sess.run(ssim)


# 8.标准偏差
def SD(image):
    avg = np.mean(image)
    temp = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            temp += np.square(image[i][j] - avg)

    return np.sqrt(temp / (image.shape[0] * image.shape[1]))


# 9.空间频率
def SF(image):
    CF = 0
    RF = 0
    for i in range(1, image.shape[0]):
        for j in range(1, image.shape[1]):
            RF += np.square(image[i][j] - image[i][j-1])
            CF += np.square(image[i][j] - image[i-1][j])
    RF = RF / (image.shape[0] * image.shape[1])
    CF = CF / (image.shape[0] * image.shape[1])
    return np.sqrt(RF + CF)


# 10.平均梯度
def AG(image):
    temp = 0
    for i in range(0, image.shape[0] - 1):
        for j in range(0, image.shape[1] - 1):
            temp += np.sqrt((np.square(image[i][j] - image[i][j+1]) + np.square(image[i][j] - image[i+1][j]))/2)
    AG = temp / (image.shape[0] * image.shape[1])
    return AG


# 11.平均梯度
def MG(image):
    temp = 0
    for i in range(1, image.shape[0]):
        for j in range(1, image.shape[1]):
            temp += np.sqrt((np.square(image[i][j] - image[i][j-1]) + np.square(image[i][j] - image[i-1][j]))/2)
    MG = temp / (image.shape[0] * image.shape[1])
    return MG


# 12.均方误差
def MSE(image_X, image_F):
    temp = 0
    for i in range(0, image_X.shape[0]):
        for j in range(0, image_X.shape[1]):
            temp += np.square(image_X[i][j] - image_F[i][j])
    MSE = temp / (image_X.shape[0] * image_X.shape[1])
    return MSE


# 13.均方根误差
def RMSE(image_X, image_F):
    temp = 0
    for i in range(0, image_X.shape[0]):
        for j in range(0, image_X.shape[1]):
            temp += np.square(image_X[i][j] - image_F[i][j])
    RMSE = np.sqrt(temp / (image_X.shape[0] * image_X.shape[1]))
    return RMSE


# 14.峰值信噪比
def PSNR(MSE):
    PSNR = 10 * np.log10(np.square(255) / MSE)


# 16.相关系数
def CC(image_X, image_F):
    avg_X = np.mean(image_X)
    avg_F = np.mean(image_F)
    temp1 = 0
    temp2 = 0
    temp3 = 0
    for i in range(image_X.shape[0]):
        for j in range(image_X.shape[1]):
            temp1 += (image_X[i][j] - avg_X)*(image_F[i][j] - avg_F)
            temp2 += np.square(image_X[i][j] - avg_X)
            temp3 += np.square(image_F[i][j] - avg_F)

    return temp1 / np.sqrt(temp2 * temp3)



# 交叉熵cross entropy
def CERF(image_X, image_F):
    grayscale_X = np.zeros(256)
    grayscale_F = np.zeros(256)
    for i in range(image_X.shape[0]):
        for j in range(image_X.shape[1]):
            grayscale_X[int(image_X[i][j])] += 1
            grayscale_F[int(image_F[i][j])] += 1
    CERF = 0
    for i in range(len(grayscale_X)):
        p1 = grayscale_X[i] / np.sum(grayscale_X)
        p2 = grayscale_F[i] / np.sum(grayscale_F)
        if p1 != 0 and p2 != 0:
            CERF += p1 * np.log2(p1 / p2)
    return CERF

# 图片清晰度 figure definition
def FD(image):
    temp = 0
    for i in range(image_X.shape[0] - 1):
        for j in range(image_X.shape[1] - 1):
            temp += np.sqrt((np.square(image[i][j] - image[i][j + 1]) + np.square(image[i][j] - image[i + 1][j])) / 2)
    FD = temp / (image_X.shape[0] * image_X.shape[1])
    return FD


# relatively warp 相对标准差
def RW(image_X, image_F):
    avg1 = np.mean(image_X)
    avg2 = np.mean(image_F)
    warp1 = 0
    warp2 = 0
    for i in range(image_X.shape[0]):
        for j in range(image_X.shape[1]):
            warp1 += np.square(image_X[i][j] - avg1)
            warp2 += np.square(image_F[i][j] - avg2)
    warp1 = np.sqrt(warp1 / (image_X.shape[0] * image_X.shape[1]))
    warp2 = np.sqrt(warp2 / (image_X.shape[0] * image_X.shape[1]))
    RW = (warp1 - warp2) / warp1
    return RW


image_X = scipy.misc.imread('1.bmp', flatten=True, mode='YCbCr').astype(np.float)
image_F = scipy.misc.imread('0.bmp', flatten=True, mode='YCbCr').astype(np.float)

#print(AG(image_F))
#print(MG(image_F))
print(RW(image_X, image_F))
