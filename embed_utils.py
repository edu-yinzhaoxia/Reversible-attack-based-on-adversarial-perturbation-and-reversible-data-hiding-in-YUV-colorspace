import math

import numpy as np
from helpers import *


def compressfun(signal):
    L = len(signal)

    precision = 32

    signal = signal[:, 0].tolist()
    # call the encoder function
    code, dic = Arithmetic_encode(signal, precision)

    return code


def crossPrediction(I):
    [m, n] = I.shape
    I1 = np.pad(I, pad_width=((1, 1), (1, 1)), mode='constant', constant_values=(0, 0))

    p = np.zeros((I1.shape))
    Ic = I1.copy()

    for ii in range(1, m+1):
        for jj in range(1, n+1):
            if (ii+jj+2)%2 == 0:
                Ic[ii, jj] = math.floor((I1[ii+1, jj] + I1[ii-1, jj] + I1[ii, jj+1] + I1[ii, jj-1])/4)
                p[ii, jj] = 1

    Ic = Ic[1:-1, 1:-1]
    p = p[1:-1, 1:-1]
    er = I - Ic

    return Ic, er, p


def EmbeddingHistogramShifting(I_pred, data, T, er, p):
    [m, n] = I_pred.shape
    ed = np.zeros((m, n))
    temp = 0

    for ii in range(m):
        for jj in range(n):
            if p[ii, jj] == 1:
                if (er[ii, jj] >= (-T)) & (er[ii, jj] < T):
                    temp = temp + 1
                    ed[ii, jj] = 2 * er[ii, jj] + data[temp-1]
                elif er[ii, jj] >= T:
                    ed[ii, jj] = er[ii, jj] + T
                elif er[ii, jj] < (-T):
                    ed[ii, jj] = er[ii, jj] - T
    I_stego = I_pred + ed
    noBitsEmbedded = temp

    return I_stego, noBitsEmbedded


def dotPrediction(I):

    [m, n] = I.shape
    I1 = np.pad(I, pad_width=((1, 1), (1, 1)), mode='constant', constant_values=(0, 0))

    p = np.zeros((I1.shape))
    Id = I1.copy()

    for ii in range(1, m + 1):
        for jj in range(1, n + 1):
            if (ii + jj + 2) % 2 != 0:
                Id[ii, jj] = math.floor((I1[ii + 1, jj] + I1[ii - 1, jj] + I1[ii, jj + 1] + I1[ii, jj - 1]) / 4)
                p[ii, jj] = 1

    Id = Id[1:-1, 1:-1]
    p = p[1:-1, 1:-1]
    er = I - Id

    return Id, er, p


def calculate_threshold(I, data, len):
    """根据嵌入数据的长度自适应选择阈值T"""
    T = 1

    # % -----Embed in Cross Pixels - ---- %
    [ICrossPred, ec, pc] = crossPrediction(I)

    while 1:
        [Ic, crossEC] = EmbeddingHistogramShifting(ICrossPred, data, T, ec, pc)

        # % -----Embed in Dot Pixels - ---- %
        [IDotPred, ed, pd] = dotPrediction(Ic)
        [Istego, dotEC] = EmbeddingHistogramShifting(IDotPred, data[crossEC:], T, ed, pd)

        totalEmbeddedData = crossEC + dotEC

        if totalEmbeddedData >= len:
            break

        T = T + 5

    return T


def PE_encode(I, T, data):
    # % % % % % -----Embedding Process - ---- % % % % %
    # % -----Embed in Cross Pixels - ---- %
    [ICrossPred, ec, pc] = crossPrediction(I)
    [Ic, crossEC] = EmbeddingHistogramShifting(ICrossPred, data, T, ec, pc)

    # % -----Embed in Dot  Pixels - ---- %
    [IDotPred, ed, pd] = dotPrediction(Ic)
    [Istego, dotEC] = EmbeddingHistogramShifting(IDotPred, data[crossEC:], T, ed, pd)

    return Istego


def ExtractionHistogramShifting(Isteg, er, T, p):
    [m, n] = Isteg.shape
    temp = 0
    dataRec = []
    e = np.zeros((m, n))

    for ii in range(m):
        for jj in range(n):
            if p[ii, jj] == 1:
                if (er[ii, jj] >= (-2*T)) & (er[ii, jj] < (2*T)):
                    # temp = temp + 1
                    # dataRec[temp - 1] = er[ii, jj] % 2
                    data = er[ii, jj] % 2
                    dataRec.append(data)
                    e[ii, jj] = np.floor(er[ii, jj] / 2)
                elif er[ii, jj] < (-2*T):
                    e[ii, jj] = er[ii, jj] + T
                elif er[ii, jj] >= (2*T):
                    e[ii, jj] = er[ii, jj] - T

    Irec = Isteg + e

    return Irec, dataRec


def PE_decode(T, Istego):
    # % -----Extraction & Recovery: Dot Pixels - ---- %
    [IDotPredExtract, edExtract, pd] = dotPrediction(Istego)
    [IDotRec, dataDotRec] = ExtractionHistogramShifting(IDotPredExtract, edExtract, T, pd)

    # % -----Extraction & Recovery: Cross  Pixels - ---- %
    [ICrossPredExtract, ecExtract, pc] = crossPrediction(IDotRec)
    [Irec, dataCrossRec] = ExtractionHistogramShifting(ICrossPredExtract, ecExtract, T, pc)

    recoveredData = dataCrossRec + dataDotRec

    return Irec, recoveredData


def embed_main(ori_yuv, advy_yuv):
    """将advy_yuv和ori_yuv的y通道差异嵌入uv通道"""

    # 计算原始图像和对抗样本y通道的差值作为嵌入数据
    ori_y = ori_yuv[:, :, 0]
    advy_y = advy_yuv[:, :, 0]
    err = np.zeros((299*299, 1))

    index = 0
    for i in range(299):
        for j in range(299):
            err[index] = ori_y[i, j] - advy_y[i, j]
            index = index + 1

    # 算术编码对嵌入信息进行压缩
    mess = compressfun(err)
    mess_s = mess

    L = len(mess)
    bpp = L / (advy_y.size) / 2
    # print('bpp = ', bpp)

    # 对抗样本的cbcr通道作为载体图像
    img_cb = advy_yuv[:, :, 1]
    img_cr = advy_yuv[:, :, 2]
    img = np.vstack((img_cb, img_cr))
    img_s = img.copy()


    # PEE嵌入
    [m, n] = img.shape  # 598*299
    count = math.ceil(L / (m*n))
    T_flag = []

    # 进行信息嵌入
    for index in range(count):

        cover = img.copy()  # 载体图像

        data = mess
        len_data = len(data)  # 带嵌入数据长度

        # 嵌入所有数据
        if len_data <= 0:
            break

        # 满嵌448*224， 前17位表示嵌入数据长度，实际嵌入448*224-17
        if len_data >= (m*n-18):
            # 将数据长度转化为17位二进制
            len_embed = bin(m*n-18)[2:]
            len_embed_data = []
            for i in range(18 - len(len_embed)):
                len_embed_data.append(0)
            for j in range(len(len_embed)):
                len_embed_data.append(int(len_embed[j]))

            data = len_embed_data + data
            embed_data = data[:m*n]

            # 根据嵌入数据计算T值
            T = calculate_threshold(cover, embed_data, m*n)
            T_flag.append(T)

            mess = data[m*n:]

        else:  # 需要填充数据
            len_embed = bin(len(data))[2:]
            len_embed_data = []
            for i in range(18 - len(len_embed)):
                len_embed_data.append(0)
            for j in range(len(len_embed)):
                len_embed_data.append(int(len_embed[j]))

            embed_data = len_embed_data + data

            for i in range(m*n-len(embed_data)):
                embed_data.append(0)

            # 根据嵌入数据计算T值
            T = calculate_threshold(cover, embed_data, 18+len(data))
            T_flag.append(T)

            mess = []

        # 进行信息嵌入
        Istego = PE_encode(cover, T, embed_data)

        img = Istego

    img_resutl = img.copy()


    # # 嵌入完成，提取数据
    # recover_data = []
    #
    # for c in range(count, 0, -1):
    #     T_value = T_flag[c-1]
    #
    #     [Irec, recoverData] = PE_decode(T_value, img)
    #
    #     # 根据前17位计算嵌入的数据长度
    #     s = ''
    #     for i in range(18):
    #         s = s + str(int(recoverData[i]))
    #     data_len = int(s, 2)
    #     data = recoverData[18:18+data_len]
    #     recover_data = data + recover_data
    #
    #     img = Irec
    #
    # print('recoverdata', mess_s == recover_data)
    # print('recoverImg', (Irec == img_s).all())

    return img_resutl







