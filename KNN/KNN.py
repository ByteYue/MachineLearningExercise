# encoding: utf-8

import numpy as np
import struct
from sklearn.neural_network import MLPClassifier
from sklearn import neighbors
import time

# 训练集文件
train_images_idx3_ubyte_file = './train-images-idx3-ubyte'
# 训练集标签文件
train_labels_idx1_ubyte_file = './train-labels-idx1-ubyte'

# 测试集文件
test_images_idx3_ubyte_file = './t10k-images-idx3-ubyte'
# 测试集标签文件
test_labels_idx1_ubyte_file = './t10k-labels-idx1-ubyte'


def decode_idx3_ubyte(idx3_ubyte_file):
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('magic_number:%d, imagesnum: %d, imagesize: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('Decoded %d' % (i + 1) + 'pieces')
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('magic_number:%d, imagesnum: %d' % (magic_number, num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.zeros((num_images, 10), dtype='int8')
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('decoded %d' % (i + 1) + 'pieces')
        digit = (int)(struct.unpack_from(fmt_image, bin_data, offset)[0])
        labels[i][digit] = 1
        offset += struct.calcsize(fmt_image)
    return labels


def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    return decode_idx3_ubyte(idx_ubyte_file)


def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    return decode_idx1_ubyte(idx_ubyte_file)


def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    """
    TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  10000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    """
    TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  10000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file)


# 用KNN来实现
def mnistUsingKNN(train_dataSet, train_labels, test_dataSet, test_labels):
    knn = neighbors.KNeighborsClassifier(algorithm='kd_tree', n_neighbors=10)
    print('Start')
    start = time.time()
    knn.fit(train_dataSet, train_labels)
    

    res = knn.predict(test_dataSet)  # 对测试集进行预测
    error_num = 0  # 统计预测错误的数目
    num = len(test_dataSet)  # 测试集的数目
    print(num)
    for i in range(num):  # 遍历预测结果
        # 比较长度为10的数组，返回包含01的数组，0为不同，1为相同
        # 若预测结果与真实结果相同，则10个数字全为1，否则不全为1
        if np.sum(res[i] == test_labels[i]) < 10:
            error_num += 1
    print("Total num:", num, " Wrong num:", \
          error_num, "  CorrectRate:", (1 - error_num / float(num)) * 100, "%")
    print('End, cost:' + str(time.time() - start))


def mnistUsingNN(train_dataSet, train_labels, test_dataSet, test_labels):
    clf = MLPClassifier(hidden_layer_sizes=(100, 50, 25),
                        activation='logistic', solver='adam',
                        learning_rate_init=0.0001, max_iter=2000)
    print('Start')
    start = time.time()
    clf.fit(train_dataSet, train_labels)
    print('End, cost:' + str(time.time() - start))

    res = clf.predict(test_dataSet)  # 对测试集进行预测

    error_num = 0  # 统计预测错误的数目
    num = len(test_dataSet)  # 测试集的数目

    for i in range(num):  # 遍历预测结果
        # 比较长度为10的数组，返回包含01的数组，0为不同，1为相同
        # 若预测结果与真实结果相同，则10个数字全为1，否则不全为1
        if np.sum(res[i] == test_labels[i]) < 10:
            error_num += 1
    print("Total num:", num, " Wrong num:",
          error_num, "  CorrectRate:", (1 - error_num / float(num)) * 100, '%')


if __name__ == '__main__':
    train_images = load_train_images()
    train_labels = load_train_labels()
    test_images = load_test_images()
    test_labels = load_test_labels()

    train_dataSet = train_images.reshape(60000, 784)
    test_dataSet = test_images.reshape(10000, 784)

    # mnistUsingNN(train_dataSet, train_labels, test_dataSet, test_labels)
    mnistUsingKNN(train_dataSet, train_labels, test_dataSet,test_labels)