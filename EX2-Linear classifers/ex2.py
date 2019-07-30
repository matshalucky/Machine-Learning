import random
import sys
import numpy as np

def main():
    # open files
    x_train_file = open(str(sys.argv[1]), mode = "r")
    y_train_file = open(str(sys.argv[2]), mode = "r")
    # parse the x file to matrix
    x_vector_list = parse_x_to_matrix(x_train_file)
    # normalize x_train values.
    x_vector_list = min_max_normalization(x_vector_list)
    # change the y_train to int.
    y_train_list = turn_y_to_integer(y_train_file)

    zip_x_y = list(zip(x_vector_list, y_train_list))
    random.shuffle(zip_x_y)
    zip_x_y = np.array(zip_x_y)
    totalAll = len(x_vector_list)
    xArr, yArr = (zip(*zip_x_y))
    xArr = np.array(xArr)
    yArr = np.array(yArr)

    # new array for the train
    xArrNew = xArr[:int(((totalAll / 6) * 5))]
    xTest = xArr[int(((totalAll / 6) * 5)):]
    yArrNew = yArr[:int(((totalAll / 6) * 5))]
    yTest = yArr[int(((totalAll / 6) * 5)):]

    # calculates total number of lines
    total = len(xTest)
    # call the perceptron algorithm.
    perceptron_w = perceptron_alg(xArrNew, yArrNew)
    pa_w = PA_alg(xArrNew, yArrNew)
    svm_w = SVM_alg(xArrNew, yArrNew)
    #x_test = parse_x_to_matrix(open(str(sys.argv[3]), mode = "r"))
    #y_test = parse_x_to_matrix(open(str(sys.argv[4]), mode = "r"))
    #print (y_test)
    #x_test = min_max_normalization(x_test)
    #y_test = min_max_normalization(y_test)
    mistakes_rate(xTest,yTest,perceptron_w)
    mistakes_rate(xTest,yTest,svm_w)
    mistakes_rate(xTest,yTest,pa_w)

    # print the needed output.
   # make_test(perceptron_w, svm_w, pa_w, xTest)


def make_test(perceptron_w, svm_w, pa_w, x_test):
    for x in x_test:
        print("perceptron: " + str(np.argmax(np.dot(perceptron_w, x)))
              + ", svm: " + str(np.argmax(np.dot(svm_w, x)))
              + ", svm: " + str(np.argmax(np.dot(svm_w, x)))
              + ", pa: " + str(np.argmax(np.dot(pa_w, x))))


def parse_x_to_matrix(x_train_file):
    x_vector_list = []
    line = x_train_file.readline()
    while line:
        # split line by ","
        x_vector = [x.strip() for x in line.split(',')]
        # change first letter to number
        if (x_vector[0] == "F"):
            x_vector[0] = 1.5
        elif (x_vector[0] == "M"):
            x_vector[0] = 2
        elif (x_vector[0] == "I"):
            x_vector[0] = 0.5
        else:
            break
        x_vector = list(map(float, x_vector))
        x_vector_list.append(x_vector)
        line = x_train_file.readline()
    return x_vector_list


def min_max_normalization(x_train):
    x_transpose = np.transpose(x_train)
    normalaized_matrix = []
    for x in x_transpose:
        min = np.min(x)
        max = np.max(x)
        if max- min == 0:
            continue
        v_tag = (x - min) / (max - min)
        normalaized_matrix.append(v_tag)
    return np.transpose(normalaized_matrix)


def turn_y_to_integer(file):
    list = []
    line = file.readline()
    while line:
        first_digit = line[0]
        integer = int(first_digit)
        list.append(integer)
        line = file.readline()
    return list


def perceptron_alg(x_train, y_train):
    # create zeros matrix
    w = np.zeros((3,8))
    eta = 0.1
    epochs = 10
    for e in range(epochs):
        zip_x_y = list(zip(x_train, y_train))
        random.shuffle(zip_x_y)
        x_train, y_train = zip(*zip_x_y)
        for x,y in zip(x_train, y_train):
            y_hat = np.argmax(np.dot(w,x))
            if y != y_hat:
                w[y, :] = w[y, :] + np.multiply(eta,x)/(e+1)
                w[y_hat, :] = w[y_hat, :] -  np.multiply(eta,x)/(e+1)
       # eta /= (e + 1)
    return w


def SVM_alg(x_train, y_train):
    # create zeros matrix
    w = np.zeros((3, 8))
    eta = 0.1
    lamda = 0.2
    epochs = 10
    for e in range(epochs):
        zip_x_y = list(zip(x_train, y_train))
        random.shuffle(zip_x_y)
        x_train, y_train = zip(*zip_x_y)
        for x, y in zip(x_train, y_train):
            y_hat = np.argmax(np.dot(w, x))
            if y != y_hat:
                w[y, :] = w[y, :]*(1-eta*lamda) + np.multiply(eta, x)
                w[y_hat, :] = w[y_hat, :]*(1-eta*lamda) - np.multiply(eta, x)
                remain_index = 3 - y - y_hat
                w[remain_index, :] = w[remain_index, :]*(1-eta*lamda)
    return w


def PA_alg(x_train, y_train):
    # create zeros matrix
    w = np.zeros((3, 8))
    eta = 0.01
    epochs = 10
    for e in range(epochs):
        zip_x_y = list(zip(x_train, y_train))
        random.shuffle(zip_x_y)
        x_train, y_train = zip(*zip_x_y)
        for x, y in zip(x_train, y_train):
            y_hat = np.argmax(np.dot(w, x))
            if y != y_hat:
                loss = max(0, 1 - np.dot(w[y, :], x) + np.dot(w[y_hat, :], x))
                normed_vec = pow(np.linalg.norm(x), 2)
                tau = loss / (2*normed_vec)
                w[y, :] = w[y, :] + x*tau
                w[y_hat, :] = w[y_hat, :] - x*tau
            eta /= np.sqrt((e + 1))
    return w


def mistakes_rate(x_test , y_test, w):
    counter = 0
    for x, y in zip(x_test, y_test):

        y_hat = np.argmax(np.dot(w, x))

        if y != y_hat:
            counter += 1
    #print (counter)
    prec = counter/ len(x_test)
    print (prec)
    return


if __name__ == "__main__":
    main()