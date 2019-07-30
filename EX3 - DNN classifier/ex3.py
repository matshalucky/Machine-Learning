import sys
import random
import numpy as np
npa = np.array


regulation = 1
lr = 0.01


def shuffle_train_data(train_x, train_y):
    train_x_y = np.concatenate((train_x, train_y), axis=1)
    np.random.shuffle(train_x_y)
    x = train_x_y[:,:-1]
    y= train_x_y[:,-1]
    return x,y

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    temp = e_x.sum()
    if temp == 0:
        temp = 1
    return e_x / temp


def dRelu(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def relu(z):
    return np.maximum(0,z)

def backword(y,h1,z1,w2,h2,x):
    index_y = int(y)
    h2[index_y] -= 1
    g_b2 = h2
    g_w2 = np.dot(h2, np.transpose(h1))

    # mayabe here
    z1 = dRelu(z1)
    # here need w2 h2
    g_b1 = np.dot(np.transpose(w2), h2) * z1
    x= np.reshape(x,(1,len(x)))
    g_w1= np.dot(g_b1,x)
    return g_w1,g_b1,g_w2,g_b2

def update(g_w1,g_b1,g_w2,g_b2 , w1,b1,w2,b2):
    w1 = regulation * w1 -lr*g_w1
    w2 = regulation * w2 -lr * g_w2
    b1 = regulation * b1 -lr * g_b1
    b2 = regulation * b2 -lr * g_b2
    return  w1, b1, w2,  b2


def predictions(w1,b1,w2,b2,x_test) :
    predict =[]
    for  x in  x_test:
        x = np.reshape(x, (784, 1))
        h1, z1, h2 = forward(w1, b1, w2, b2, x)
        y_hat = np.unravel_index(np.argmax(h2), h2.shape)
        predict.append(y_hat[0])
    return predict

def save_predictions(array_of_predictions ):
    ndarray_of_predictions = np.asarray(array_of_predictions)
    f = open('C:/Users/Matan/PycharmProjects/ex3/test_y', 'w+')
    size = len(array_of_predictions)
    x = ndarray_of_predictions.reshape((size, 1))
    np.savetxt(f, x, newline='\n', fmt='%i')
    f.close()


def test(w1,b1,w2,b2,x_test,y_test) :
    correct_answer=0

    for y,x in zip( y_test,x_test):
        x = np.reshape(x, (784, 1))
        h1, z1, h2 = forward(w1, b1, w2, b2, x)
        y_hat = np.unravel_index(np.argmax(h2),h2.shape)
        if(y_hat[0] == int(y)):
            correct_answer +=1


    print(correct_answer/len(y_test))



def forward(w1, b1, w2, b2, x):
    z1 = np.dot(w1, x) + b1
    h1 = relu(z1)
   # if h1.max()!= 0.0 :
    #    h1=h1/h1.max()
    z2 = np.dot(w2, h1) + b2
    h2 = softmax(z2)
    return h1, z1, h2


def main():
    # checks we get enough argument
    if len(sys.argv) != 4:
        print("not enough argument")
        return
    INPUT_SIZE =10000
    # gets the two arguments which are the algorithm's input
    train_x = np.loadtxt(fname=sys.argv[1],max_rows=INPUT_SIZE)
    train_y = np.loadtxt(fname=sys.argv[2],max_rows=INPUT_SIZE)
    train_y = np.reshape(train_y, (INPUT_SIZE, 1))
   # test_x =  np.loadtxt(fname=sys.argv[3],max_rows=50)

    train_x /= 255.0

    H = 20
    PIXELS = 784
    w1 = np.random.uniform(-0.08,0.08,(H,PIXELS))
    b1 = np.random.uniform(-0.08, 0.08, (H, 1))
    w2 = np.random.uniform(-0.08, 0.08, (10, H))
    b2 = np.random.uniform(-0.08, 0.08, (10, 1))
    i = 0
    epochs = 5
    for j in range(epochs):
        train_x_curr, train_y_curr = shuffle_train_data(train_x, train_y)
        train_set_x = train_x_curr[:int(((INPUT_SIZE / 6) * 5))]
        train_test_x = train_x_curr[int(((INPUT_SIZE / 6) * 5)):]
        train_set_y = train_y_curr[:int(((INPUT_SIZE / 6) * 5))]
        train_test_y = train_y_curr[int(((INPUT_SIZE / 6) * 5)):]
        for x in train_set_x:
            x = np.reshape(x,(PIXELS,1))
            h1,z1,h2 =forward(w1,b1,w2,b2,x)
            g_w1, g_b1, g_w2, g_b2 = backword(train_set_y[i], h1, z1, w2, h2, x)
            w1, b1, w2, b2 = update(g_w1,g_b1,g_w2,g_b2, w1, b1, w2, b2)
            i += 1
        i=0

    pre = predictions(w1,b1,w2,b2,train_test_x)
    save_predictions(pre)
    test(w1, b1, w2, b2, train_test_x, train_test_y)
    print("matan")

if __name__ == "__main__":
    main()

