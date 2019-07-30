import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.misc import imread
import math
from init_centroids import init_centroids


def print_cent(iter_num,cent):
    print("iter " + str(iter_num) + ":", end=" ")
    if type(cent) == list:
        cent = np.asarray(cent)
    if len(cent.shape) == 1:
        output = ' '.join(str(np.floor(100*cent)/100).split()).replace('[ ', '[').replace('\n', ' ').replace(' ]',']').replace(' ', ', ')
        print(output)
    else:
        output= ' '.join(str(np.floor(100*cent)/100).split()).replace('[ ', '[').replace('\n', ' ').replace(' ]',']').replace(' ', ', ')[1:-1]
        print(output)


def reset_average(averages):
    for average in averages:
        average.sum_r = 0
        average.sum_g = 0
        average.sum_b = 0
        average.n = 0
    return


def distance(centroid, pixel):
    d = (centroid[0]-pixel[0])**2 + (centroid[1]-pixel[1])**2 + (centroid[2]-pixel[2])**2
    d = math.sqrt(d)
    return d


# update the image to the centroid value
def update_image(centroids, image):
    for pixel in image:
        centroid_num = int(pixel[0])
        pixel[0] = centroids[centroid_num][0]
        pixel[1] = centroids[centroid_num][1]
        pixel[2] = centroids[centroid_num][2]
    return


def print_centroids(iter_num, centroids):
    print("iter " + str(iter_num) + ":", end=" ")
    size = len(centroids)
    for i in range(0, size):
        # print only two digits after the decimal point
        s1 = str(np.floor(centroids[i][0]*100)/100)
        s2 = str(np.floor(centroids[i][1]*100)/100)
        s3 = str(np.floor(centroids[i][2]*100)/100)
        output = "[" + s1+", " + s2 + ", " + s3 + "]"
        print(output, end="")
        if i != size-1:
            print(", ", end="")
    print()
    return


# get the average array and update to the new centroids
def update_centroid(centroid_array, centroids_average, k):
    for i in range(0, k):
        if centroids_average[i].n != 0:
            centroid_array[i][0] = centroids_average[i].sum_r / centroids_average[i].n
            centroid_array[i][1] = centroids_average[i].sum_g / centroids_average[i].n
            centroid_array[i][2] = centroids_average[i].sum_b / centroids_average[i].n

    return


# class to help calculate the new centroids
class Average:
    sum_r = 0
    sum_g = 0
    sum_b = 0
    n = 0


def average_loss(image, centroids, k):
    distance_sum = 0
    pixel_num = 0
    for pixel in image:
        min_distance = distance(centroids[0], pixel)

        for i in range(1, k):
            curr_distance = distance(centroids[i], pixel)
            if curr_distance < min_distance:
                min_distance = curr_distance
        pixel_num += 1
        min_distance= min_distance ** 2
        distance_sum += min_distance
    return distance_sum/pixel_num


def k_means(image, centroids, k):
    # print the first centroids
    print_cent(0, centroids)
    centroids_average = []
    last_iter = 9
    # initialize average array
    for i in range(0, k):
        centroids_average.append(Average())
    # run for ten iterations
    for iteration in range(0, 10):
        for pixel in image:
            min_distance = distance(centroids[0], pixel)
            min_index = 0
            # find the closest centroid
            for i in range(1, k):
                curr_distance = distance(centroids[i], pixel)
                if curr_distance < min_distance:
                    min_distance = curr_distance
                    min_index = i
            centroids_average[min_index].sum_r += pixel[0]
            centroids_average[min_index].sum_g += pixel[1]
            centroids_average[min_index].sum_b += pixel[2]
            centroids_average[min_index].n += 1
            # if its the last iter put in r the group of the pixel
            if iteration == last_iter:
                pixel[0] = min_index
        # get the new centroids ready
        update_centroid(centroids, centroids_average, k)
        print_cent(iteration+1, centroids)
        reset_average(centroids_average)
    # change each pixel to his centroid
    update_image(centroids, image)
    return


def main():
    # run for k=2,4,8,16
    for i in range(1, 5):
        k = 2 ** i
        print("k=" + str(k) + ":")
        path = 'dog.jpeg'
        A = imread(path)
        A = A.astype(float) / 255.
        img_size = A.shape
        X = A.reshape(img_size[0] * img_size[1], img_size[2])
        centroids = init_centroids(X, k)
        k_means(X, centroids, k)
    return


if __name__ == "__main__":
    main()
