import numpy as np
import skimage
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import os
import scipy.misc as sm
from scipy import ndimage

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def load_data(dir_name = 'test_imgs'):    
    '''
    Load images from the "test_imgs" directory
    Images are in JPG and we convert it to gray scale images
    '''
    gray_imgs = []
    gray_rotated_imgs = []
    raw_imgs = []
    names = []
    for filename in os.listdir(dir_name):
        if os.path.isfile(dir_name + '/' + filename):
            img = mpimg.imread(dir_name + '/' + filename)

            raw_imgs.append(img)

            gray_img = rgb2gray(img)
            gray_imgs.append(gray_img)

            rotated_img = ndimage.rotate(gray_img, 90, reshape=True)
            gray_rotated_imgs.append(rotated_img)

            names.append(filename)

            # save after rgb
            #plt.imshow(img, None)
            #plt.savefig('./archive/plots/'+filename)
            #plt.clf()
    return raw_imgs, gray_imgs, gray_rotated_imgs, names


def visualize(imgs, format=None, gray=False):
    plt.figure(figsize=(20, 40))
    for i, img in enumerate(imgs):
        if img.shape[0] == 3:
            img = img.transpose(1,2,0)
        plt_idx = i+1
        plt.subplot(2, 2, plt_idx)
        plt.imshow(img, format)
    plt.show()

def fixYOutliers(y_array, sample_size = 10, range_limit = 1.001, multiplier = 100):    #get median of previous 5 values
    #get median of post 5 values
    max_index = len(y_array) - sample_size
    prev = y_array[0:sample_size]
    post = y_array[sample_size+1:sample_size+sample_size+1]
    for index, y_point in enumerate(y_array):
        if max_index >= index >= sample_size: #skip first 5
            prev.sort() 
            post.sort()
            middle_index = len(prev) // 2
            prev_med_val = (prev[middle_index] + prev[~middle_index]) / 2
            post_med_val = (post[middle_index] + post[~middle_index]) / 2
    
            avg = (prev_med_val + post_med_val)/2
            
            #if not (float(avg-range_limit) <= float(y_array[index]) and float(y_array[index])>= float(avg+range_limit)):
            #if not prev[middle_index] <= y_array[index] <= post[middle_index]:
            mid = (prev[middle_index] + post[middle_index]) / 2 * multiplier
            if not (mid/range_limit) <= y_array[index]*multiplier <= (mid*range_limit):
                #print(y_array[index]," is NOT between ",avg-range_limit, " and ",avg+range_limit)
                y_array[index] = mid/multiplier#(prev[middle_index] + post[middle_index])/2#avg
                #print("setting new value", y_array[index])
            #else:
            #    print(y_array[index]*multiplier, " is between ",mid/range_limit, " and ",mid*range_limit)
            prev = y_array[index-sample_size:index]
            post = y_array[index:index+sample_size]
    return y_array