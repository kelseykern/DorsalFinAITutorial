from __future__ import print_function 
from utils import utils
import canny_edge_detector as ced #credit to https://github.com/FienSoP/
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import math

animal = "shark"

def clean_data(array_of_img_dirs):

    array_of_matrix = []
    array_of_left_x_array = []
    array_of_left_y_array = []

    array_of_right_x_array = [] 
    array_of_right_y_array = []

    all_edge_images = []
    all_raw_images = []
    all_img_names = []
    all_rotated_images = []
    graph_points = []

    center_colors = []

    for dir_name in array_of_img_dirs:
        if dir_name == 'images/'+animal+'s_cropped/set/succeeded_in_dot_17':
            lthreshold = 0.09
            hthreshold = 0.17
        elif dir_name == 'images/'+animal+'s_cropped/set/succeeded_in_dot_3':
            lthreshold = 0.09
            hthreshold = 0.3
        elif dir_name =='images/'+animal+'s_cropped/set/succeeded_in_dot_6':
            lthreshold = 0.09
            hthreshold = 0.6
        
        raw_imgs, gray_imgs, gray_rotated_imgs, img_names_array = utils.load_data(dir_name)

        all_img_names.extend(img_names_array)

        all_raw_images.extend(raw_imgs)

        detector1 = ced.cannyEdgeDetector(gray_rotated_imgs, sigma=1.4, kernel_size=5, lowthreshold=lthreshold, highthreshold=hthreshold, weak_pixel=100)
        edge_rotated_imgs = detector1.detect()
        all_rotated_images.extend(edge_rotated_imgs)

    # 1. Create array of edge detected matrixes
        detector2 = ced.cannyEdgeDetector(gray_imgs, sigma=1.4, kernel_size=5, lowthreshold=lthreshold, highthreshold=hthreshold, weak_pixel=100)
        edge_images = detector2.detect()
        all_edge_images.extend(edge_images)

    # 2. Create single line graph from each matrix
    # we will only focus on the first pixel seen from top down
    img_index=0
    for edge_img in all_edge_images:
        x_array = []
        y_array = []
        matrix = np.array(edge_img)
        dimensions = np.shape(matrix)
        new_matrix = np.zeros(dimensions, dtype=float, order='C')
        img_filename = all_img_names[img_index]

        # create list of graph points for original matrix
        i = 0
        x=len(matrix.transpose())
        for column in reversed(matrix.transpose()):
            #iterate top to bottom
            #loop until we hit our first pixel
            j = 0
            y=len(column)
            for p in column:
                if p == 255:
                    #we found a pixel
                    #store location and go to next column
                    new_matrix[j][i] = 255
                    graph_points.append([x,y])
                    x_array.append(x)
                    y_array.append(y)
                    break
                else:
                    new_matrix[j][i] = 0
                j=j+1
                y=y-1
            i=i+1
            x=x-1

        # Extract color
        save_dir = './output/plots/raw_'+img_filename
        #plt.imshow(raw_imgs[::-1], origin='lower')
        mid_x = x_array[y_array.index(max(y_array))]
        raw_image = all_raw_images[img_index]
        rgb = raw_image[::-1][0][mid_x]
        #plt.savefig(save_dir)
        #plt.clf()
        center_colors.append(rgb)

        # 3. Detrend the data
        y_new = []
        first_x = x_array[0]
        first_y = y_array[0]
        last_x = x_array[len(x_array)-1]
        last_y = y_array[len(y_array)-1]
        for i, val in enumerate(y_array):
            y_new.append(((last_y - first_y) / (last_x - first_x)) * (x_array[i]-last_x) + last_y)
        # detrended
        y_array = np.subtract(y_array, y_new)   

        # 4. Remove right side
        # remove all values after this index from both arrays
        maxval = max(y_array)
        highest_y_array_index = np.where(y_array == maxval)
        highest_y_array_index = highest_y_array_index[0][0]

        # our arrays are based off a transposed iteration, so they are backwards
        x_array = x_array[highest_y_array_index:]
        y_array = y_array[highest_y_array_index:]

        array_of_matrix.append(new_matrix)
        array_of_left_x_array.append(x_array)
        array_of_left_y_array.append(y_array)

        img_index+=1

    # Create single line graph for rotated images
    img_index=0
    for rotated_img in all_rotated_images:
        x_array = []
        y_array = []
        matrix2 = np.array(rotated_img)
        dimensions = np.shape(matrix2)
        img_filename = all_img_names[img_index]

        # create list of graph points from rotated image
        x=len(matrix2.transpose())
        for column in reversed(matrix2.transpose()):
            #loop until we hit our first pixel
            j = 0
            y=len(column)
            for p in column:
                if p == 255:
                    #we found a pixel
                    x_array.append(x)
                    y_array.append(y)
                    break
                j=j+1
                y=y-1
            x=x-1

        # 4. Use first 20 points only
        middle_index = math.floor(len(x_array) - 20)
        x_array = x_array[middle_index:]
        y_array = y_array[middle_index:]

        array_of_right_x_array.append(x_array)
        array_of_right_y_array.append(y_array)

        img_index+=1

    return array_of_matrix, array_of_left_x_array, array_of_left_y_array, all_img_names, center_colors, array_of_right_x_array, array_of_right_y_array

def calculate_values(array_of_left_x_array,array_of_left_y_array,data_file_name, img_names, center_colors, array_of_right_x_array, array_of_right_y_array):

    img_index=0
    for i, ar in enumerate(array_of_left_x_array):
        img_filename = img_names[img_index]
        save_dir = './output/plots/'+img_filename

        x = array_of_left_x_array[i]
        y = array_of_left_y_array[i]

        x_right = array_of_right_x_array[i]
        y_right = array_of_right_y_array[i]

        # 6. Polynomial calculation
        poly_curve_size = 7

        z = np.polyfit(x, y, poly_curve_size)
        f = np.poly1d(z)

        ## print z values to a data file with labels (do not add the last one which is the y intercept)
        for index1,point in enumerate(z):
            print(z[index1],",", end="", sep="", file=open(data_file_name, 'a'))

        z_right = np.polyfit(x_right, y_right, poly_curve_size)
        for index1,point in enumerate(z_right):
            print(z_right[index1],",", end="", sep="", file=open(data_file_name, 'a'))

        # print center color feature value RGB
        print(center_colors[i][0],",", end="", sep="", file=open(data_file_name, 'a'))
        print(center_colors[i][1],",", end="", sep="", file=open(data_file_name, 'a'))
        print(center_colors[i][2],",", end="", sep="", file=open(data_file_name, 'a'))
        
        # print height width ratio
        height = max(y) - min(y)
        width = max(x) - min(x)
        ratio = height/width
        print(ratio,",", end="", sep="", file=open(data_file_name, 'a'))

        # print class label
        print(animal[0], file=open(data_file_name, 'a'))

        img_index+=1

if __name__ == '__main__':
    array_of_img_dirs = []

    array_of_img_dirs.append('images/'+animal+'s_cropped/set/succeeded_in_dot_17')
    array_of_img_dirs.append('images/'+animal+'s_cropped/set/succeeded_in_dot_3')
    array_of_img_dirs.append('images/'+animal+'s_cropped/set/succeeded_in_dot_6')
    
    # run images through edge detector and create arrays of xy points
    array_of_matrix, array_of_left_x_array, array_of_left_y_array, img_names, center_colors, array_of_right_x_array, array_of_right_y_array = clean_data(array_of_img_dirs)

    # calculate graph equation values and store in data file
    data_file_name = 'output/'+animal+'_all.data'
    calculate_values(array_of_left_x_array, array_of_left_y_array, data_file_name, img_names, center_colors, array_of_right_x_array, array_of_right_y_array)