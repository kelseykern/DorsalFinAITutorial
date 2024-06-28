import canny_edge_detector as ced # credit to https://github.com/FienSoP/
from utils import utils
from matplotlib import pyplot as plt
import numpy as np


def clean_images():

    # Step 1: Edge Detect
    dir_name = 'images/sharks_cropped/set/succeeded_in_dot_17'
    imgs_final = []
    img_names = []

    raw_imgs, imgs, gray_rotated_imgs, img_names_array  = utils.load_data(dir_name)
    img_names.extend(img_names_array)

    detector = ced.cannyEdgeDetector(imgs, sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=.17, weak_pixel=100)
    imgs_final.extend(detector.detect())

    img_index=0
    for img in imgs_final:
        img_filename = img_names[img_index]
        save_dir = './output/plots/'+img_filename

        matrix = np.array(img)

        # Step 2: Create list of graph points
        x_array = []
        y_array = []
        i = 0
        x=len(matrix.transpose())
        for column in reversed(matrix.transpose()):
            # loop until we hit our first pixel
            j = 0
            y = len(column)
            for p in column:
                if p == 255:
                    # we found a pixel
                    x_array.append(x)
                    y_array.append(y)
                    break
                j=j+1
                y=y-1
            i=i+1
            x=x-1

        # Step 3: Detrend the data
        y_new = []
        first_x = x_array[0]
        first_y = y_array[0]
        last_x = x_array[len(x_array)-1]
        last_y = y_array[len(y_array)-1]
        for i, val in enumerate(y_array):
            y_new.append(
                ( (last_y - first_y) / (last_x - first_x) ) * (x_array[i]-last_x) + last_y
                )
        #plt.plot(x_array,y_new,'ro') # red line
        #plt.plot(x_array,y_array,'bo') # original
        # apply detrending
        y_array = np.subtract(y_array,y_new)

        # Step 4: Remove the right side
        maxval = max(y_array)
        highest_y_array_index = np.where(y_array == maxval)
        highest_y_array_index = highest_y_array_index[0][0]

        # our arrays are based off a transpose iteration, so they ar backwards
        x_array = x_array[highest_y_array_index:]
        y_array = y_array[highest_y_array_index:]

        # Step 5: Calculate the polynomial
        # polyfit ( x-coordinate value of the input array,
        #           y-coordinate value of the input array,
        #           integer value that fits the polynomial degree,
        #           optionalParams )
        poly_curve_size = 7
        z = np.polyfit(x_array, y_array, poly_curve_size)
        f = np.poly1d(z)
        x_new = np.linspace(x_array[0], x_array[-1],50)
        y_new = f(x_new)
        
        # create data file of polynomial values
        animal = 's'
        output_file_name = "output/sharks_curve_7_points.data"
        for i, point in enumerate(z):
            print(z[i],',',end='',sep='',file=open(output_file_name, 'a'))
        print(animal, file=open(output_file_name,'a'))

        plt.plot(x_array,y_array,'o',x_new,y_new)
        #plt.imshow(img[::-1],origin='lower')
        #plt.plot(x_array,y_array,'bo')
        plt.savefig(save_dir)
        plt.clf()
        
        img_index+=1


def print_matrix(matrix, filename):
    np.set_printoptions(threshold=np.inf)
    sourceFile = open('output/matrix_pixels.txt', 'w')
    for row in matrix:
        for val in row:
            print(val, file=sourceFile, end='')
        print('', file = sourceFile)
    sourceFile.close()

if __name__ == '__main__':
    clean_images()