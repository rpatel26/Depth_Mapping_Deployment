#import the necessary modules
import freenect
import cv2
import numpy as np
import sys
import os, errno

path = "/home/pib/Depth_Mapping_Deployment/Machine_Learning/dataset"
#function to get RGB image from kinect
def get_video():
    array,_ = freenect.sync_get_video()
    array = cv2.cvtColor(array,cv2.COLOR_RGB2BGR)
    return array

#function to get depth image from kinect
def get_depth():
    array,_ = freenect.sync_get_depth()
    array = array.astype(np.uint8)
    return array

if __name__ == "__main__":
    directory = input("Name of object: ")
    num_pic = int(input("Number of pictures: "))
    #get a frame from RGB camera
    #get a frame from depth sensor
    #depth = get_depth()
    #display RGB image
    #cv2.imshow('RGB image',frame)
    #display depth image
    #cv2.imshow('Depth image',depth)
    filename = "{}/{}".format(path, directory)
    print(filename)
    try:
        os.mkdir(filename)
        print(filename)
    except OSError as e:
        print("error")
        if e.errno != errno.EEXIST:
            raise
    for i in range(num_pic):
        frame = get_video()
        cv2.imwrite("{}/{}_{}.jpg".format(filename, directory, num_pic), frame)
        # quit program when 'esc' key is pressed
#    k = cv2.waitKey(5) & 0xFF
#    if k == 27:
#        break
#cv2.destroyAllWindows()



