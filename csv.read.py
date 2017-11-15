import os, glob
import pandas as pd
import numpy as np

dir = '/home/gakusei/PycharmProjects/HAR_DRNN/kinect/non_sequence/*.csv'
book_dict = {}
#print(glob.glob(dir))
for file_path in glob.glob(dir):
    file_name = os.path.basename(file_path)
    print(file_name)
    book_dict[file_name] = np.loadtxt(file_path,delimiter=',',skiprows=1,usecols=(2,3,4))

print (book_dict)