import os, glob
import pandas as pd

dir = '/home/gakusei/PycharmProjects/HAR_DRNN/kinect/non_sequence/*.csv'
book_dict = {}
#print(glob.glob(dir))
for file_path in glob.glob(dir):
    file_name = os.path.basename(file_path)
    print(file_name)
    book_dict[file_name] = pd.read_csv(file_path,header=None).iloc[1:,2:5]

print (book_dict)