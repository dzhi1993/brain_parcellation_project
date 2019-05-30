# from nibabel.gifti import giftiio
# import nibabel as nb
# import numpy as np
#
# gifti_image = nb.load("Cond01_No-Go.func.gii")
# # index = nifti_image.get_fdata()
#
# # print(index.shape)
# img_data = [x.data for x in gifti_image.darrays]
# #np_data = np.reshape(img_data, (len(img_data[0]),))
# np_data = np.reshape(img_data, (len(img_data[0]), len(img_data[0][0])))
# print(np_data, np_data.shape)


import csv

txt_file = "Mapping.txt"
csv_file = "mycsv.csv"


# with open(txt_file, 'r') as infile, open(csv_file, 'w') as outfile:
#      stripped = (line.strip() for line in infile)
#      lines = (line.split(",") for line in stripped if line)
#      writer = csv.writer(outfile)
#      writer.writerows(lines)

#import os
import pandas as pd

df = pd.read_csv(txt_file, sep=",")
df.to_csv(csv_file, index=False)
