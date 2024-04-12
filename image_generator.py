import csv
import shutil
from glob import glob
import os

"""
create directories with 'create_directories.sh'
"""

mask_images = sorted(glob(os.path.join('data/masks', '*.png')))
file_names = [file[13:-4] for file in mask_images]

src_dir_imgs = "data/r_imgs/"
src_dir_mask = "data/masks/"

def read_csv_as_list(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return data[0]


train = read_csv_as_list('lists/train.csv')
test = read_csv_as_list('lists/test.csv')
validation = read_csv_as_list('lists/validation.csv')



for image in file_names:
    
    img_filename = src_dir_imgs + "R_" + image + ".tif"
    msk_filename = src_dir_mask + "M_" + image + ".png"
    
    dest_img = "imgs/" + "R_" + image + ".tif"
    dest_msk = "mask/" + "M_" + image + ".png"

    if image in test:
        dest_path = "workdata/test/"
        
        shutil.copy(img_filename, dest_path + dest_img)
        shutil.copy(msk_filename, dest_path + dest_msk)
        
    elif image in validation:
        dest_path = "workdata/validation/"
        
        shutil.copy(img_filename, dest_path + dest_img)
        shutil.copy(msk_filename, dest_path + dest_msk)
