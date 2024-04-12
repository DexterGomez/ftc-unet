from PIL import Image, ImageOps
from glob import glob
import os
import csv


#get directories
resized_images = sorted(glob(os.path.join('data/r_imgs', '*.tif')))
mask_images = sorted(glob(os.path.join('data/masks', '*.png')))

#get list of validation images
def read_csv_as_list(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return data[0]

validation_images = read_csv_as_list('lists/validation.csv') + read_csv_as_list('lists/test.csv')

def transform_imgs(img):
    transformations = [
        img,
        ImageOps.flip(img), #vertical flip
        ImageOps.mirror(img), #horizonal flip
        img.rotate(15),
        img.rotate(30),
        img.rotate(45),
        img.rotate(60),
        img.rotate(75),
        ImageOps.flip(img.rotate(15)),
        ImageOps.flip(img.rotate(30)),
        ImageOps.flip(img.rotate(45)),
        ImageOps.flip(img.rotate(60)),
        ImageOps.flip(img.rotate(75)),
        ImageOps.mirror(img.rotate(15)),
        ImageOps.mirror(img.rotate(30)),
        ImageOps.mirror(img.rotate(45)),
        ImageOps.mirror(img.rotate(60)),
        ImageOps.mirror(img.rotate(75)),
    ]
    
    return transformations

print("Starting image augmentation\n")

for image_index in range(len(resized_images)):
    if resized_images[image_index][14:-4] in validation_images:
        continue

    #generate augmented imafe of i image filename
    #image = imread(resized_images[image_index])/255
    image = Image.open(resized_images[image_index])
    augmented = transform_imgs(image)
    
    #Save the augmented images
    for augmented_index in range(len(augmented)):

        new_filename = 'workdata/train/imgs/{}_{:02d}.tif'.format(resized_images[image_index][12:-4], augmented_index)
        #imsave(new_filename, augmented[augmented_index])

        #img = Image.fromarray(augmented[augmented_index])
        #img.save(new_filename)
        augmented[augmented_index].save(new_filename)

print("Image augmentation done\nStarting mask augmentation...\n")

for masks_index in range(len(mask_images)):
    if mask_images[masks_index][13:-4] in validation_images:
        continue

    #mask = imread(mask_images[masks_index])
    mask = Image.open(mask_images[masks_index])
    augmented = transform_imgs(mask)

    for augmented_index in range(len(augmented)):
        new_filename = 'workdata/train/mask/{}_{:02d}.png'.format(mask_images[masks_index][11:-4], augmented_index)
        #imsave(new_filename, augmented[augmented_index])
        
        #img = Image.fromarray(augmented[augmented_index])
        #img.save(new_filename)

        augmented[augmented_index].save(new_filename)

print("Masks augmentation done\n")