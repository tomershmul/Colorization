import os
from PIL import Image


def main():
    code_folder = os.getcwd()
    os.chdir('../data/imagesFolder/images2')
    image_orig_folder = os.getcwd()
    files = os.listdir(image_orig_folder)
    os.chdir('../imagenet64')
    image_dest_folder = os.getcwd()

    i = 1
    for img_name in files:
        img_path = os.path.join(image_orig_folder, img_name)
        img = Image.open(img_path)
        img_resize = img.resize((64, 64))
        img_name_64 = os.path.splitext(img_name)[0] + '_64.png'
        img_path_64 = os.path.join(image_dest_folder, img_name_64)
        img_resize.save(img_path_64)

        if (i % 2000 == 0):
            print('Saved', i)

        i = i + 1

if __name__ == '__main__':
    main()
