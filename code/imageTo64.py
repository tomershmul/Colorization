import os
from PIL import Image


def main():
    resize = 128
    code_folder = os.getcwd()
    os.chdir('../../../datasets/imagenet_20classes/')
    image_orig_folder = os.getcwd()
    dirs = os.listdir(image_orig_folder)
    os.chdir(code_folder)
    image_dest_folder = ('../../../datasets/imagenet_20classes_{}x{}/'.format(resize,resize))

    i = 1
    for directory in dirs:
        os.mkdir(os.path.join(image_dest_folder, directory))
        files = os.listdir(os.path.join(image_orig_folder,directory))
        for img_name in files:
            img_path = os.path.join(image_orig_folder, directory, img_name)
            img = Image.open(img_path)
            img_resize = img.resize((resize, resize))
            img_name_resize = os.path.splitext(img_name)[0] + '_{}.png'.format(resize)
            img_path_resize = os.path.join(image_dest_folder, directory,img_name_resize)
            try:
                img_resize.save(img_path_resize)
            except Exception as ex:
                print ("can't save image: {}\n{}".format(img_name_resize, ex))

        if (i % 2000 == 0):
            print('Saved', i)

        i = i + 1

if __name__ == '__main__':
    main()
