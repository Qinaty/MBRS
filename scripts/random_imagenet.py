import os, random, shutil

def Choose(source_dir, target_dir):
    dir_list = os.listdir(source_dir)
    sample_dir = random.sample(dir_list, 10)
    for directory in sample_dir:
        # print(os.path.join(source_dir, directory))
        images = os.listdir(os.path.join(source_dir, directory))
        sample_images = random.sample(images, 10)
        for image in sample_images:
            shutil.move(os.path.join(source_dir, directory, image), target_dir)

if __name__=='__main__':
    source_dir = "../../datasets/ImageNet/train"
    target_dir = "../datasets_test_100/train"
    Choose(source_dir, target_dir)