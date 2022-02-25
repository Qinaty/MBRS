import os, random, shutil

def Choose(source_dir, target_dir):
    source_images = os.listdir(source_dir)
    sample_images = random.sample(source_images, 100)
    for image in sample_images:
        shutil.move(os.path.join(source_dir, image), target_dir)


if __name__=='__main__':
    source_dir = "../../datasets/coco2014/test2014"
    target_dir = "../datasets_test_100/validation"
    Choose(source_dir, target_dir)
