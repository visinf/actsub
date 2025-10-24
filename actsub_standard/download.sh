#Download the datasets for evaluation. 
wget -P ${DATASETS} http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz
wget -P ${DATASETS} http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/SUN.tar.gz
wget -P ${DATASETS} http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/Places.tar.gz
wget -P ${DATASETS} https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
wget -P ${DATASETS} https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
wget -P ${DATASETS} https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
wget -P ${DATASETS} https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
wget -P ${DATASETS} https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz
wget -P ${DATASETS} http://data.csail.mit.edu/places/places365/test_256.tar

#Download the datasets for tuning. These datasets are downloaded from OpenOOD (https://github.com/Jingkang50/OpenOOD). 

gdown https://drive.google.com/uc?id=1Z82cmvIB0eghTehxOGP5VTdLt7OD3nk6 -O ${DATASETS}/ninco.zip
gdown https://drive.google.com/uc?id=1CCHAGWqA1KJTFFswuF9cbhmB-j98Y1Sb -O ${DATASETS}/mnist.zip
gdown https://drive.google.com/uc?id=1EzjMN2gq-bVV7lg-MEAdeuBuz-7jbGYU -O ${DATASETS}/in-r.zip


#Create subfolders for the datasets and set definitions for the directories.

IMAGENET_DATASET_DIR="${DATASETS}/imagenet/val"
PLACES365_DATASET_DIR="${DATASETS}/Places365"
mkdir -p ${IMAGENET_DATASET_DIR}
mkdir -p ${PLACES365_DATASET_DIR}
mkdir -p ${DATASETS}/Textures/images
mkdir -p ${DATASETS}/NINCO/images
mkdir -p ${DATASETS}/MNIST/images

#Extract the datasets into the specified folders.

tar -xf ${DATASETS}/cifar-10-python.tar.gz -C ${DATASETS}
tar -xf ${DATASETS}/cifar-100-python.tar.gz -C ${DATASETS}
tar -xf ${DATASETS}/iSUN.tar.gz -C ${DATASETS}
tar -xf ${DATASETS}/iNaturalist.tar.gz -C ${DATASETS}
tar -xf ${DATASETS}/SUN.tar.gz -C ${DATASETS}
tar -xf ${DATASETS}/Places.tar.gz -C ${DATASETS}
tar -xf ${DATASETS}/dtd-r1.0.1.tar.gz -C ${DATASETS}
tar -xf ${DATASETS}/test_256.tar -C ${PLACES365_DATASET_DIR}
tar -xf ${DATASETS}/ILSVRC2012_img_val.tar -C ${IMAGENET_DATASET_DIR}
unzip -q ${DATASETS}/ninco.zip -d ${DATASETS}/ninco_tmp
unzip -q ${DATASETS}/mnist.zip -d ${DATASETS}/mnist_tmp
unzip -q ${DATASETS}/in-r.zip -d ${DATASETS}/imagenet-r



#Modify the data structure to match with the dataloader. 

mv ${DATASETS}/dtd/images/*/* ${DATASETS}/Textures/images
mv ${PLACES365_DATASET_DIR}/test_256  ${PLACES365_DATASET_DIR}/images
mv ${DATASETS}/iSUN/iSUN_patches ${DATASETS}/iSUN/images
mv ${DATASETS}/mnist_tmp/test/* ${DATASETS}/MNIST/images

#From the NINCO dataset, we remove the categories that overlap with Places dataset.
rm -rf ${DATASETS}/ninco_tmp/f_field_road/
rm -rf ${DATASETS}/ninco_tmp/f_forest_path/
rm -rf ${DATASETS}/ninco_tmp/s_sky/

mv ${DATASETS}/ninco_tmp/*/* ${DATASETS}/NINCO/images

#Remove leftover temporary files.

rm -rf ${DATASETS}/ninco_tmp/
rm -rf ${DATASETS}/dtd/
rm -rf ${DATASETS}/mnist_tmp/

#Download and extract CIFAR checkpoints.

wget -P ${DATASETS} https://www.dropbox.com/s/o5r3t3f0uiqdmpm/checkpoints.zip
unzip -j ${DATASETS}/checkpoints.zip -d ${DATASETS}/checkpoints_tmp
mv ${DATASETS}/checkpoints_tmp/densenet100_cifar10.pth checkpoints/densenet100_cifar10.pth
mv ${DATASETS}/checkpoints_tmp/densenet100_cifar100.pth checkpoints/densenet100_cifar100.pth

rm -rf ${DATASETS}/checkpoints_tmp

