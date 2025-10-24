#Download the datasets for tuning. These datasets are downloaded from OpenOOD (https://github.com/Jingkang50/OpenOOD). 

rm -rf ${DATASETS}/NINCO/
rm -rf ${DATASETS}/MNIST/
rm -rf ${DATASETS}/imagenet-r/

gdown https://drive.google.com/uc?id=1Z82cmvIB0eghTehxOGP5VTdLt7OD3nk6 -O ${DATASETS}/ninco.zip
gdown https://drive.google.com/uc?id=1CCHAGWqA1KJTFFswuF9cbhmB-j98Y1Sb -O ${DATASETS}/mnist.zip
gdown https://drive.google.com/uc?id=1EzjMN2gq-bVV7lg-MEAdeuBuz-7jbGYU -O ${DATASETS}/in-r.zip

mkdir -p ${DATASETS}/NINCO/images
mkdir -p ${DATASETS}/MNIST/images

unzip -q ${DATASETS}/ninco.zip -d ${DATASETS}/ninco_tmp
unzip -q ${DATASETS}/mnist.zip -d ${DATASETS}/mnist_tmp
unzip -q ${DATASETS}/in-r.zip -d ${DATASETS}/imagenet-r

rm -rf ${DATASETS}/ninco_tmp/f_field_road/
rm -rf ${DATASETS}/ninco_tmp/f_forest_path/
rm -rf ${DATASETS}/ninco_tmp/s_sky/


mv ${DATASETS}/mnist_tmp/test/* ${DATASETS}/MNIST/images
mv ${DATASETS}/ninco_tmp/*/* ${DATASETS}/NINCO/images

rm -rf ${DATASETS}/ninco_tmp/
rm -rf ${DATASETS}/mnist_tmp/