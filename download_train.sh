#Download and extact the train split of ImageNet-1k.
wget -P ${DATASETS} https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
mkdir -p ${DATASETS}/imagenet/train
tar -xf ${DATASETS}/ILSVRC2012_img_train.tar -C ${DATASETS}/imagenet/train
cd ${DATASETS}/imagenet/train
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done