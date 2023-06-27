sudo mkdir /nfs
sudo mkdir /nfs/dataset
sudo mkdir /nfs/cache

cd /nfs/dataset
# CIFAR10
wget -c https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -zxvf cifar-10-python.tar.gz
# CIFAR100
wget -c https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -zxvf cifar-100-python.tar.gz
# MovieLens
wget -c http://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip
git clone https://github.com/hexiangnan/adversarial_personalized_ranking.git
mv adversarial_personalized_ranking/Data/* ml-1m/

# REDDIT-MULTI-12K
wget -c https://www.chrsmrrs.com/graphkerneldatasets/REDDIT-MULTI-12K.zip
unzip REDDIT-MULTI-12K.zip

# Wikitext-2
wget -c https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
unzip wikitext-2-v1.zip

# SQuAD2
wget -c https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json

# COCO
wget -c http://images.cocodataset.org/zips/train2017.zip
wget -c http://images.cocodataset.org/zips/val2017.zip
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip

# ImageNet
wget -c http://image-net.org/image/ilsvrc2012/ILSVRC2012_img_train.tar
wget -c http://image-net.org/image/ilsvrc2012/ILSVRC2012_img_val.tar
tar -zxvf ILSVRC2012_img_train.tar
tar -zxvf ILSVRC2012_img_val.tar


# cache
cd /nfs/cache
# http://nlp.uoregon.edu/download/embeddings/
wget http://nlp.uoregon.edu/download/embeddings/glove.6B.100d.txt
wget http://nlp.uoregon.edu/download/embeddings/glove.6B.200d.txt
wget http://nlp.uoregon.edu/download/embeddings/glove.6B.300d.txt
wget http://nlp.uoregon.edu/download/embeddings/glove.6B.50d.txt