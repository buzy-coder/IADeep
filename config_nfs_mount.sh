# use nfs to mount data path for training
# https://www.digitalocean.com/community/tutorials/how-to-set-up-an-nfs-mount-on-ubuntu-20-04

# 1.on master node
sudo apt-get install nfs-kernel-server
sudo cat >> /etc/exports << EOF
/nfs/dataset *(rw,sync,no_wdelay,nohide,no_subtree_check,no_root_squash)
/nfs/cache *(rw,sync,no_wdelay,nohide,no_subtree_check,no_root_squash)
EOF

# 2.On each worker node 
sudo apt install nfs-common
sudo mkdir /nfs
sudo mkdir /nfs/dataset
sudo mkdir /nfs/cache
sudo mount ${master_node_ip}:/nfs/dataset /nfs/dataset
sudo mount ${master_node_ip}:/nfs/cache /nfs/cache