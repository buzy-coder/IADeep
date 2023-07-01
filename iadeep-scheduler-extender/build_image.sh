# copy etcd key to etcd_key
sudo cp /etc/kubernetes/pki/etcd/healthcheck-client.crt ./etcd_key
sudo cp /etc/kubernetes/pki/etcd/healthcheck-client.key ./etcd_key
sudo cp /etc/kubernetes/pki/etcd/ca.crt ./etcd_key

# build image
docker build -t www.myharbor.com/library/gpushare-scheduler-extender:1.0 .
docker push www.myharbor.com/library/gpushare-scheduler-extender:1.0
