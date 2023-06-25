# copy etcd key to etcd_key
sudo cp /etc/kubernetes/pki/etcd/healthcheck-client.crt ./etcd_key
sudo cp /etc/kubernetes/pki/etcd/healthcheck-client.key ./etcd_key
sudo cp /etc/kubernetes/pki/etcd/ca.crt ./etcd_key

# build image
docker build -t 10.119.46.41:30003/library/iadeep_benchmarks:1.0 .
docker push 10.119.46.41:30003/library/iadeep_benchmarks:1.0