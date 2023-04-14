# copy etcd key to etcd_key
cp /etc/kubernetes/pki/apiserver-etcd-client.crt ./etcd_key
cp /etc/kubernetes/pki/apiserver-etcd-client.key ./etcd_key
cp /etc/kubernetes/pki/etcd/ca.crt ./etcd_key

# build image
docker build -t 10.119.46.41:30003/library/iadeep_benchmarks:1.0 .
docker push 10.119.46.41:30003/library/iadeep_benchmarks:1.0