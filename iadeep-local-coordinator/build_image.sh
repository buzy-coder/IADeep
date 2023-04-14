# copy etcd key to etcd_key
cp /etc/kubernetes/pki/apiserver-etcd-client.crt ./etcd_key
cp /etc/kubernetes/pki/apiserver-etcd-client.key ./etcd_key
cp /etc/kubernetes/pki/etcd/ca.crt ./etcd_key

# download libraries
go mod init iadeep-local-coordinator
go mod vendor
go mod tidy

# build image
docker build -t 10.119.46.41:30003/library/iadeep-local-coordinator:1.0 .
docker push 10.119.46.41:30003/library/iadeep-local-coordinator:1.0
