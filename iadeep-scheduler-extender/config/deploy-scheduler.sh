cp ./scheduler-config.yaml /etc/kubernetes/
cp ./iadeep-scheduler.yaml /etc/kubernetes/manifests/
systemctl restart kubelet