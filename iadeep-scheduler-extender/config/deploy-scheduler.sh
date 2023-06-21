# cp ./scheduler-config.yaml /etc/kubernetes/
# cp ./iadeep-scheduler.yaml /etc/kubernetes/manifests/
# systemctl restart kubelet

sudo rm /etc/kubernetes/manifests/iadeep-scheduler.yaml
sudo rm /etc/kubernetes/iadeep-scheduler-policy-config.yaml
kubectl delete -f iadeep-schd-extender.yaml
sudo cp ./iadeep-scheduler-policy-config.yaml /etc/kubernetes/
sudo cp ./iadeep-scheduler.yaml /etc/kubernetes/manifests/
kubectl create -f iadeep-schd-extender.yaml
sudo systemctl restart kubelet