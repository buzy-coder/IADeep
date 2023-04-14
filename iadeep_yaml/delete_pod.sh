kubectl delete pod $(kubectl get pod | grep -vE "gpushare|coordinator|tuner|NAME" | awk '{print $1}')
