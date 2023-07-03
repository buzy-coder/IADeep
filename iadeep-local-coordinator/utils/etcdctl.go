package utils

import (
	"bytes"
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/binary"
	"encoding/json"
	"io/ioutil"
	"log"
	"os"
	"strconv"
	"strings"
	"time"

	"go.etcd.io/etcd/clientv3"
	"k8s.io/client-go/util/workqueue"
)

type TuningPods struct {
	DevId      int
	PodNames   []string
	Batchsizes []int
	Count      int
}

// create an etcd client
func CreateEtcdClient() (*clientv3.Client, error) {
	pem_prefix := os.Getenv("ETCD_KEY")
	cert_cert := pem_prefix + "healthcheck-client.crt"
	cert_key := pem_prefix + "healthcheck-client.key"
	ca_cert := pem_prefix + "ca.crt"
	cert, err := tls.LoadX509KeyPair(cert_cert, cert_key)
	if err != nil {
		return nil, err
	}

	caData, err := ioutil.ReadFile(ca_cert)
	if err != nil {
		return nil, err
	}

	pool := x509.NewCertPool()
	pool.AppendCertsFromPEM(caData)

	_tlsConfig := &tls.Config{
		Certificates: []tls.Certificate{cert},
		RootCAs:      pool,
	}
	etcd_server_ip := os.Getenv("ETCD_SERVER_IP")
	etcd_port := os.Getenv("ETCD_PORT")
	client, err := clientv3.New(clientv3.Config{
		Endpoints:   []string{"https://" + etcd_server_ip + ":" + etcd_port},
		DialTimeout: 20 * time.Second,
		TLS:         _tlsConfig,
	})
	if err != nil {
		log.Printf("error: get connect to etcd err due to %+v", err)
		return nil, err
	}

	return client, err
}

// put content to etcd
func PutContentToEtcd(first_key, second_key, content string) (bool, error) {
	client, err := CreateEtcdClient()
	if err != nil {
		log.Printf("error: get connect to etcd err due to %+v", err)
	}
	defer client.Close()

	ctx, cancel := context.WithTimeout(context.TODO(), 10*time.Second)
	_, err = client.Put(ctx, "/gpushare/"+first_key+"/"+second_key, content)
	cancel()
	if err != nil {
		log.Printf("Put key of %+v failed", first_key+"/"+second_key)
		return false, err
	}
	return true, nil
}

// get content from etcd
func GetContentFromEtcd(first_key, second_key string) (string, error) {
	key := "/gpushare/" + first_key + "/" + second_key
	// log.Printf("GetContentFromEtcd key is: %+v", key)
	client, err := CreateEtcdClient()
	if err != nil {
		log.Printf("error: get connect to etcd err due to %+v", err)
	}
	defer client.Close()
	// ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	resp, err := client.Get(context.TODO(), key)
	// cancel()
	if err != nil {
		log.Printf("Get value of key %+v failed", key)
		return "", err
	}
	// log.Printf("Get value %+v of key %+v", resp.Kvs, key)
	if len(resp.Kvs) > 0 {
		for _, kv := range resp.Kvs {
			// log.Printf("key:%v, value:%+v", kv.Key, kv.Value)
			return string(kv.Value), nil
		}
	}
	return "", err
}

// watch key status
func WatchKeyFromEtcd(first_key, second_key string) (string, error) {
	// event type 0:PUT 1:Delete 2:Expire
	second_key = "/gpushare/" + first_key + "/" + second_key
	client, err := CreateEtcdClient()
	if err != nil {
		log.Printf("error: get connect to etcd err due to %+v", err)
	}
	defer client.Close()
	// Process still stuck here sometime, cannot reproduce the bug
	// try to solve this problem by adding a timeout to watch function
	for {
		// ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
		ctx, cancel := context.WithCancel(context.Background())
		rch := client.Watch(ctx, second_key)
		defer cancel()
		select {
		case <-ctx.Done():
			log.Printf("error: %v", ctx.Err())
			return "", ctx.Err()
		case res := <-rch:
			for _, ev := range res.Events {
				log.Printf("%s %q :%q\n", ev.Type, ev.Kv.Key, ev.Kv.Value)
				log.Printf("ev.Type == clientv3.EventTypePut: %+v", ev.Type == clientv3.EventTypePut)
				if ev.Type == clientv3.EventTypePut {
					return string(ev.Kv.Value), nil
				}
			}
		}
	}
}

// delete content by key
func DeleteContentFromEtcd(first_key, second_key string) (bool, error) {
	key := "/gpushare/" + first_key + "/" + second_key
	client, err := CreateEtcdClient()
	if err != nil {
		log.Printf("error: get connect to etcd err due to %+v", err)
	}
	defer client.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)

	_, err = client.Delete(ctx, key)
	cancel()
	if err != nil {
		log.Printf("Delete key %+v failed", key)
		return false, err
	}
	return true, nil
}

// watch prefix
func WatchPrefixByEtcd(prefix string, tuningQueue workqueue.RateLimitingInterface) {
	prefix = "/gpushare/" + prefix
	client, err := CreateEtcdClient()
	if err != nil {
		log.Printf("error: get connect to etcd err due to %+v", err)
	}
	defer client.Close()

	rch := client.Watch(context.Background(), prefix, clientv3.WithPrefix())
	log.Printf("watching prefix:%s now...", prefix)
	for {
		for wresp := range rch {
			for _, ev := range wresp.Events {
				log.Printf("%s %q :%q\n", ev.Type, ev.Kv.Key, ev.Kv.Value)
				if ev.Type == clientv3.EventTypePut {
					key, pod_names_str := string(ev.Kv.Key), ev.Kv.Value
					devId, _ := strconv.Atoi(strings.Split(key, "/")[2])
					var pod_names_arr []string
					err := json.Unmarshal([]byte(pod_names_str), &pod_names_arr)
					log.Printf("pod_names_arr is: %+v", pod_names_arr)
					if err == nil {
						tuningPods := &TuningPods{
							DevId:      devId,
							PodNames:   pod_names_arr,
							Batchsizes: []int{},
							Count:      0,
						}
						tuningQueue.Add(tuningPods)
					} else {
						log.Printf("Error: unmarshal pod names string err due to %+v", err)
					}
					// watch tuning tag
					// for {
					// 	tuning, err := GetContentFromEtcd(prefix, string(devId))
					// 	if err != nil {
					// 		log.Printf("Error: Get tuning tag err due to %+v", err)
					// 	} else {
					// 		if tuning == "false" || tuning == "" {
					// 			tuningQueue.Add(tuningPods)
					// 			break
					// 		}
					// 	}
					// }
				}
			}
		}
	}

}

func BytesToInt(b []byte) int {

	bytesBuffer := bytes.NewBuffer(b)
	var x int32
	binary.Read(bytesBuffer, binary.BigEndian, &x)
	return int(x)

}
