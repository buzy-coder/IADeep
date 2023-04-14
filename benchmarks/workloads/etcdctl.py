import etcd3
import os

class ETCD_WRAPER:
    def __init__(self):
        api_server_ip = os.getenv("ETCD_SERVER_IP")
        self.api_server_ip = api_server_ip
        self.domain = "/gpushare"
        pem_prefix= "/workspace/etcd_key/"
        # version_prefix = "/v3"
        self.client = etcd3.client(host=self.api_server_ip, port=os.getenv("ETCD_PORT"), cert_cert=pem_prefix+"/healthcheck-client.crt",cert_key=pem_prefix+"/healthcheck-client.key",ca_cert=pem_prefix+"/ca.crt")

    def get(self, podName, para):
        key = self.domain+"/"+podName+"/"+para
        value,_ =  self.client.get(key)
        if value is None:
            return None
        value = eval(str(value.decode('utf-8')))
        return value

    def put(self,podName, para, value):
        key = self.domain+"/"+podName+"/"+para
        # if type(value) == int or type(value) == list:
        value = str(value)  
        self.client.put(key,value)

    def delete(self, podName, para):
        key = self.domain+"/"+podName+"/"+para
        self.client.delete(key)

    def watch(self, podName, para):
        key = self.domain+"/"+podName+"/"+para
        events_iterator, cancel = self.client.watch(key)
        for event in events_iterator:
            print(event)
            if isinstance(event, etcd3.events.PutEvent):
                cancel()

etcd_wraper = ETCD_WRAPER()