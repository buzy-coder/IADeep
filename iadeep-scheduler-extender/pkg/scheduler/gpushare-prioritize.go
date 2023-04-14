package scheduler

import (
	"gpushare-scheduler-extender/pkg/cache"
	"gpushare-scheduler-extender/pkg/methods"
	"gpushare-scheduler-extender/pkg/utils"
	"log"
	"strconv"

	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/kubernetes"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
)

func NewGPUsharePrioritize(clientset *kubernetes.Clientset, c *cache.SchedulerCache) *Prioritize {
	log.Printf("NewGPUsharePrioritize: %v", clientset)
	return &Prioritize{
		Name: "gpusharingprioritize",
		Func: func(_ *v1.Pod, nodeName string, c *cache.SchedulerCache) (schedulerapi.HostPriority, error) {
			var hostPriority schedulerapi.HostPriority
			node, _ := c.GetNode(nodeName)
			sumGPUMemUtil := 0
			GPUs := utils.GetGPUStatusInNode(node)
			hasNonTuningGPU := false
			for idx, status := range GPUs {
				log.Printf("debug: %v GPU %v usage %v", nodeName, idx, status.MemUtil)
				sumGPUMemUtil += int(status.MemUtil)
				if !methods.GetTuningStatusFromEtcd(nodeName, strconv.Itoa(idx)) {
					hasNonTuningGPU = true
				}
			}
			avgGPUMemUtil := sumGPUMemUtil / len(GPUs)
			log.Printf("debug: %v avg GPU usage %v", nodeName, avgGPUMemUtil)

			var score int
			if hasNonTuningGPU {
				score = 100 - avgGPUMemUtil
			} else {
				score = 0
			}

			hostPriority = schedulerapi.HostPriority{
				Host:  nodeName,
				Score: score,
			}
			log.Println(hostPriority)
			return hostPriority, nil
		},
		cache: c,
	}
}
