package scheduler

import (
	"fmt"
	"log"

	"gpushare-scheduler-extender/pkg/cache"
	"gpushare-scheduler-extender/pkg/utils"

	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/kubernetes"
)

func NewGPUsharePredicate(clientset *kubernetes.Clientset, c *cache.SchedulerCache) *Predicate {
	log.Println("NewGPUsharePredicate")
	return &Predicate{
		Name: "gpusharingfilter",
		Func: func(pod *v1.Pod, nodeName string, c *cache.SchedulerCache) (bool, error) {
			log.Printf("info: check if the pod name %s can be scheduled on node %s", pod.Name, nodeName)
			nodeInfo, err := c.GetNodeInfo(nodeName)
			if err != nil {
				return false, err
			}
			n := nodeInfo.GetNode()
			log.Printf("nodeLabels are: %+v", n.Labels)
			log.Printf("node %v is GPUSharingNode %v", n.Name, utils.IsGPUSharingNode(n))
			if !utils.IsGPUSharingNode(n) {
				return false, fmt.Errorf("The node %s is not for GPU share, need skip", nodeName)
			}

			allocatable := nodeInfo.Assume(pod)
			if !allocatable {
				return false, fmt.Errorf("Insufficient GPU Memory in one device")
			} else {
				log.Printf("info: The pod %s in the namespace %s can be scheduled on %s",
					pod.Name,
					pod.Namespace,
					nodeName)
			}
			return true, nil
		},
		cache: c,
	}
}
