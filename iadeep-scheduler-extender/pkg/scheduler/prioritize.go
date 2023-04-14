package scheduler

import (
	"gpushare-scheduler-extender/pkg/cache"
	"log"
	"strings"

	v1 "k8s.io/api/core/v1"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
)

type Prioritize struct {
	Name  string
	Func  func(pod *v1.Pod, nodeName string, c *cache.SchedulerCache) (schedulerapi.HostPriority, error)
	cache *cache.SchedulerCache
}

func (p Prioritize) Handler(args schedulerapi.ExtenderArgs) schedulerapi.HostPriorityList {
	pod := args.Pod
	new_job := strings.Split(pod.Name, "-")[0]
	log.Printf("debug: prioritize new pod node is: %v", new_job)
	// nodes := *args.Nodes
	nodeNames := *args.NodeNames
	// nodes := args.Nodes.Items
	log.Printf("debug: nodeshandler nodeNames: %+v", nodeNames)

	// var processInfo [][]utils.ProcessInfo
	// var interferences []float64
	var priorityList schedulerapi.HostPriorityList
	// var gpuStatus []utils.GPUStatus

	for _, nodeName := range nodeNames {
		// node, err := p.cache.GetNode(nodeName)
		// if err != nil {
		// 	log.Printf("Error: Failed to get node with nodename %v due to %v", nodeName, err)
		// }
		hostPriority, err := p.Func(pod, nodeName, p.cache)
		if err != nil {
			log.Printf("Error: Failed to init hostPriority due to %v", err)
		}
		priorityList = append(priorityList, hostPriority)

		// 只有一个node，不排优先级
		// gpuStatus = utils.GetGPUStatusInNode(node)
		// log.Printf("debug: gpuStatus is %+v", gpuStatus)

		// gpuProcesses := utils.GetGPUStatusInNode(node)

		// log.Printf("debug: hasProcesses is %v on node: %v", gpuProcesses, nodeName)
		// if len(gpuStatus) == 0 || gpuStatus == nil {
		// 	hostPriority, err := p.Func(pod, nodeName, p.cache)
		// 	if err != nil {
		// 		log.Printf("Error: Failed to init hostPriority due to %v", err)
		// 	}
		// 	priorityList = append(priorityList, hostPriority)
		// } else {
		// 	var base_job []string
		// 	for _, gpustatus := range gpuStatus {
		// 		for _, processes := range gpustatus.Process {
		// 			base_job = append(base_job, processes.Name)
		// 		}
		// 	}
		// 	if len(base_job) > 0 {
		// 		interference := methods.GetInterferenceScore(base_job, new_job)
		// 		log.Printf("debug: get interference %v between base_job %v and new_job %v", interference, base_job, new_job)
		// 		interferences = append(interferences, interference)
		// 	}
		// 	log.Printf("debug: get all interferences are: %v", interferences)
		// 	index := 0
		// 	min_val := 0.0
		// 	if len(interferences) > 0 {
		// 		index, min_val = methods.GetMinValue(interferences)
		// 	}
		// 	log.Printf("debug: index: %v", index)
		// 	log.Printf("debug: min interference is %v", min_val)
		// 	hostPriority := schedulerapi.HostPriority{
		// 		Host:  nodeName,
		// 		Score: int(100 - min_val),
		// 	}
		// 	priorityList = append(priorityList, hostPriority)
		// }
	}

	log.Printf("debug: priorityList %+v", priorityList)
	return priorityList
}
