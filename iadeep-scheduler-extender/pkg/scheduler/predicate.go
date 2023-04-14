package scheduler

import (
	"gpushare-scheduler-extender/pkg/cache"
	"log"

	v1 "k8s.io/api/core/v1"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
)

type Predicate struct {
	Name  string
	Func  func(pod *v1.Pod, nodeName string, c *cache.SchedulerCache) (bool, error)
	cache *cache.SchedulerCache
}

func (p Predicate) Handler(args schedulerapi.ExtenderArgs) *schedulerapi.ExtenderFilterResult {
	pod := args.Pod
	nodeNames := *args.NodeNames
	log.Printf("debug: init Filter nodenames are %v", nodeNames)
	canSchedule := make([]string, 0, len(nodeNames))
	canNotSchedule := make(map[string]string)

	for _, nodeName := range nodeNames {
		result, err := p.Func(pod, nodeName, p.cache)
		if err != nil {
			canNotSchedule[nodeName] = err.Error()
		} else {
			if result {
				canSchedule = append(canSchedule, nodeName)
			}
		}
	}

	result := schedulerapi.ExtenderFilterResult{
		NodeNames:   &canSchedule,
		FailedNodes: canNotSchedule,
		Error:       "",
	}
	log.Printf("debug: Filter nodenames are %v", result)
	return &result
}
