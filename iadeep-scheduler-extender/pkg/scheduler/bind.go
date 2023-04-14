package scheduler

import (
	"gpushare-scheduler-extender/pkg/cache"
	"log"

	"k8s.io/apimachinery/pkg/types"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
)

// Bind is responsible for binding node and pod
type Bind struct {
	Name  string
	Func  func(podName string, podNamespace string, podUID types.UID, node string, cache *cache.SchedulerCache) error
	cache *cache.SchedulerCache
}

// Handler handles the Bind request
func (b Bind) Handler(args schedulerapi.ExtenderBindingArgs) *schedulerapi.ExtenderBindingResult {
	log.Printf("debug: bindings node %v and pod %v", args.Node, args.PodName)
	err := b.Func(args.PodName, args.PodNamespace, args.PodUID, args.Node, b.cache)
	errMsg := ""
	if err != nil {
		errMsg = err.Error()
	}
	return &schedulerapi.ExtenderBindingResult{
		Error: errMsg,
	}
}
