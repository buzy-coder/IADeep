package cache

import (
	"log"
	"sync"

	"gpushare-scheduler-extender/pkg/utils"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
)

type DeviceInfo struct {
	Node   *v1.Node
	Idx    int
	PodMap map[types.UID]*v1.Pod
	// usedGPUMem  uint
	TotalGPUMem uint
	Rwmu        *sync.RWMutex
	Tuning      bool
}

func (d *DeviceInfo) GetPods() []*v1.Pod {
	pods := []*v1.Pod{}
	for _, pod := range d.PodMap {
		log.Printf("debug: getdevinfo %v on dev %v", pod.Name, d.Idx)
		pods = append(pods, pod)
	}
	return pods
}

func newDeviceInfo(n *v1.Node, index int, totalGPUMem uint) *DeviceInfo {
	return &DeviceInfo{
		Node:        n,
		Idx:         index,
		TotalGPUMem: totalGPUMem,
		PodMap:      map[types.UID]*v1.Pod{},
		Rwmu:        new(sync.RWMutex),
		Tuning:      false,
	}
}

func (d *DeviceInfo) GetTotalGPUMemory() uint {
	return d.TotalGPUMem
}

func (d *DeviceInfo) GetUsedGPUMemory() (gpuMem uint) {
	// log.Printf("debug: GetUsedGPUMemory() podMap %v, and its address is %p", d.podMap, d)
	d.Rwmu.RLock()
	defer d.Rwmu.RUnlock()
	for _, pod := range d.PodMap {
		if pod.Status.Phase == v1.PodSucceeded || pod.Status.Phase == v1.PodFailed {
			log.Printf("debug: skip the pod %s in ns %s due to its status is %s", pod.Name, pod.Namespace, pod.Status.Phase)
			continue
		}
		// gpuMem += utils.GetGPUMemoryFromPodEnv(pod)
		gpuMem += utils.GetGPUMemoryFromPodAnnotation(pod)
	}
	return gpuMem
}

func (d *DeviceInfo) addPod(pod *v1.Pod) {
	log.Printf("debug: dev.addPod() Pod %s in ns %s with the GPU ID %d will be added to device map",
		pod.Name,
		pod.Namespace,
		d.Idx)
	d.Rwmu.Lock()
	defer d.Rwmu.Unlock()
	d.PodMap[pod.UID] = pod
	log.Printf("debug: success! dev.addPod()")
}

func (d *DeviceInfo) removePod(pod *v1.Pod) {
	log.Printf("debug: dev.removePod() Pod %s in ns %s with the GPU ID %d will be removed from device map",
		pod.Name,
		pod.Namespace,
		d.Idx)
	d.Rwmu.Lock()
	defer d.Rwmu.Unlock()
	delete(d.PodMap, pod.UID)
	// log.Printf("debug: dev.removePod() after updated is %v, and its address is %p",
	// 	d.podMap,
	// 	d)
	log.Printf("debug: success! dev.removePod()")
}
