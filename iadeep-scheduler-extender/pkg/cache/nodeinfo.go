package cache

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"

	"github.com/go-gota/gota/dataframe"
	"github.com/go-gota/gota/series"
	"k8s.io/apimachinery/pkg/types"

	"gpushare-scheduler-extender/pkg/methods"
	"gpushare-scheduler-extender/pkg/utils"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/util/workqueue"
)

const (
	OptimisticLockErrorMsg = "the object has been modified; please apply your changes to the latest version and try again"
	JobExit                = "-2"
	JobRunning             = "1"
	JobStoped              = "0"
)

// NodeInfo is node level aggregated information.
type NodeInfo struct {
	Name           string
	Node           *v1.Node
	Devs           map[int]*DeviceInfo
	GpuCount       int
	GpuTotalMemory int
	IPAddress      string
	TuningQueue    workqueue.RateLimitingInterface
	Rwmu           *sync.RWMutex
}

type TuningPods struct {
	BasePods []*v1.Pod
	NewPod   *v1.Pod
	Node     *NodeInfo
	DevId    int
}

// Create Node Level
func NewNodeInfo(node *v1.Node, tuningQueue workqueue.RateLimitingInterface) *NodeInfo {
	log.Printf("debug: NewNodeInfo() creates nodeInfo for %s", node.Name)

	devMap := map[int]*DeviceInfo{}
	for i := 0; i < utils.GetGPUCountInNode(node); i++ {
		gpuStatus := utils.GetGPUStatusInNode(node)[i]
		devMemoryCapacity := gpuStatus.MemCapacity
		// devMap[i] = newDeviceInfo(i, uint(utils.GetTotalGPUMemory(node)/utils.GetGPUCountInNode(node)))
		devMap[i] = newDeviceInfo(node, i, uint(devMemoryCapacity))
	}

	if len(devMap) == 0 {
		log.Printf("warn: node %s with nodeinfo %v has no devices",
			node.Name,
			node)
	}

	return &NodeInfo{
		Name:           node.Name,
		Node:           node,
		Devs:           devMap,
		GpuCount:       utils.GetGPUCountInNode(node),
		GpuTotalMemory: utils.GetTotalGPUMemory(node),
		IPAddress:      node.Status.Addresses[0].Address,
		TuningQueue:    tuningQueue,
		Rwmu:           new(sync.RWMutex),
	}
}

// Only update the devices when the length of devs is 0
func (n *NodeInfo) Reset(node *v1.Node) {
	n.GpuCount = utils.GetGPUCountInNode(node)
	n.GpuTotalMemory = utils.GetTotalGPUMemory(node)
	n.Node = node
	if n.GpuCount == 0 {
		log.Printf("warn: Reset for node %s but the gpu count is 0", node.Name)
	}

	if n.GpuTotalMemory == 0 {
		log.Printf("warn: Reset for node %s but the gpu total memory is 0", node.Name)
	}

	if len(n.Devs) == 0 && n.GpuCount > 0 {
		devMap := map[int]*DeviceInfo{}
		gpu_length := utils.GetGPUCountInNode(node)
		log.Printf("gpu_length is: %+v", gpu_length)
		for i := 0; i < utils.GetGPUCountInNode(node); i++ {
			devMap[i] = newDeviceInfo(node, i, uint(n.GpuTotalMemory/n.GpuCount))
		}
		n.Devs = devMap
	}
	log.Printf("info: Reset() update nodeInfo for %s with devs %v", node.Name, n.Devs)
}

func (n *NodeInfo) GetName() string {
	return n.Name
}

func (n *NodeInfo) GetDevs() []*DeviceInfo {
	devs := make([]*DeviceInfo, n.GpuCount)
	for i, dev := range n.Devs {
		devs[i] = dev
	}
	log.Printf("n.devs are: %+v", n.Devs)
	return devs
}

func (n *NodeInfo) GetNode() *v1.Node {
	return n.Node
}

func (n *NodeInfo) GetTotalGPUMemory() int {
	return n.GpuTotalMemory
}

func (n *NodeInfo) GetGPUCount() int {
	return n.GpuCount
}

func (n *NodeInfo) removePod(pod *v1.Pod) {
	n.Rwmu.Lock()
	log.Printf("trace: NodeRWLocker: %v removePod() RLocked", n.Name)
	defer log.Printf("trace: NodeRWLocker: %v removePod() RUnlocked", n.Name)
	defer n.Rwmu.Unlock()

	id := utils.GetGPUIDFromAnnotation(pod)
	if id >= 0 {
		dev, found := n.Devs[id]
		if !found {
			log.Printf("warn: Pod %s in ns %s failed to find the GPU ID %d in node %s", pod.Name, pod.Namespace, id, n.Name)
		} else {
			dev.removePod(pod)
			// remove pod to device of node
			// pod_names, err := methods.GetPodContentByEtcd(n.Name, strconv.Itoa(id))
			// if err != nil {
			// 	log.Printf("Error: Get content err due to %+v", err)
			// }
			// pod_names_arr := strings.Split(pod_names, ",")
			// var pod_names_arr_new []string
			// if len(pod_names_arr) > 0 {
			// 	for _, pod_name := range pod_names_arr {
			// 		if pod_name != pod.Name {
			// 			pod_names_arr_new = append(pod_names_arr_new, pod_name)
			// 		}
			// 	}
			// }
			// pod_names_str_new := strings.Join(pod_names_arr_new, ",")
			// methods.PutPodContentByEtcd(n.Name, strconv.Itoa(id), pod_names_str_new)
		}
	} else {
		log.Printf("warn: Pod %s in ns %s is not set the GPU ID %d in node %s", pod.Name, pod.Namespace, id, n.Name)
	}
}

// Add the Pod which has the GPU id to the node
func (n *NodeInfo) addOrUpdatePod(pod *v1.Pod) (added bool) {
	n.Rwmu.Lock()
	log.Printf("trace: NodeRWLocker: %v addOrUpdatePod() RLocked", n.Name)
	defer log.Printf("trace: NodeRWLocker: %v addOrUpdatePod() RUnlocked", n.Name)
	defer n.Rwmu.Unlock()

	id := utils.GetGPUIDFromAnnotation(pod)
	log.Printf("debug: addOrUpdatePod() Pod %s in ns %s with the GPU ID %d should be added to device map",
		pod.Name,
		pod.Namespace,
		id)
	if id >= 0 {
		dev, found := n.Devs[id]
		log.Printf("dev idx is %+v and node is %+v", dev.Idx, n.Name)
		if !found {
			log.Printf("warn: Pod %s in ns %s failed to find the GPU ID %d in node %s", pod.Name, pod.Namespace, id, n.Name)
		} else {
			dev.addPod(pod)
			added = true
		}
	} else {
		log.Printf("warn: Pod %s in ns %s is not set the GPU ID %d in node %s", pod.Name, pod.Namespace, id, n.Name)
	}
	return added
}

// check if the pod can be allocated on the node
func (n *NodeInfo) Assume(pod *v1.Pod) (allocatable bool) {
	allocatable = false

	n.Rwmu.RLock()
	log.Printf("trace: NodeRWLocker: %v Assume() RLocked", n.Name)
	defer log.Printf("trace: NodeRWLocker: %v Assume() RUnlocked", n.Name)
	defer n.Rwmu.RUnlock()

	availableGPUs := n.getAvailableGPUs()
	reqGPU := uint(utils.GetGPUMemoryFromPodResource(pod))
	log.Printf("debug: n.devs are: %+v", n.Devs)
	log.Printf("debug: q: %v in node %s", availableGPUs, n.Name)
	log.Printf("debug: Length of AvailableGPUs is: %v in node %s", len(availableGPUs), n.Name)

	if len(availableGPUs) > 0 {
		for devID := 0; devID < len(n.Devs); devID++ {
			availableGPU, ok := availableGPUs[devID]
			log.Printf("AvailableGPU is %+v, ok is %+v", availableGPUs, ok)
			if ok {
				if availableGPU >= reqGPU {
					allocatable = true
					break
				}
			}
		}
	}
	log.Printf("debug: Pod %v AvailableGPUs is allocatable %v in node %v", pod.Name, allocatable, n.Name)
	return allocatable

}

// Judge if base job should tune by get remaining epoch from etcd
// remaing jct is 3, return false
func FilterBaseJobWillComplete(base_pods []*v1.Pod) []*v1.Pod {
	var new_base_pods []*v1.Pod
	epoch_to_complete := 7.0
	if len(base_pods) <= 0 {
		return base_pods
	}
	for _, base_pod := range base_pods {
		cur_epoch, err := methods.GetPodContentByEtcd(base_pod.Name, "cur_epoch")
		if err != nil {
			log.Printf("Get cur_epoch of pod %v is err due to %+v", base_pod.Name, err)
		}
		batchsize, err := methods.GetPodContentByEtcd(base_pod.Name, "batchsize")
		if err != nil {
			log.Printf("Get cur_epoch of pod %v is err due to %+v", base_pod.Name, err)
		}
		cur_epoch_, _ := strconv.Atoi(cur_epoch)
		batchsize_, _ := strconv.Atoi(batchsize)
		job_name := strings.Split(base_pod.Name, "-")[0]
		if strings.Contains(job_name, "alexnet") || strings.Contains(job_name, "neumf") || strings.Contains(job_name, "adgcl") {
			epoch_to_complete = 3
		}
		epochs := methods.FitBatchsizeEpoch3(job_name, batchsize_)
		if epochs-float64(cur_epoch_)-epoch_to_complete > 0 {
			new_base_pods = append(new_base_pods, base_pod)
		}
	}
	return new_base_pods
}

type OnlineInterference struct {
	Job_name           string
	Pod_name           string
	Co_batch_time      float64
	Batch_size         int
	Single_batch_time  float64
	Final_interference float64
}

func (n *NodeInfo) Allocate(clientset *kubernetes.Clientset, pod *v1.Pod) (err error) {
	schedule_start_time := time.Now().UnixMilli()
	log.Printf("Updated node labels Tuning: %v", n.Node.Labels)
	n.Rwmu.Lock()
	log.Printf("trace: NodeRWLocker: %v Allocate() RLocked", n.Name)
	defer log.Printf("trace: NodeRWLocker: %v Allocate() RUnlocked", n.Name)
	defer n.Rwmu.Unlock()
	log.Printf("info: Allocate() ----Begin to allocate GPU for gpu mem for pod %s in ns %s----", pod.Name, pod.Namespace)
	// 1. Update the pod spec
	devId, found := n.allocateGPUID(clientset, pod)
	log.Printf("debug: devid is %v, found is %v", devId, found)
	var newPod *v1.Pod
	var basePods []*v1.Pod

	if found {
		log.Printf("info: Allocate() 1. Allocate GPU ID %d to pod %s in ns %s.----", devId, pod.Name, pod.Namespace)

		basePods = n.Devs[devId].GetPods()
		newPod, err = n.updatePod(clientset, pod, devId)
		if err != nil {
			log.Printf("Error: Failed to update pod %v due to %v", pod.Name, err)
		}
	} else {
		err = fmt.Errorf("the node %s can't place the pod %s in ns %s,and the pod spec is %v", pod.Spec.NodeName, pod.Name, pod.Namespace, pod)
	}

	// 2. update the device info if the pod is update sucessfully
	if err == nil {
		ok := n.updatePodOnDev(newPod, devId)
		if !ok {
			log.Printf("update pod %v on dev err, found is %v", newPod.Name, ok)
		} else {
			log.Printf("debug: successfully allocate pod %v on devId: %v", newPod.Name, devId)
		}
	}
	// 3. bind the pod to the node
	if err == nil {
		err = n.bindingPod(clientset, newPod)
		if err != nil {
			log.Fatalf("can not bind pod %v on node %v due to err %+v", newPod.Name, n.Name, err)
			return err
		}
	}
	// record scheduling time
	schedule_end_time := time.Now().UnixMilli()
	schedule_time := strconv.Itoa(int(schedule_end_time - schedule_start_time))
	if err == nil {
		_, err = methods.PutPodContentByEtcd(newPod.Name, "schedule_time", schedule_time)
		if err != nil {
			log.Printf("Put pod %v schedule_time in etcd err due to %+v", newPod.Name, err)
		}
	}
	log.Printf("schedule_time is: %+v", schedule_time)
	// record end!
	// 4. send data to tuner
	if err == nil && os.Getenv("SCHEDULER") == "IADEEP" {
		log.Printf("Using scheduler IADeep")
		var job_names []string
		var pod_names []string
		var batchsizes []int
		var t_m0 []float64
		var base_jobs []methods.JobInfo
		var new_job methods.JobInfo
		var interference float64

		for _, base_pod := range basePods {
			job_names = append(job_names, strings.Split(base_pod.Name, "-")[0])
			pod_names = append(pod_names, base_pod.Name)
			ok, job_info := methods.GetJobInfo(strings.Split(base_pod.Name, "-")[0])
			if ok {
				batchsizes = append(batchsizes, job_info.Best_batchsize)
				t_m0 = append(t_m0, job_info.T_m0)
			}
			job_info.Pod_name = base_pod.Name
			log.Printf("job_info.base_Pod_name is : %+v", job_info.Pod_name)
			base_jobs = append(base_jobs, *job_info)
		}
		job_names = append(job_names, strings.Split(newPod.Name, "-")[0])
		pod_names = append(pod_names, newPod.Name)
		ok, job_info := methods.GetJobInfo(strings.Split(newPod.Name, "-")[0])
		if ok {
			batchsizes = append(batchsizes, job_info.Best_batchsize)
			t_m0 = append(t_m0, job_info.T_m0)
		}
		job_info.Pod_name = newPod.Name
		log.Printf("job_info.Pod_name is : %+v", job_info.Pod_name)
		new_job = *job_info
		log.Printf("job_names are: %+v", job_names)
		log.Printf("pod_names are: %+v", pod_names)
		log.Printf("batchsizes are: %+v", batchsizes)
		log.Printf("length of basePods is: %+v", len(basePods))
		if len(basePods) > 0 {
			log.Printf("edit etcd.")
			for i, pod_name := range pod_names {
				log.Printf("pod_name is: %+v", pod_name)
				log.Printf("i is: %+v", i)
				// Just make tuned_batch_size different from init batchsize
				// -1, +1 or whatever any other operations are both OK
				methods.PutPodContentByEtcd(pod_name, "tuned_batch_size", strconv.Itoa(batchsizes[i]-1))
				methods.PutPodContentByEtcd(pod_name, "mini_batch_time_m0", strconv.FormatFloat(t_m0[i], 'E', -1, 64))
				methods.PutPodContentByEtcd(pod_name, "tuning", "1")

			}
			log.Printf("base_jobs is: %+v", base_jobs)
			log.Printf("new_jobs is: %+v", new_job)
			both_batch_time := methods.GetBothMinibatchTimeFromEtcd(base_jobs, new_job)
			log.Printf("both_batch_time:%+v", both_batch_time)

			log.Printf("len(both_batch_time):%+v", len(both_batch_time))

			var online_interferences []OnlineInterference
			for i := 0; i < len(both_batch_time); i++ {
				test := OnlineInterference{
					Co_batch_time:     both_batch_time[i].Time,
					Batch_size:        both_batch_time[i].Batchsize,
					Pod_name:          pod_names[i],
					Job_name:          job_names[i],
					Single_batch_time: t_m0[i],
				}
				online_interferences = append(online_interferences, test)

				log.Printf("online_interference.Co_batch_time is:%+v", online_interferences[i].Co_batch_time)
				log.Printf("online_interferences.Batch_size is:%+v", online_interferences[i].Batch_size)
				log.Printf("online_interferences.Single_batch_time is:%+v", online_interferences[i].Single_batch_time)
			}

			// 	addtional_interference, _ := methods.GetAdditionalInterferenceScoreFromCsv(job_names[0:len(job_names)-1], job_names[len(job_names)-1])

			csvFile, err := os.Open(os.Getenv("CSV_FOLDER") + "/online_interference.csv")
			if err != nil {
				log.Fatal(err)
			}
			defer csvFile.Close()

			csvDf := dataframe.ReadCSV(csvFile)

			jobs := methods.CreateJobKV(job_names[0:len(job_names)-1], job_names[len(job_names)-1])
			log.Printf("jobs are: %+v", jobs)
			for k, v := range jobs {
				num := strconv.FormatInt(int64(v), 2)
				fil := csvDf.Filter(
					dataframe.F{Colname: k, Comparator: series.Eq, Comparando: v},
					dataframe.F{Colname: k + "_num", Comparator: series.Eq, Comparando: num},
				)
				csvDf = fil
				log.Printf("k is %v and v is %v", k, v)
			}
			if csvDf.Nrow() <= 0 {

				var1 := methods.OnlineCreatePredRecord(jobs)
				for m := 0; m < len(both_batch_time); m++ {
					interference += (online_interferences[m].Co_batch_time - online_interferences[m].Single_batch_time) / online_interferences[m].Single_batch_time
					log.Printf("interference is : %+v", interference)
				}
				log.Printf("interference is : %+v", interference)
				methods.WriteRecord(var1, interference)
				log.Printf("interference is : %+v", interference)
			}

		}

	}
	log.Printf("info: Allocate() ----End to allocate GPU for gpu mem for pod %s in ns %s----", pod.Name, pod.Namespace)
	return err
}

// update pod
func (n *NodeInfo) updatePod(clientset *kubernetes.Clientset, pod *v1.Pod, devId int) (*v1.Pod, error) {

	log.Printf("debug: update pod %v", pod.Name)
	// patchedAnnotationBytes, err := utils.PatchPodAnnotationSpec(pod, devId, n.GetTotalGPUMemory()/n.GetGPUCount())
	patchedAnnotationBytes, err := utils.PatchPodAnnotationSpec(pod, devId, int(utils.GetGPUStatusInNode(n.Node)[devId].MemCapacity))
	log.Printf("debug: totalGPUmemoryByDev: %v", int(utils.GetGPUStatusInNode(n.Node)[devId].MemCapacity))
	if err != nil {
		return pod, fmt.Errorf("failed to generate patched annotations, reason: %v", err)
	}
	update_Pod, err := clientset.CoreV1().Pods(pod.Namespace).Patch(pod.Name, types.StrategicMergePatchType, patchedAnnotationBytes)
	log.Printf("debug: update pod %v to bind n %v", update_Pod.Name, n.Name)
	if err != nil {
		log.Printf("updatePod is err due to %v", err)
		if err.Error() == OptimisticLockErrorMsg {
			// retry
			pod, err = clientset.CoreV1().Pods(pod.Namespace).Get(pod.Name, metav1.GetOptions{})
			if err != nil {
				return pod, err
			}
			update_Pod, err = clientset.CoreV1().Pods(pod.Namespace).Patch(pod.Name, types.StrategicMergePatchType, patchedAnnotationBytes)
			if err != nil {
				return update_Pod, err
			}
		} else {
			log.Printf("Error: Failed to patch pod %v", pod)
			return pod, err
		}
	}
	return update_Pod, err
}

// binding pod to node
func (n *NodeInfo) bindingPod(clientset *kubernetes.Clientset, pod *v1.Pod) error {
	log.Printf("debug: bind pod %v to node %v", pod.Name, n.Name)
	binding := &v1.Binding{
		ObjectMeta: metav1.ObjectMeta{Name: pod.Name, UID: pod.UID},
		Target:     v1.ObjectReference{Kind: "Node", Name: n.Name},
	}
	log.Printf("info: Allocate() 2. Try to bind pod %s in %s namespace to node %s with %v",
		pod.Name,
		pod.Namespace,
		pod.Spec.NodeName,
		binding)
	err := clientset.CoreV1().Pods(pod.Namespace).Bind(binding)
	if err != nil {
		log.Printf("warn: Failed to bind the pod %s in ns %s due to %v", pod.Name, pod.Namespace, err)
		return err
	}
	return err
}

// update pod on dev
func (n *NodeInfo) updatePodOnDev(pod *v1.Pod, devId int) bool {
	log.Printf("info: Allocate() 3. Try to add pod %s in ns %s to dev %d",
		pod.Name,
		pod.Namespace,
		devId)
	dev, found := n.Devs[devId]
	if !found {
		log.Printf("warn: Pod %s in ns %s failed to find the GPU ID %d in node %s", pod.Name, pod.Namespace, devId, n.Name)
	} else {
		dev.addPod(pod)
		pods := dev.GetPods()
		log.Printf("length of pods is %+v", len(pods))
		if len(pods) > 1 {
			// add pod to device of node
			// pod_names, err := methods.GetPodContentByEtcd(n.Name, strconv.Itoa(devId))
			var pod_names []string
			for _, obj := range pods {
				pod_names = append(pod_names, obj.Name)
			}
			log.Printf("pod_names are: %+v", pod_names)

			pod_names_str, err := json.Marshal(pod_names)
			if err != nil {
				panic(err)
			}

			log.Printf("nodeName is: %+v", n.Name)
			log.Printf("json.Marshal(pod_names) are: %+v", string(pod_names_str))
			if os.Getenv("SCHEDULER") == "IADEEP" {
				methods.PutPodContentByEtcd(n.Name, strconv.Itoa(devId), string(pod_names_str))
				log.Printf("put end.")
			}
		}
	}
	return found
}

type Interference struct {
	DevId        int
	interference float64
}

// allocate the GPU ID to the pod
func (n *NodeInfo) allocateGPUID(clientset *kubernetes.Clientset, pod *v1.Pod) (candidateDevID int, found bool) {

	reqGPU := uint(0)
	found = false
	candidateDevID = -1
	candidateGPUMemory := uint(0)
	availableGPUs := n.getAvailableGPUs()
	log.Printf("ndevs length is: %v", len(n.Devs))

	new_job := pod.Name

	var interferences []Interference

	reqGPU = uint(utils.GetGPUMemoryFromPodResource(pod))

	if reqGPU > uint(0) {
		log.Printf("info: reqGPU for pod %s in ns %s: %d", pod.Name, pod.Namespace, reqGPU)
		log.Printf("info: AvailableGPUs: %v in node %s", availableGPUs, n.Name)
		nodeInfo, err := clientset.CoreV1().Nodes().Get(n.Name, metav1.GetOptions{})
		if err != nil {
			panic(err)
		}
		gpustatus := utils.GetGPUStatusInNode(nodeInfo)
		log.Printf("debug: gpuStatus: %+v", gpustatus)

		log.Printf("checkDevID of %+v", n.Devs)

		if len(availableGPUs) > 0 {
			for devID := 0; devID < len(n.Devs); devID++ {
				log.Printf("check devID of %+v of Tuning %+v", devID, n.Devs[devID].Tuning)
				var jobs []string
				availableGPU, ok := availableGPUs[devID]
				pods := n.Devs[devID].PodMap
				if len(pods) == 0 {
					methods.PutPodContentByEtcd(n.Name, strconv.Itoa(devID), "")
				}
				gpuUtil := gpustatus[devID].GpuUtil
				memUtil := gpustatus[devID].MemUtil
				// process := gpustatus[devID].Process
				memUsed := gpustatus[devID].MemUsed
				log.Printf("memUsed is %+v of devId %+v", memUsed, devID)
				memCapacity := gpustatus[devID].MemCapacity
				devIsTuning := methods.GetTuningStatusFromEtcd(n.Name, strconv.Itoa(devID))
				log.Printf("Device %+v on node %+v has tuning status %+v", devID, n.Name, devIsTuning)
				// if memCapacity > 12 && !devIsTuning && ok { //for 3090 GPUs and not Tuning GPUs
				// if memCapacity > 12 && ok { //for 3090 GPUs and not Tuning GPUs
				if !devIsTuning && ok { //for not Tuning GPUs
					if availableGPU >= reqGPU {
						if candidateDevID == -1 || candidateGPUMemory > availableGPU {
							if os.Getenv("SCHEDULER") == "ANTMAN" {
								log.Printf("return devID %d on node %s", devID, n.Name)
								candidateGPUMemory = availableGPU
								return devID, true
							} else {
								if memCapacity-memUsed == memCapacity { //return idle GPU
									return devID, true
								}
								if gpuUtil < utils.GPUUtilThreshold && memUtil < utils.MemUtiliThreshold {
									// some processes on GPU devID
									interference_value := 0.0
									if len(pods) > 0 {
										for _, item := range pods {
											job_name := strings.Split(item.Name, "-")[0]
											jobs = append(jobs, job_name)
										}
										// base_job := strings.Join(jobs, ",")
										new_job_name := strings.Split(new_job, "-")[0]
										log.Printf("debug base_job: %v", jobs)
										log.Printf("debug new_job: %v", new_job_name)
										total_records := methods.ReadRecord()
										if os.Getenv("SCHEDULER") == "IADEEP" {
											interference_value, _ = methods.GetInterferenceFromFile(jobs, new_job_name, total_records)
										} else if os.Getenv("SCHEDULER") == "KERNELEST" {
											interference_value, _ = methods.GetInterferenceScoreFromUtilCsv(jobs, new_job_name)
										}
										log.Printf("interference on devID %v is: %v", devID, interference_value)
									}
									interferences = append(interferences, Interference{
										DevId:        devID,
										interference: interference_value,
									})
									found = true
								}
							}
						}
					}

				}
			}
			if found && os.Getenv("ORG_CODE") != "true" {
				sort.Slice(interferences, func(i, j int) bool {
					return interferences[i].interference < interferences[j].interference
				})
				candidateDevID = interferences[0].DevId
			}
			log.Printf("debug: interferences %v on devID %v found %v", interferences, candidateDevID, found)
		}

		if found {
			log.Printf("info: Find candidate dev id %d for pod %s in ns %s successfully.",
				candidateDevID,
				pod.Name,
				pod.Namespace)
		} else {
			log.Printf("warn: Failed to find available GPUs %d for the pod %s in the namespace %s",
				reqGPU,
				pod.Name,
				pod.Namespace)
		}
	}
	log.Printf("allocateGPUID is %v, and found is %v", candidateDevID, found)
	return candidateDevID, found
}

func (n *NodeInfo) getAvailableGPUs() (availableGPUs map[int]uint) {
	allGPUs := n.getAllGPUs()
	usedGPUs := n.getUsedGPUs()
	unhealthyGPUs := n.getUnhealthyGPUs()
	availableGPUs = map[int]uint{}
	for id, totalGPUMem := range allGPUs {
		if usedGPUMem, found := usedGPUs[id]; found {
			availableGPUs[id] = totalGPUMem - usedGPUMem
		}
	}
	log.Printf("info: available GPU list %v before removing unhealty GPUs", availableGPUs)
	for id, _ := range unhealthyGPUs {
		log.Printf("info: delete dev %d from availble GPU list", id)
		delete(availableGPUs, id)
	}
	log.Printf("info: available GPU list %v after removing unhealty GPUs", availableGPUs)

	return availableGPUs
}

// device index: gpu memory
func (n *NodeInfo) getUsedGPUs() (usedGPUs map[int]uint) {
	usedGPUs = map[int]uint{}
	for _, dev := range n.Devs {
		usedGPUs[dev.Idx] = dev.GetUsedGPUMemory()
	}
	log.Printf("info: getUsedGPUs: %v in node %s, and devs %v", usedGPUs, n.Name, n.Devs)
	return usedGPUs
}

// device index: gpu memory
func (n *NodeInfo) getAllGPUs() (allGPUs map[int]uint) {
	allGPUs = map[int]uint{}
	for _, dev := range n.Devs {
		allGPUs[dev.Idx] = dev.TotalGPUMem
	}
	log.Printf("info: getAllGPUs: %v in node %s, and dev %v", allGPUs, n.Name, n.Devs)
	return allGPUs
}

// getUnhealthyGPUs get the unhealthy GPUs from configmap
func (n *NodeInfo) getUnhealthyGPUs() (unhealthyGPUs map[int]bool) {
	unhealthyGPUs = map[int]bool{}
	name := fmt.Sprintf("unhealthy-gpu-%s", n.GetName())
	log.Printf("info: try to find unhealthy node %s", name)
	cm := getConfigMap(name)
	if cm == nil {
		return
	}

	if devicesStr, found := cm.Data["gpus"]; found {
		log.Printf("warn: the unhelathy gpus %s", devicesStr)
		idsStr := strings.Split(devicesStr, ",")
		for _, sid := range idsStr {
			id, err := strconv.Atoi(sid)
			if err != nil {
				log.Printf("warn: failed to parse id %s due to %v", sid, err)
			}
			unhealthyGPUs[id] = true
		}
	} else {
		log.Println("info: skip, because there are no unhealthy gpus")
	}

	return

}
