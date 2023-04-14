package tuning

import (
	"bufio"
	"bytes"
	"context"
	"encoding/csv"
	"encoding/json"
	"io/ioutil"
	"net/http"
	"os"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/util/workqueue"

	"iadeep-local-coordinator/utils"

	"log"
)

type TuningController struct {
	clientset *kubernetes.Clientset
	// tuningQueue is a rate limited work queue. This is used to queue work to be
	// processed instead of performing it as soon as a change happens. This
	// means we can ensure we only process a fixed amount of resources at a
	// time, and makes it easy to ensure we are never processing the same item
	// simultaneously in two different workers.
	node        *v1.Node
	tuningQueue workqueue.RateLimitingInterface
}

type Batch_PD struct {
	Batchsize int
	PD        float64
}

type JobInfo struct {
	Name              string
	Pod_name          string
	Best_batchsize    int
	Epoch_time        float64
	JCT               float64
	Samples           int
	Batchsizes_PD     []Batch_PD
	Minibatch_time_m0 float64
}

type ReqData struct {
	DevId           int
	Job_names       []string
	Pod_names       []string
	Init_batchsizes []int
	Batchsize_info  []int
	PD_info         []float64
}

type RespResult struct {
	Err    string `json:"err"`
	Result []int  `json:"result"`
}

func NewTuningController(clientset *kubernetes.Clientset, tuningQueue workqueue.RateLimitingInterface, stopCh <-chan struct{}) (*TuningController, error) {
	log.Printf("info: Creating event broadcaster")

	nodeName := os.Getenv("NODE_NAME")
	log.Printf("nodeName is: %+v", nodeName)
	ctx := context.TODO()

	node, err := clientset.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
	if err != nil {
		log.Printf("Info: get nodes from client err due to %+v", err)
	}

	c := &TuningController{
		clientset:   clientset,
		node:        node,
		tuningQueue: tuningQueue,
	}

	return c, nil
}

// Run will set up the event handlers
func (c *TuningController) Run(threadiness int, stopCh <-chan struct{}) error {
	defer runtime.HandleCrash()
	defer c.tuningQueue.ShutDown()

	log.Println("info: Starting Tuning Controller.")
	log.Println("info: Waiting for informer caches to sync")
	log.Printf("info: Starting %v workers.", threadiness)

	for i := 0; i < threadiness; i++ {
		go wait.Until(c.runWorker, time.Second, stopCh)
	}

	log.Println("info: Started workers")
	<-stopCh
	log.Println("info: Shutting down workers")

	return nil
}

// runWorker is a long-running function that will continually call the
// processNextTuningQueueItem function in order to read and process a message on the
// workqueue.
func (c *TuningController) runWorker() {
	for c.processNextTuningQueueItem() {
	}
}

// processNextTuningQueueItem will read a single work item off the podQueue and
// attempt to process it.
func (c *TuningController) processNextTuningQueueItem() bool {
	log.Println("trace: beginProcessNextTuningQueueItem()")
	key, quit := c.tuningQueue.Get()
	if quit {
		return false
	}
	defer c.tuningQueue.Done(key)
	defer log.Println("trace: endProcessNextTuningQueueItem()")
	tuningPods := key.(*utils.TuningPods)
	nodeName := c.node.Name
	forget, err := TuningBatchsizes(tuningPods, nodeName, c.tuningQueue)

	if err == nil {
		if forget {
			c.tuningQueue.Forget(key)
		}

		return false
	} else {
		log.Printf("error: Tuning batchsizes is err due to %v", err)
	}

	c.tuningQueue.AddRateLimited(key)
	return true
}

func CheckPodsCompletion(podNames []string) bool {
	for _, podName := range podNames {
		complete, err := utils.GetContentFromEtcd(podName, "complete")
		if err != nil {
			log.Printf("warn: Read %v/complete from etcd failed due to %v", podName, err)
		}
		if complete == "1" {
			return true
		}
	}
	return false
}

// input: node name, dev id
func TuningBatchsizes(tuningPods *utils.TuningPods, nodeName string, tuningQueue workqueue.RateLimitingInterface) (bool, error) {
	log.Print("Tuning batch sizes function start.")
	devId, pod_names := tuningPods.DevId, tuningPods.PodNames
	log.Printf("nodeName is %s, and devId is %v", nodeName, devId)

	var job_names []string
	var init_batchsizes []int

	var Batchsize_info []int
	var PD_info []float64

	hasCompletedPods := CheckPodsCompletion(pod_names)
	if hasCompletedPods {
		log.Println("info: Some pods have 'Completed'/'Failed'/'Terminated' during tuning process. Current tuning process was gived up.")
		return true, nil
	}
	log.Println("info: All pods are running. Keep tuning.")

	// 2022_Oct_31 update:
	// If GPU already has tuned pods, only tuning new coming
	// pods instead of tuning all pods on that GPU.
	// Here I use the length of pod_names to detect if there are any tuned pods
	has_tuned_pods := len(pod_names) > 2
	if has_tuned_pods {
		pod_name := pod_names[len(pod_names)-1]
		log.Printf("trace: Device %v/%v alread has tuned pods, tuning process will only perform on pod %v", nodeName, devId, pod_name)
		job_name := strings.Split(pod_name, "-")[0]
		ok, job_info := GetJobInfo(job_name)
		if !ok {
			log.Printf("error: Failed to get job info of %v", job_name)
			return false, nil
		}
		job_info.Pod_name = pod_name
		batch_pd := GetBothPDFromEtcd([]JobInfo{}, *job_info)

		Batchsize_info = append(Batchsize_info, batch_pd[0].Batchsize)
		PD_info = append(PD_info, batch_pd[0].PD)
		job_names = append(job_names, job_name)
		pod_names = append(pod_names, pod_name)
		init_batchsizes = append(init_batchsizes, job_info.Best_batchsize)
	} else {
		log.Printf("trace: Device %v/%v doesn't have tuned pods, tuning process will perform on pods %v", nodeName, devId, pod_names)
		var base_job_batchs []JobInfo
		var new_job_batch JobInfo
		for idx, pod_name := range pod_names {
			log.Printf("trace: start get info of pod %s", pod_name)
			base_job_name := strings.Split(pod_name, "-")[0]
			ok, job_info := GetJobInfo(base_job_name)
			if !ok {
				log.Printf("error: Failed to get base job %v", base_job_name)
				return false, nil
			}
			job_info.Pod_name = pod_name
			init_batchsizes = append(init_batchsizes, job_info.Best_batchsize)

			if idx < len(pod_names)-1 {
				base_job_batchs = append(base_job_batchs, *job_info)
			} else {
				new_job_batch = *job_info
			}
			job_names = append(job_names, base_job_name)
		}
		log.Printf("debug: base_job_batchs are: %+v", base_job_batchs)
		minibatch_times := GetBothPDFromEtcd(base_job_batchs, new_job_batch)
		log.Printf("debug: minibatch_times are %+v", minibatch_times)

		pds := 0.0
		for _, batch := range minibatch_times {
			pds += batch.PD
		}
		for _, pod_name := range pod_names {
			tuned_batch_size_str, _ := utils.GetContentFromEtcd(pod_name, "tuned_batch_size")
			tuned_batch_size, _ := strconv.Atoi(tuned_batch_size_str)
			log.Printf("debug: Tunned batchsizes: %v", tuned_batch_size_str)
			Batchsize_info = append(Batchsize_info, tuned_batch_size)
		}

		PD_info = append(PD_info, pds)
	}

	resdata := ReqData{
		DevId:           devId,
		Job_names:       job_names,
		Pod_names:       pod_names,
		Init_batchsizes: init_batchsizes,
		Batchsize_info:  Batchsize_info,
		PD_info:         PD_info,
	}
	log.Printf("debug: send to ucb-server reqdata is %+v", resdata)
	tuned_batchsizes := SendTuneRequest(resdata) //数组
	log.Printf("debug: tuned_batchsizes are: %+v", tuned_batchsizes)
	if reflect.DeepEqual(tuningPods.Batchsizes, tuned_batchsizes) {
		tuningPods.Count += 1
		log.Printf("debug: tuningPods.Count is: %+v", tuningPods.Count)
	} else {
		tuningPods.Batchsizes = tuned_batchsizes
		tuningPods.Count = 1
		log.Printf("debug: tuningPods.Count is: %+v", tuningPods.Count)
		log.Printf("debug: tuningPods.Batchsizes is: %+v", tuningPods.Batchsizes)
	}

	if len(pod_names) > 0 && len(tuned_batchsizes) > 0 && tuned_batchsizes[0] > 0 && tuningPods.Count < 5 {
		log.Printf("info: edit etcd.")
		if has_tuned_pods {
			pod_names = pod_names[len(pod_names)-1:]
		}
		for i, pod_name := range pod_names {
			utils.PutContentToEtcd(pod_name, "tuning", "1")
			batch_sizes_str := strconv.Itoa(tuned_batchsizes[i])
			utils.PutContentToEtcd(pod_name, "tuned_batch_size", batch_sizes_str)
		}
		log.Printf("info: tuned batch size")
		tuningQueue.Add(tuningPods)
	} else {
		log.Printf("info: Tuning finished! Current pods on %v/%v are %v", nodeName, devId, pod_names)
		// modify tuning tag to false
		utils.DeleteContentFromEtcd(nodeName, strconv.Itoa(tuningPods.DevId))
	}
	return true, nil
}

// get PD of base jobs and new job from etcd
func GetBothPDFromEtcd(base_jobs []JobInfo, new_job JobInfo) []Batch_PD {
	var job_batch_pd []Batch_PD
	var waitGroup sync.WaitGroup
	var chan_arr = make(chan Batch_PD, len(base_jobs)+1)
	complete_jobs := map[string]JobInfo{}
	defer close(chan_arr)

	waitGroup.Add(1)
	log.Printf("new_job of %v start time is %v", new_job.Pod_name, time.Now().Unix())
	go GetJobPDFromEtcd(new_job.Pod_name, &waitGroup, chan_arr)
	log.Printf("new_job of %v end time is %v", new_job.Pod_name, time.Now().Unix())

	for i := 0; i < len(base_jobs); i++ {
		log.Printf("base_jobs of pod %v start time is %v", base_jobs[i].Pod_name, time.Now().Unix())
		complete, _ := utils.GetContentFromEtcd(base_jobs[i].Pod_name, "complete")
		if complete == "1" {
			complete_jobs[base_jobs[i].Pod_name] = base_jobs[i]
		} else {
			waitGroup.Add(1)
			go GetJobPDFromEtcd(base_jobs[i].Pod_name, &waitGroup, chan_arr)
		}
		log.Printf("base_jobs of pod %v end time is %v", base_jobs[i].Pod_name, time.Now().Unix())

	}

	for i := 0; i < len(base_jobs)+1; i++ {
		var item Batch_PD
		if i < len(base_jobs) {
			if complete_job, ok := complete_jobs[base_jobs[i].Pod_name]; ok {
				batchsize_str, err := utils.GetContentFromEtcd(complete_job.Pod_name, "tuned_batch_size")
				if err != nil {
					log.Printf("get tuned_batch_size from etcd err %+v", err)
				}
				batchsize, _ := strconv.Atoi(batchsize_str)
				item = Batch_PD{
					Batchsize: batchsize,
					PD:        0.0,
				}
			} else {
				item = <-chan_arr
			}
		} else {
			item = <-chan_arr
		}

		job_batch_pd = append(job_batch_pd, item)
	}
	log.Printf("get batch pd from etcd of pods %+v", job_batch_pd)

	waitGroup.Wait()
	log.Printf("waitGroup.Wait() done of pod %+v!", new_job.Pod_name)
	return job_batch_pd
}

// get PD (performance degradation) from etcd
func GetJobPDFromEtcd(podName string, waitGroup *sync.WaitGroup, c chan Batch_PD) {
	start_time := time.Now().Unix()
	log.Printf("GetJobPDFromEtcd: pod name is %v", podName)
	log.Printf("GetJobPDFromEtcd: start time is %v", start_time)
	job_batch_time := Batch_PD{}
	var (
		val string
		err error
	)
	// Try to get content from etcd first, if returns empty,
	// then watch that key until the value is put into that key
	if val, _ = utils.GetContentFromEtcd(podName, "pd"); val == "" {
		log.Printf("trace: %v get pd from etcd failed, watch etcd now", podName)
		val, err = utils.WatchKeyFromEtcd(podName, "pd")
	}
	// After get the value, delete it from etcd to avoid get the
	// same value next time
	utils.DeleteContentFromEtcd(podName, "pd")
	if err != nil {
		log.Printf("get pd from etcd err due to %+v", err)
	}
	batchsize_str, err := utils.GetContentFromEtcd(podName, "tuned_batch_size")
	if err != nil {
		log.Printf("get tuned_batch_size from etcd err %+v", err)
	}
	batchsize, _ := strconv.Atoi(batchsize_str)
	if pd, err := strconv.ParseFloat(val, 64); err == nil {
		job_batch_time.PD = pd
	}
	job_batch_time.Batchsize = batchsize

	end_time := time.Now().Unix()
	log.Printf("GetJobPDFromEtcd time is %v", end_time-start_time)
	c <- job_batch_time
	log.Printf("wait group done of pod %v!", podName)
	end_time = time.Now().Unix()
	log.Printf("GetJobPDFromEtcd: end time is %v", end_time)
	log.Printf("FinishGetJobPDFromEtcd time is %v", end_time-start_time)
	defer waitGroup.Done()
}

// calculate the averget minibatch time
func GetAvg(durations []float64) float64 {
	var sum, avg float64
	sort.Float64sAreSorted(durations)
	for k := 1; k < len(durations); k++ {
		sum += durations[k]
	}
	avg = sum / float64(len(durations))
	log.Printf("Info: The average of minibatch durations is %v", avg)
	return avg
}

func GetJobInfo(name string) (bool, *JobInfo) {
	jobs, err := GetOfflineInfo()
	if err != nil {
		log.Println("info: GetOfflineInfo failed!")
		return false, nil
	}
	for _, item := range jobs {
		if item.Name == name {
			log.Printf("info: Find job %v success", name)
			return true, &item
		}
	}
	log.Printf("info: Find job %v failed", name)
	return false, nil
}

func GetOfflineInfo() ([]JobInfo, error) {
	csvFolder := os.Getenv("CSV_FOLDER")
	csvFile, err := os.Open(csvFolder + "/offline_info.csv")
	if err != nil {
		log.Fatal(err)
	}
	reader := csv.NewReader(bufio.NewReader(csvFile))
	defer csvFile.Close()
	var jobs []JobInfo
	content, err := reader.ReadAll()
	if err != nil {
		log.Fatalf("can not readall of offline_info.csv, err is %+v", err)
	}
	for i, line := range content {
		if i == 0 {
			continue
		} else {
			best_batchsize, err := strconv.ParseInt(line[1], 10, 64)
			if err != nil {
				log.Fatalf("can not read best_batchsize due to err %+v", err)
				return jobs, err
			}
			epoch_time, err := strconv.ParseFloat(line[2], 64)
			if err != nil {
				log.Fatalf("can not read epoch_time due to err %+v", err)
				return jobs, err
			}
			jct, err := strconv.ParseFloat(line[3], 64)
			if err != nil {
				log.Fatalf("can not read jct due to err %+v", err)
				return jobs, err
			}
			samples, err := strconv.ParseInt(line[4], 10, 64)
			if err != nil {
				log.Fatalf("can not read samples due to err %+v", err)
				return jobs, err
			}
			minibatch_time_m0, err := strconv.ParseFloat(line[5], 64)
			if err != nil {
				log.Fatalf("can not read minibatch_time_m0 due to err %+v", err)
				return jobs, err
			}
			job := JobInfo{
				Name:              line[0],
				Best_batchsize:    int(best_batchsize),
				Epoch_time:        epoch_time,
				JCT:               jct,
				Samples:           int(samples),
				Minibatch_time_m0: minibatch_time_m0,
			}
			jobs = append(jobs, job)
		}
	}
	// log.Printf("offline jobs are %+v", jobs)
	return jobs, err
}

// Get the GPU count of the node
func GetGPUCountInNode(node *v1.Node) int {
	val, ok := node.Status.Capacity["gpushare/gpu-count"]

	if !ok {
		return int(0)
	}

	return int(val.Value())
}

func SendTuneRequest(reqdata ReqData) []int {
	var res []int = []int{0, 0}

	bytesData, err := json.Marshal(reqdata)
	// reader := bytes.NewReader(bytesData)

	if err != nil {
		log.Fatal(err)
	}
	start_req := time.Now().Unix()
	request, err := http.NewRequest("POST", "http://127.0.0.1:8888", bytes.NewBuffer(bytesData))
	log.Printf("SendTuneRequest request is %+v", request)
	if err != nil {
		log.Fatal(err)
	}
	request.Header.Set("Content-Type", "application/json; charset=UTF-8")

	client := &http.Client{
		Timeout: 200 * time.Second,
	}
	response, err := client.Do(request) //Do 方法发送请求，返回 HTTP 回复
	log.Printf("SendTuneRequest response is %+v", response)
	if err != nil {
		log.Printf("Failed to get response from client due to %v", err.Error())
		return res
	}
	defer request.Body.Close()
	log.Println("response Status:", response.Status)
	log.Println("response Headers:", response.Header)
	respBytes, err := ioutil.ReadAll(response.Body)
	log.Printf("SendTuneRequest respBytes is %+v", string(respBytes))

	if err != nil {
		log.Printf("Failed to convert response to respBytes from client due to %v", err.Error())
		return res
	}

	respResult := RespResult{}
	jsonErr := json.Unmarshal(respBytes, &respResult)
	if jsonErr != nil {
		log.Fatal(jsonErr)
	}
	end_req := time.Now().Unix()
	log.Printf("SendTuneRequest request time is %+v", end_req-start_req)
	return respResult.Result
}
