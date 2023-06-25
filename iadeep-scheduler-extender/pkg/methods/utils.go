package methods

import (
	"bufio"
	"bytes"
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"go.etcd.io/etcd/clientv3"
	v1 "k8s.io/api/core/v1"
)

// "github.com/cryptowilliam/goutil/container/gpoly/polyfit"
// )

type JobInfo struct {
	Name             string
	Pod_name         string
	Best_batchsize   int
	Epoch_time       float64
	JCT              float64
	Samples          int
	T_m0             float64
	Batchsizes_times []Batch_time
}

type Batch_time struct {
	Pod_name  string
	Batchsize int
	Time      float64
}

type ReqData struct {
	Job_names         []string
	Best_batchsizes   []int
	Batchsize_info    [][]int
	Interference_info []float64
	Threshold         int
}

type RequestData struct {
	Pod_names  []string
	Job_names  []string
	Batchsizes []int
}

type RespResult struct {
	Err    string `json:"err"`
	Result []int  `json:"result"`
}

func GetOfflineInfo() ([]JobInfo, error) {
	csvFile, err := os.Open(os.Getenv("CSV_FOLDER") + "/offline_info.csv")
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
			t_m0, err := strconv.ParseFloat(line[5], 64)
			if err != nil {
				log.Fatalf("can not read t_m0 due to err %+v", err)
				return jobs, err
			}
			job := JobInfo{
				Name:           line[0],
				Best_batchsize: int(best_batchsize),
				Epoch_time:     epoch_time,
				JCT:            jct,
				Samples:        int(samples),
				T_m0:           t_m0,
			}
			jobs = append(jobs, job)
		}
	}
	// log.Printf("offline jobs are %+v", jobs)
	return jobs, err
}

func GetJobInfo(name string) (bool, *JobInfo) {
	log.Printf("debug: job name is %v", name)
	jobs, err := GetOfflineInfo()
	if err != nil {
		log.Printf("debug: didn't find the job!")
	}
	for _, item := range jobs {
		if item.Name == name {
			log.Printf("find job is true")
			return true, &item
		}
	}
	log.Printf("debug: find job %+v, err is %+v", name, err)
	return false, nil
}

func NewSlice(start int, count int, step int) []Batch_time {
	s := make([]int, count)
	var batch_times []Batch_time
	for i := range s {
		s[i] = start
		batch_times = append(batch_times, Batch_time{
			Batchsize: start,
			Time:      0.0,
		})
		start += step
	}
	return batch_times
}

func SetBatchsizes(best_batchsize int, pod_name string) []Batch_time {
	var batch_times []Batch_time
	interval := 128
	if strings.Contains(pod_name, "adgcl") {
		interval = 8
	}
	if strings.Contains(pod_name, "vgg19") {
		interval = 64
	}
	step := [5]int{-2, -1, 0, 1, 2}
	for i := 0; i < len(step); i++ {
		batchsize := best_batchsize + interval*step[i]
		if batchsize <= 0 {
			batchsize = best_batchsize
		}
		batch_times = append(batch_times, Batch_time{
			Pod_name:  pod_name,
			Batchsize: batchsize,
			Time:      0.0,
		})
	}
	return batch_times
}

// get minibatch time from etcd
func GetJobMinibatchTimeFromEtcd(podName string, waitGroup *sync.WaitGroup, c chan Batch_time) {
	start_time := time.Now().Unix()
	log.Printf("GetJobMinibatchTimeFromEtcd: pod name is %v", podName)
	log.Printf("GetJobMinibatchTimeFromEtcd: start time is %v", start_time)
	job_batch_time := Batch_time{}
	job_batch_time.Pod_name = podName
	defer waitGroup.Done()
	// Try to get content from etcd first, if returns empty,
	// then watch that key until the value is put into that key
	var err error
	var val string
	if val, err = GetPodContentByEtcd(podName, "minibatch_time"); val == "" || err != nil {
		val, err = WatchKeyByEtcd(podName, "minibatch_time")
	}
	if err != nil {
		log.Printf("get minibatch_time from etcd err %+v", err)
	}
	minibatch_step, err := GetPodContentByEtcd(podName, "minibatch_step")
	if err != nil {
		log.Printf("Failed get minibatch_step %+v of pod %+v from etcd due to err %+v", minibatch_step, podName, err)
	} else {
		log.Printf("get minibatch_step %+v of pod %+v from etcd", minibatch_step, podName)
	}
	batchsize_str, err := GetPodContentByEtcd(podName, "batchsize")
	if err != nil {
		log.Printf("get minibatch from etcd err %+v", err)
	}
	batchsize, _ := strconv.Atoi(batchsize_str)
	var batch_time float64
	// batch_times := []float64{}
	res := strings.Split(val, ",")
	log.Printf("get minibatchtimes %+v from etcd", val)
	log.Printf("get minibatchtimes array %+v from etcd", res)

	for _, item := range res {
		log.Printf("res is %v", item)
		item = strings.ReplaceAll(item, " ", "")
		item = strings.ReplaceAll(item, "[", "")
		item = strings.ReplaceAll(item, "]", "")
		batch_time, _ = strconv.ParseFloat(item, 64)
		// batch_times = append(batch_times, batch_time)
	}
	log.Printf("val is : %+v", val)
	log.Printf("res is : %+v ", res)
	job_batch_time.Batchsize = batchsize
	log.Printf("job_batch_time.Batchsize is: %+v ", job_batch_time.Batchsize)
	log.Printf("batch_time is: %+v ", batch_time)

	// job_batch_time.Time = GetAvg(batch_times)
	job_batch_time.Time = batch_time
	log.Printf("job_batch_time.Time  is: %+v ", job_batch_time.Time)
	// _, err = DeletePodContentByEtcd(podName, "minibatch_time")
	// if err != nil {
	// 	log.Printf("delete minibatch_time from etcd err %+v", err)
	// }
	end_time := time.Now().Unix()
	log.Printf("GetJobMinibatchTimeFromEtcd time is %v", end_time-start_time)
	c <- job_batch_time
	log.Printf("wait group done of pod %v!", podName)
	end_time = time.Now().Unix()
	log.Printf("GetJobMinibatchTimeFromEtcd: end time is %v", end_time)
	log.Printf("FinishGetJobMinibatchTimeFromEtcd time is %v", end_time-start_time)
	// return batch_time

}

// get minibatch time of base jobs and new job from etcd
func GetBothMinibatchTimeFromEtcd(base_jobs []JobInfo, new_job JobInfo) []Batch_time {
	var job_batchtime []Batch_time
	var waitGroup sync.WaitGroup
	var chan_arr = make(chan Batch_time, len(base_jobs)+1)
	// var batch_times []float64
	complete_jobs := map[string]JobInfo{}
	defer close(chan_arr)
	for i := 0; i < len(base_jobs); i++ {
		log.Printf("base_jobs of pod %v start time is %v", base_jobs[i].Pod_name, time.Now().Unix())
		complete, _ := GetPodContentByEtcd(base_jobs[i].Pod_name, "complete")
		if complete == "1" {
			complete_jobs[base_jobs[i].Pod_name] = base_jobs[i]
		} else {
			waitGroup.Add(1)
			go GetJobMinibatchTimeFromEtcd(base_jobs[i].Pod_name, &waitGroup, chan_arr)
			// batch_time := GetJobMinibatchTimeFromEtcd(base_jobs[i].Pod_name, &waitGroup, chan_arr)

		}
		log.Printf("base_jobs of pod %v end time is %v", base_jobs[i].Pod_name, time.Now().Unix())

	}
	waitGroup.Add(1)
	log.Printf("new_job of %v start time is %v", new_job.Pod_name, time.Now().Unix())

	go GetJobMinibatchTimeFromEtcd(new_job.Pod_name, &waitGroup, chan_arr)
	// batch_times = append(batch_times, new_batch_time)
	// log.Printf("batchtimes are  %+v", batch_times)
	log.Printf("new_job of %v end time is %v", new_job.Pod_name, time.Now().Unix())
	waitGroup.Wait()

	// log.Printf("get minibatch time from etcd: %+v of pod %v", job_batchtime, new_job.Pod_name)

	log.Printf("waitGroup.Wait() done of pod %+v!", new_job.Pod_name)
	log.Printf("waitGroup.Wait() done of pod %+v!", new_job.Pod_name)

	for i := 0; i < len(base_jobs)+1; i++ {
		log.Printf("begin batchtime")
		var item Batch_time
		if i < len(base_jobs) {
			if complete_job, ok := complete_jobs[base_jobs[i].Pod_name]; ok {
				batchsize_str, err := GetPodContentByEtcd(complete_job.Pod_name, "batchsize")
				if err != nil {
					log.Printf("get minibatch from etcd err %+v", err)
				}
				batchsize, _ := strconv.Atoi(batchsize_str)
				item = Batch_time{
					Pod_name:  complete_job.Pod_name,
					Batchsize: batchsize,
					Time:      0.0,
				}

				log.Printf("complete_job.Pod_name : %+v", complete_job.Pod_name)
				// log.Printf("//Time is : %+v", batch_times[i])
				job_batchtime = append(job_batchtime, item)
				log.Printf("//job_batchtime is : %+v", job_batchtime)

			} else {
				item = <-chan_arr
				log.Printf("/item is : %+v", item)
			}
		} else {
			item = <-chan_arr
			log.Printf("//item is : %+v", item)
		}

		job_batchtime = append(job_batchtime, item)
		log.Printf("job_batchtime2 is : %+v", job_batchtime)
	}
	log.Printf("/ etcd of pods %+v", job_batchtime)

	return job_batchtime
}

// delete max and min, and then cal the avg
func GetAvg(durations []float64) float64 {
	var sum, avg float64
	// sort delete max and min
	// for i := 0; i < len(durations)-1; i++ {
	// 	for j := 0; j < len(durations)-1; j++ {
	// 		if durations[j] > durations[j+1] {
	// 			durations[j], durations[j+1] = durations[j+1], durations[j]
	// 		}
	// 	}
	// }
	// fmt.Println(durations)

	// avg
	sort.Float64sAreSorted(durations)
	for k := 1; k < len(durations); k++ {
		sum += durations[k]
	}
	avg = sum / float64(len(durations))
	log.Printf("Info: The average of minibatch durations is %v", avg)
	return avg

	//return first minibatch time
	// return durations[0]
}

func PredBasePodRemainingJct(pod_name string, epoch_time float64) float64 {

	cur_epoch, err := GetPodContentByEtcd(pod_name, "cur_epoch")
	if err != nil {
		log.Printf("Failed get cur_epoch %+v of pod %+v from etcd due to err %+v", cur_epoch, pod_name, err)
	}
	epoch, _ := strconv.ParseFloat(cur_epoch, 64)
	log.Printf("current epoch is %v", epoch)
	batchsize, err := GetPodContentByEtcd(pod_name, "batchsize")
	if err != nil {
		log.Printf("Failed get batchsize %+v of pod %+v from etcd due to err %+v", batchsize, pod_name, err)
	}
	job_name := strings.Split(pod_name, "-")[0]
	cur_batchsize, _ := strconv.Atoi(batchsize)
	fit_epoch := FitBatchsizeEpoch3(job_name, cur_batchsize)
	log.Printf("fit_epoch is: %v of pod %v", fit_epoch, pod_name)
	log.Printf("estimate epoch_time is: %v of pod %v", epoch_time, pod_name)

	remaining_jct := (float64(fit_epoch) - epoch) * epoch_time
	// remaining_jct := float64(fit_epoch) * epoch_time
	log.Printf("remaining_jct of pod %v is %v", pod_name, remaining_jct)
	return remaining_jct
}

func PredNewPodRemainingJct(pod_name string, epoch_time float64) float64 {

	batchsize, err := GetPodContentByEtcd(pod_name, "batchsize")
	if err != nil {
		log.Printf("Failed get batchsize %+v of pod %+v from etcd due to err %+v", batchsize, pod_name, err)
	}
	job_name := strings.Split(pod_name, "-")[0]
	cur_batchsize, _ := strconv.Atoi(batchsize)
	fit_epoch := FitBatchsizeEpoch3(job_name, cur_batchsize)

	remaining_jct := float64(fit_epoch) * epoch_time
	log.Printf("remaining_jct of pod %v is %v", pod_name, remaining_jct)
	return remaining_jct
}

func GetMinInterBatchsizes(base_jobs_batchs []JobInfo, new_job_batches JobInfo, job_name []string) []int {
	log.Printf("job_names_for_ucb: %+v", job_name)
	var interference_score []float64
	var batchsizes [][]int
	var best_batchsizes []int
	for i := 0; i < len(base_jobs_batchs[0].Batchsizes_times); i++ {

		// base_job_interferences := 0.0
		var base_job_batchsizes []int
		remaining_jcts := 0.0
		// var remaining_jcts []float64

		for j := 0; j < len(base_jobs_batchs); j++ {

			pod_name := base_jobs_batchs[j].Pod_name
			batchsize_time := base_jobs_batchs[j].Batchsizes_times[i]
			epoch_time := float64(base_jobs_batchs[j].Samples/batchsize_time.Batchsize) * float64(base_jobs_batchs[j].Batchsizes_times[i].Time)
			// log.Printf("estimate epoch_time of pod %v is %v", epoch_time, base_jobs_batchs[j].Pod_name)
			remaining_jct := PredBasePodRemainingJct(pod_name, epoch_time*1.2)
			// log.Printf("PredBasePodRemainingJct of pod %v is %v", base_jobs_batchs[j].Pod_name, remaining_jct)
			remaining_jcts = remaining_jcts + remaining_jct
			// remaining_jcts = append(remaining_jcts, remaining_jct)

			// epoch := FuncBatchsizeEpoch(base_jobs_batchs[j].Name, base_jobs_batchs[j].Batchsizes_times[i].Batchsize)
			// log.Printf("pred_epoch is %v", epoch)
			// batchsize_time := base_jobs_batchs[j].Batchsizes_times[i]
			// log.Printf("pred_iterations is %v", float64(base_jobs_batchs[j].Samples/batchsize_time.Batchsize))
			// pred_jct := float64(base_jobs_batchs[j].Samples/batchsize_time.Batchsize) * float64(base_jobs_batchs[j].Batchsizes_times[i].Time) * float64(epoch)
			// log.Printf("pred_jct is %v", pred_jct)
			// log.Printf("best_batchsize jct is %v", base_jobs_batchs[j].JCT)
			// base_job_interference := (pred_jct - base_jobs_batchs[j].JCT) / base_jobs_batchs[j].JCT
			// log.Printf("base_job_interference is %+v of pod %v", base_job_interference, base_jobs_batchs[j].Pod_name)
			// base_job_interferences += base_job_interference

			// base_job_batchsizes = append(base_job_batchsizes, batchsize_time.Batchsize)
			// if i == 0 {
			// 	best_batchsizes = append(best_batchsizes, base_jobs_batchs[j].Best_batchsize)
			// }
		}
		// epoch := FuncBatchsizeEpoch(new_job_batches.Name, new_job_batches.Batchsizes_times[i].Batchsize)
		// new_job_interference := (float64(new_job_batches.Samples/new_job_batches.Batchsizes_times[i].Batchsize)*float64(new_job_batches.Batchsizes_times[i].Time)*float64(epoch) - new_job_batches.JCT) / new_job_batches.JCT
		// log.Printf("new_job_interference is +%v of pod %v", new_job_interference, new_job_batches.Pod_name)
		// interference_score = append(interference_score, base_job_interferences+new_job_interference)

		new_pod_name := new_job_batches.Pod_name
		epoch_time := float64(new_job_batches.Samples/new_job_batches.Batchsizes_times[i].Batchsize) * float64(new_job_batches.Batchsizes_times[i].Time)
		remaining_jct := PredNewPodRemainingJct(new_pod_name, epoch_time*1.2)
		log.Printf("PredNewPodRemainingJct of pod %v is %v", new_pod_name, remaining_jct)
		remaining_jcts = remaining_jcts + remaining_jct
		// remaining_jcts = append(remaining_jcts, remaining_jct)
		// sort.Float64sAreSorted(remaining_jcts)
		interference_score = append(interference_score, remaining_jcts)

		// base_job_batchsizes = append(base_job_batchsizes, new_job_batches.Batchsizes_times[len(new_job_batches.Batchsizes_times)-1-i].Batchsize) //逆序
		base_job_batchsizes = append(base_job_batchsizes, new_job_batches.Batchsizes_times[i].Batchsize) //顺序
		batchsizes = append(batchsizes, base_job_batchsizes)
	}
	best_batchsizes = append(best_batchsizes, new_job_batches.Best_batchsize)

	resdata := ReqData{
		Job_names:         job_name,
		Best_batchsizes:   best_batchsizes,
		Batchsize_info:    batchsizes,
		Interference_info: interference_score,
		Threshold:         100.0,
	}
	log.Printf("send to ucb-server reqdata is %+v", resdata)
	// log.Printf("res is SendTuneRequest(resdata): %v", SendTuneRequest(resdata))
	// 记录tune的数据
	bytesData, _ := json.Marshal(resdata)
	_, err := PutPodContentByEtcd(new_job_batches.Pod_name, "tune_data", string(bytesData))
	if err != nil {
		log.Printf("PutPodContentByEtcd error: %v", err)
	}
	//记录完毕！
	return SendTuneRequest(resdata)
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
	fmt.Println("response Status:", response.Status)
	fmt.Println("response Headers:", response.Header)
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

func SendDataToTuner(requestData RequestData, ip_addr string) error {

	bytesData, err := json.Marshal(requestData)

	if err != nil {
		log.Fatal(err)
	}
	start_req := time.Now().Unix()
	request, err := http.NewRequest("POST", "http://"+ip_addr+":8888", bytes.NewBuffer(bytesData))
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
		return err
	}
	defer request.Body.Close()
	fmt.Println("response Status:", response.Status)
	fmt.Println("response Headers:", response.Header)
	respBytes, err := ioutil.ReadAll(response.Body)
	log.Printf("SendTuneRequest respBytes is %+v", string(respBytes))

	if err != nil {
		log.Printf("Failed to convert response to respBytes from client due to %v", err.Error())
		return err
	}

	respResult := RespResult{}
	jsonErr := json.Unmarshal(respBytes, &respResult)
	if jsonErr != nil {
		log.Fatal(jsonErr)
	}
	end_req := time.Now().Unix()
	log.Printf("SendTuneRequest request time is %+v", end_req-start_req)
	return nil
}

// fit functions of batchsize and epoch
func FuncBatchsizeEpoch(jobname string, batchsize int) int {
	x := float64(batchsize)
	y := 0.0
	switch jobname {
	case "vgg16":
		y = 2.743e-12*math.Pow(x, 4) - 2.13e-8*math.Pow(x, 3) + 5.079e-05*math.Pow(x, 2) - 0.03461*x + 21.54
	case "vgg19":
		y = 1.365e-12*math.Pow(x, 4) - 1.091e-8*math.Pow(x, 3) + 2.717e-05*math.Pow(x, 2) - 0.01505*x + 16.65
	case "squeezenet":
		y = 1.018e-12*math.Pow(x, 4) - 1.451e-8*math.Pow(x, 3) + 5.966e-05*math.Pow(x, 2) - 0.05378*x + 48.99
	case "googlenet":
		y = 1.261e-11*math.Pow(x, 4) - 9.753e-08*math.Pow(x, 3) + 0.0002436*math.Pow(x, 2) - 0.2471*x + 130.4
	case "alexnet":
		y = 4.969e-10*math.Pow(x, 4) - 1.882e-06*math.Pow(x, 3) + 0.002193*math.Pow(x, 2) - 0.9125*x + 132
	case "resnet152":
		y = 2.635e-11*math.Pow(x, 4) - 9.407e-08*math.Pow(x, 3) + 9.876e-05*math.Pow(x, 2) - 0.04027*x + 25.7
	case "neumf":
		y = 2.476e-14*math.Pow(x, 4) - 3.435e-10*math.Pow(x, 3) + 1.287e-06*math.Pow(x, 2) - 0.0009933*x + 10.12
	case "adgcl":
		y = -5.366e-07*math.Pow(x, 4) + 0.0001549*math.Pow(x, 3) + 0.0004747*math.Pow(x, 2) - 0.33*x + 72.56
	default:
		y = 0.0
	}
	return int(math.Ceil(y))
}

func FitBatchsizeEpoch3(jobname string, batchsize int) float64 {
	x := float64(batchsize)
	y := 0.0
	switch jobname {
	case "vgg16":
		y = -1.235e-13*math.Pow(x, 4) - 2.105e-09*math.Pow(x, 3) + 1.13e-05*math.Pow(x, 2) - 0.01093*x + 20.66
	case "vgg19":
		y = 1.72e-10*math.Pow(x, 4) - 3.192e-06*math.Pow(x, 3) + 0.02557*math.Pow(x, 2) - 15.84*x + 8670
		return y / x
	case "squeezenet":
		y = 2.855e-10*math.Pow(x, 4) - 4.811e-06*math.Pow(x, 3) + 0.04797*math.Pow(x, 2) - 6.652*x + 8270
		return y / x
	case "googlenet":
		y = 1.645e-13*math.Pow(x, 4) - 3.138e-09*math.Pow(x, 3) + 2.108e-05*math.Pow(x, 2) - 0.06284*x + 118.4
	case "alexnet":
		y = 9.907e-11*math.Pow(x, 4) - 4.494e-07*math.Pow(x, 3) + 0.0007033*math.Pow(x, 2) - 0.4296*x + 102.1
	case "resnet152":
		y = 5.501e-10*math.Pow(x, 3) - 2.25e-07*math.Pow(x, 2) - 0.007032*x + 24.23
	case "neumf":
		y = 1.05e-11*math.Pow(x, 3) - 1.728e-07*math.Pow(x, 2) + 0.0008914*x + 9.588
	case "adgcl":
		y = -0.0001112*math.Pow(x, 4) + 0.04306*math.Pow(x, 3) - 3.056*math.Pow(x, 2) + 142.3*x - 576.2
		return y / x
	default:
		y = 0.0
	}
	// return int(math.Ceil(y))
	return y
}

func FitBatchsizeEpoch2(jobname string, batchsize int) float64 {
	x := float64(batchsize)
	y := 0.0
	switch jobname {
	case "vgg16":
		y = -9.06e-10*math.Pow(x, 4) + 5.033e-06*math.Pow(x, 3) + 0.001095*math.Pow(x, 2) + 13.24*x + 381.4
	case "vgg19":
		y = -9.713e-07*math.Pow(x, 3) + 0.01344*math.Pow(x, 2) + 6.042*x + 888.4
	case "squeezenet":
		y = 6.338e-10*math.Pow(x, 4) - 8.367e-06*math.Pow(x, 3) + 0.05242*math.Pow(x, 2) + 0.1664*x + 3903
	case "googlenet":
		y = 1.067e-10*math.Pow(x, 4) + 6.121e-06*math.Pow(x, 3) - 0.0289*math.Pow(x, 2) + 60.07*x + 5273
	case "alexnet":
		y = -4.382e-06*math.Pow(x, 3) + 0.02455*math.Pow(x, 2) - 3.395*x + 5103
	case "resnet152":
		y = 2.87e-06*math.Pow(x, 3) - 0.0119*math.Pow(x, 2) + 25.45*x - 231.7
	case "neumf":
		y = -2e-08*math.Pow(x, 3) + 0.0002333*math.Pow(x, 2) + 10.45*x - 164.4
	case "adgcl":
		y = -0.0001112*math.Pow(x, 4) + 0.04306*math.Pow(x, 3) - 3.056*math.Pow(x, 2) + 142.3*x - 576.2
	default:
		y = 0.0
	}
	// return int(math.Ceil(y))
	return y / x
}

// fit functions of batchsize and epoch use 3
func FitBatchsizeEpoch(jobname string, batchsize int) float64 {
	x := float64(batchsize)
	y := 0.0
	switch jobname {
	case "vgg16":
		y = -1.99e-09*math.Pow(x, 3) + 1.225e-05*math.Pow(x, 2) + 0.01169*x + 19.03
	case "vgg19":
		y = -1.302e-09*math.Pow(x, 3) + 7.99e-06*math.Pow(x, 2) - 0.003637*x + 15.3
	case "squeezenet":
		y = -3.241e-10*math.Pow(x, 3) + 4.319e-06*math.Pow(x, 2) + 0.008145*x + 38.41
	case "googlenet":
		y = -8.719e-09*math.Pow(x, 3) + 6.636e-05*math.Pow(x, 2) + -0.1416*x + 118.8
	case "alexnet":
		y = -1.046e-07*math.Pow(x, 3) + 0.0003542*math.Pow(x, 2) - 0.3148*x + 88.71
	case "resnet152":
		y = 2.141e-10*math.Pow(x, 3) + 1.267e-06*math.Pow(x, 2) - 0.008578*x + 23.4
	case "neumf":
		y = 1.652e-12*math.Pow(x, 3) - 5.929e-08*math.Pow(x, 2) + 0.0005131*x + 9.862
	case "adgcl":
		y = -9.148e-05*math.Pow(x, 3) + 0.03417*math.Pow(x, 2) - 1.884*x + 91.69
	default:
		y = 0.0
	}
	// return int(math.Ceil(y))
	return y
}

// use etcd to write content
func PutPodContentByEtcd(first_key, second_key, content string) (bool, error) {
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

// use etcd to write content
func GetPodContentByEtcd(pod_name, key string) (string, error) {
	key = "/gpushare/" + pod_name + "/" + key
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
	log.Printf("Get value %+v of key %+v", resp.Kvs, key)
	if len(resp.Kvs) > 0 {
		for _, kv := range resp.Kvs {
			log.Printf("key:%v, value:%+v", kv.Key, kv.Value)
			return string(kv.Value), nil
		}
	}
	return "", err
}

func WatchKeyByEtcd(first_key, second_key string) (string, error) {
	// event type 0:PUT 1:Delete 2:Expire
	second_key = "/gpushare/" + first_key + "/" + second_key
	client, err := CreateEtcdClient()
	if err != nil {
		log.Printf("error: get connect to etcd err due to %+v", err)
	}
	defer client.Close()
	for {
		ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
		rch := client.Watch(ctx, second_key)
		defer cancel()
		select {
		case <-ctx.Done():
			log.Printf("error: %v", ctx.Err())
			return "", ctx.Err()
		case wresp := <-rch:
			for _, ev := range wresp.Events {
				log.Printf("%s %q :%q\n", ev.Type, ev.Kv.Key, ev.Kv.Value)
				if ev.Type == clientv3.EventTypePut {
					return string(ev.Kv.Value), nil
				}
			}
		}
	}
}

func DeleteAllPodStatusContentByEtcd() (bool, error) {
	key := "/gpushare/pod_status/"
	client, err := CreateEtcdClient()
	if err != nil {
		log.Printf("error: get connect to etcd err due to %+v", err)
	}
	defer client.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	resp, err := client.Get(ctx, key, clientv3.WithPrefix())
	if err != nil {
		log.Printf("error: failed to get pod status of due to %+v", err)
		cancel()
		return false, err
	}
	if len(resp.Kvs) > 0 {
		for _, kv := range resp.Kvs {
			log.Printf("key:%v, value:%+v", kv.Key, kv.Value)
			client.Delete(ctx, string(kv.Key))
		}
	}
	cancel()
	return true, nil
}

func DeletePodContentByEtcd(pod_name, key string) (bool, error) {
	key = "/gpushare/" + pod_name + "/" + key
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
		DialTimeout: 10 * time.Second,
		TLS:         _tlsConfig,
	})
	if err != nil {
		log.Printf("error: get connect to etcd err due to %+v", err)
		return nil, err
	}

	return client, err
}

// sorted job name list by pod list
func SortJobStr(podList []*v1.Pod, newPod *v1.Pod) string {
	var podStr []string
	for _, item := range podList {
		podStr = append(podStr, item.Name)
	}
	if newPod != nil {
		podStr = append(podStr, newPod.Name)
	}
	sort.Strings(podStr)
	return strings.Join(podStr, ",")
}

func GetTuningStatusFromEtcd(nodeName string, devID string) bool {
	key := "/gpushare/" + nodeName + "/" + devID
	client, _ := CreateEtcdClient()
	defer client.Close()
	resp, _ := client.Get(context.TODO(), key)
	log.Printf("Get value %+v of key %+v", resp.Kvs, key)
	if len(resp.Kvs) > 0 {
		for _, kv := range resp.Kvs {
			var pods []string
			_ = json.Unmarshal([]byte(kv.Value), &pods)
			log.Printf("key:%v, value:%+v", string(kv.Key), pods)
			return len(pods) > 1
		}
	}
	return false
}
