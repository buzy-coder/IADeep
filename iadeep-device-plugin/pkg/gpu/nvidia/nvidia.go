package nvidia

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"

	log "github.com/golang/glog"

	"github.com/NVIDIA/gpu-monitoring-tools/bindings/go/nvml"

	"golang.org/x/net/context"
	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/deviceplugin/v1beta1"
)

var (
	gpuMemory uint
	metric    MemoryUnit
)

type GPUStatus struct {
	UID         uint
	MemCapacity uint
	MemUsed     uint
	GpuUtil     uint
	MemUtil     uint
	Process     []ProcessDetail
}

type ProcessDetail struct {
	PID        uint
	Name       string
	MemoryUsed uint64
}

func check(err error) {
	if err != nil {
		log.Fatalln("Fatal:", err)
	}
}

func generateFakeDeviceID(realID string, fakeCounter uint) string {
	return fmt.Sprintf("%s-_-%d", realID, fakeCounter)
}

func extractRealDeviceID(fakeDeviceID string) string {
	return strings.Split(fakeDeviceID, "-_-")[0]
}

func setGPUMemory(raw uint) {
	v := raw
	if metric == GiBPrefix {
		v = raw / 1024
	}
	gpuMemory = v
	log.Infof("set gpu memory: %d", gpuMemory)
}

func getGPUMemory() uint {
	return gpuMemory
}

func getDeviceCount() uint {
	n, err := nvml.GetDeviceCount()
	check(err)
	return n
}

func getProcessInfo(pro nvml.ProcessInfo) ProcessDetail {
	var processDetail ProcessDetail
	pid := strconv.Itoa(int(pro.PID))
	cmd := exec.Command("cat", "/root/proc/"+pid+"/cmdline")
	out, err := cmd.Output()
	log.Infof("output: %v", string(out))
	check(err)
	job_name := strings.Replace(string(out), " ", "", -1)
	job_name = strings.Replace(job_name, "python", "", -1)
	// name := strings.Replace(job_name, "workloads/", "", -1)
	// name = strings.Split(name, "--")[0]
	// name_str := strings.Replace(name, ".py", "", -1)
	name_str := strings.Replace(job_name, "/workspace/workloads/", "", -1)
	name_str = strings.Split(name_str, ".py")[0]
	log.Infof("output: %v", name_str)

	processDetail = ProcessDetail{
		PID:        pro.PID,
		Name:       name_str,
		MemoryUsed: pro.MemoryUsed,
	}
	return processDetail
}

func getDevices() ([]*pluginapi.Device, map[string]uint, []GPUStatus) {
	n, err := nvml.GetDeviceCount()
	check(err)

	var devs []*pluginapi.Device
	realDevNames := map[string]uint{}
	gpuStatus := []GPUStatus{}

	for i := uint(0); i < n; i++ {
		d, err := nvml.NewDevice(i)
		check(err)
		// realDevNames = append(realDevNames, d.UUID)
		var id uint
		log.Infof("Deivce %s's Path is %s", d.UUID, d.Path)
		_, err = fmt.Sscanf(d.Path, "/dev/nvidia%d", &id)
		check(err)
		realDevNames[d.UUID] = id
		// var KiB uint64 = 1024
		log.Infof("# device Memory: %d", uint(*d.Memory))
		if getGPUMemory() == uint(0) {
			setGPUMemory(uint(*d.Memory))
		}

		status, err := d.Status()
		check(err)

		// GPU util and Mem util
		var processes []ProcessDetail
		for _, item := range status.Processes {
			log.Infof("# process: %T", item.Type.String())
			if item.Type.String() == "C" {
				processes = append(processes, getProcessInfo(item))
			}
		}
		log.Infof("process info: %v", status.Processes)
		memCapacity := uint(*status.Memory.Global.Free/1024) + uint(*status.Memory.Global.Used/1024)
		gpuStatus = append(gpuStatus, GPUStatus{
			UID:         id,
			MemCapacity: memCapacity,
			MemUsed:     uint(*status.Memory.Global.Used/1024),
			GpuUtil:     uint(*status.Utilization.GPU),
			MemUtil:     uint(*status.Utilization.Memory),
			Process:     processes,
		})

		for j := uint(0); j < getGPUMemory(); j++ {
			fakeID := generateFakeDeviceID(d.UUID, j)
			if j == 0 {
				log.Infoln("# Add first device ID: " + fakeID)
			}
			if j == getGPUMemory()-1 {
				log.Infoln("# Add last device ID: " + fakeID)
			}
			devs = append(devs, &pluginapi.Device{
				ID:     fakeID,
				Health: pluginapi.Healthy,
			})
		}
	}
	log.Infof("debug: get gpuStatus is %+v", gpuStatus)
	return devs, realDevNames, gpuStatus
}

func deviceExists(devs []*pluginapi.Device, id string) bool {
	for _, d := range devs {
		if d.ID == id {
			return true
		}
	}
	return false
}

func watchXIDs(ctx context.Context, devs []*pluginapi.Device, xids chan<- *pluginapi.Device) {
	eventSet := nvml.NewEventSet()
	defer nvml.DeleteEventSet(eventSet)

	for _, d := range devs {
		realDeviceID := extractRealDeviceID(d.ID)
		err := nvml.RegisterEventForDevice(eventSet, nvml.XidCriticalError, realDeviceID)
		if err != nil && strings.HasSuffix(err.Error(), "Not Supported") {
			log.Infof("Warning: %s (%s) is too old to support healthchecking: %s. Marking it unhealthy.", realDeviceID, d.ID, err)

			xids <- d
			continue
		}

		if err != nil {
			log.Fatalf("Fatal error:", err)
		}
	}

	for {
		select {
		case <-ctx.Done():
			return
		default:
		}

		e, err := nvml.WaitForEvent(eventSet, 5000)
		if err != nil && e.Etype != nvml.XidCriticalError {
			continue
		}

		// FIXME: formalize the full list and document it.
		// http://docs.nvidia.com/deploy/xid-errors/index.html#topic_4
		// Application errors: the GPU should still be healthy
		if e.Edata == 31 || e.Edata == 43 || e.Edata == 45 {
			continue
		}

		if e.UUID == nil || len(*e.UUID) == 0 {
			// All devices are unhealthy
			for _, d := range devs {
				xids <- d
			}
			continue
		}

		for _, d := range devs {
			if extractRealDeviceID(d.ID) == *e.UUID {
				xids <- d
			}
		}
	}
}
