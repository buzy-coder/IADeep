package utils

import (
	"encoding/json"
	"log"

	v1 "k8s.io/api/core/v1"
)

type ProcessInfo struct {
	PID        uint   `json:"PID"`
	Name       string `json:"Name"`
	MemoryUsed uint64 `json:"MemoryUsed"`
}

type GPUStatus struct {
	UID         uint          `json:"UID"`
	MemCapacity uint          `json:"MemCapacity"`
	MemUsed     uint          `json:"MemUsed"`
	GpuUtil     uint          `json:"GpuUtil"`
	MemUtil     uint          `json:"MemUtil"`
	Process     []ProcessInfo `json:"Process"`
}

// Is the Node for GPU sharing
func IsGPUSharingNode(node *v1.Node) bool {
	return GetTotalGPUMemory(node) > 0
}

// Get the total GPU memory of the Node
func GetTotalGPUMemory(node *v1.Node) int {
	val, ok := node.Status.Capacity[ResourceName]

	if !ok {
		return 0
	}
	log.Printf("debug: GetTotalGPUMemory is %v of node %v", int(val.Value()), node.Name)
	return int(val.Value())
}

// Get the GPU count of the node
func GetGPUCountInNode(node *v1.Node) int {
	val, ok := node.Status.Capacity[CountName]

	if !ok {
		return int(0)
	}

	return int(val.Value())
}

// Get the GPU memory capacity of the node
func GetMemCapacityInNode(node *v1.Node) []int {
	val, ok := node.Annotations[MemCapacity]
	var memCapacity []int
	if !ok {
		return make([]int, 0)
	}
	err := json.Unmarshal([]byte(val), &memCapacity)
	if err != nil {
		log.Printf("error: %v", err)
	}
	return memCapacity
}

// Get the GPU memory used of the node
func GetGPUStatusInNode(node *v1.Node) []GPUStatus {
	val, ok := node.Annotations[GPUInfo]
	var gpuStatus []GPUStatus
	if !ok {
		return gpuStatus
	}
	err := json.Unmarshal([]byte(val), &gpuStatus)
	if err != nil {
		log.Printf("error: %v", err)
	}
	// log.Printf("GetGPUStatusInNode is %+v", gpuStatus)
	return gpuStatus
}

// Get the GPU memory used of the node
func GetGPUMemUsedInNode(node *v1.Node) []int {
	val, ok := node.Annotations[MemUsed]
	var memUsed []int
	if !ok {
		return make([]int, 0)
	}
	err := json.Unmarshal([]byte(val), &memUsed)
	if err != nil {
		log.Printf("error: %v", err)
	}
	return memUsed
}

// Get the GPU processes of the node
func GetGPUProcessesInNode(node *v1.Node) [][]ProcessInfo {
	val, ok := node.Annotations[GPUProcesses]
	log.Printf("processes: %v", val)
	var gpuProcesses [][]ProcessInfo
	if !ok {
		log.Printf("GetGPUProcessesInNode ok: %v", ok)
		return make([][]ProcessInfo, 0)
	}
	err := json.Unmarshal([]byte(val), &gpuProcesses)
	if err != nil {
		log.Printf("error: %v", err)
	}
	log.Printf("gpuProcesses: %v", gpuProcesses)
	return gpuProcesses
}

// Get the GPU utilization of the node
func GetGPUUtilInNode(node *v1.Node) []int {
	val, ok := node.Annotations[GPUUtilization]
	var gpuUtil []int
	if !ok {
		return make([]int, 0)
	}
	err := json.Unmarshal([]byte(val), &gpuUtil)
	if err != nil {
		log.Printf("error: %v", err)
	}
	return gpuUtil
}

// Get the mem utilization of the node
func GetMemUtilInNode(node *v1.Node) []int {
	val, ok := node.Annotations[MemUtilization]
	var memUtil []int
	if !ok {
		return make([]int, 0)
	}
	err := json.Unmarshal([]byte(val), &memUtil)
	if err != nil {
		log.Printf("error: %v", err)
	}
	return memUtil
}
