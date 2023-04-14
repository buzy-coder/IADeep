package utils

const (
	ResourceName   = "gpushare/gpu-mem"
	CountName      = "gpushare/gpu-count"
	MemCapacity    = "gpushare/mem-capacity"
	MemUsed        = "gpushare/mem-used"
	GPUUtilization = "gpushare/gpu-util"
	MemUtilization = "gpushare/mem-util"
	GPUProcesses   = "gpushare/gpu-processes"
	HasProcesses   = "gpushare/has-processes"
	GPUInfo        = "gpushare/gpu-info"

	GPUUtilThreshold  = 99
	MemUtiliThreshold = 99
	TotalRecords      = 21

	EnvNVGPU              = "NVIDIA_VISIBLE_DEVICES"
	EnvResourceIndex      = "ALIYUN_COM_GPU_MEM_IDX"
	EnvResourceByPod      = "ALIYUN_COM_GPU_MEM_POD"
	EnvResourceByDev      = "ALIYUN_COM_GPU_MEM_DEV"
	EnvAssignedFlag       = "ALIYUN_COM_GPU_MEM_ASSIGNED"
	EnvResourceAssumeTime = "ALIYUN_COM_GPU_MEM_ASSUME_TIME"
	EnvPodName            = "POD_NAME"
)
