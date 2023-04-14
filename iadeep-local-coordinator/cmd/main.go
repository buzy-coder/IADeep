package main

import (
	"flag"
	"iadeep-local-coordinator/tuning"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/comail/colog"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/workqueue"

	"iadeep-local-coordinator/utils"
)

const RecommendedKubeConfigPathEnv = "KUBECONFIG"

var (
	clientset    *kubernetes.Clientset
	resyncPeriod = 30 * time.Second
	clientConfig clientcmd.ClientConfig
)

func initKubeClient() {
	kubeConfig := ""
	if len(os.Getenv(RecommendedKubeConfigPathEnv)) > 0 {
		// use the current context in kubeconfig
		// This is very useful for running locally.
		kubeConfig = os.Getenv(RecommendedKubeConfigPathEnv)
	}

	// Get kubernetes config.
	restConfig, err := clientcmd.BuildConfigFromFlags("", kubeConfig)
	// restConfig, err := rest.InClusterConfig()

	if err != nil {
		log.Fatalf("Error building kubeconfig: %s", err.Error())
	}

	// create the clientset
	// clientset, err = kubernetes.NewForConfig(restConfig)
	// if err != nil {
	// 	log.Fatalf("fatal: Failed to init rest config due to %v", err)
	// }

	clientset = kubernetes.NewForConfigOrDie(restConfig)
}

func main() {
	flag.CommandLine.Parse([]string{})
	colog.SetDefaultLevel(colog.LInfo)
	colog.SetMinLevel(colog.LInfo)
	colog.SetFormatter(&colog.StdFormatter{
		Colors: true,
		Flag:   log.Ldate | log.Ltime | log.Lshortfile,
	})
	colog.Register()
	level := StringToLevel(os.Getenv("LOG_LEVEL"))
	log.Print("Log level was set to ", strings.ToUpper(level.String()))
	colog.SetMinLevel(level)

	initKubeClient()
	port := os.Getenv("PORT")
	if _, err := strconv.Atoi(port); err != nil {
		port = "39999"
	}

	threadness := StringToInt(os.Getenv("THREADNESS"))
	// threadness := 10
	log.Print("THREADNESS was set to ", threadness)

	stopCh := utils.SetupSignalHandler()

	tuningQueue := workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "tuningQueue")

	nodeName := os.Getenv("NODE_NAME")
	log.Printf("nodeName is: %+v", nodeName)
	go utils.WatchPrefixByEtcd(nodeName, tuningQueue)

	controller, err := tuning.NewTuningController(clientset, tuningQueue, stopCh)

	if err != nil {
		log.Fatalf("Failed to create new tuning controller due to %v", err)
	}

	go controller.Run(threadness, stopCh)

	log.Printf("info: server starting on the port :%s", port)
	if err := http.ListenAndServe(":"+port, nil); err != nil {
		log.Fatal(err)
	}

}

func StringToLevel(levelStr string) colog.Level {
	switch level := strings.ToUpper(levelStr); level {
	case "TRACE":
		return colog.LTrace
	case "DEBUG":
		return colog.LDebug
	case "INFO":
		return colog.LInfo
	case "WARNING":
		return colog.LWarning
	case "ERROR":
		return colog.LError
	case "ALERT":
		return colog.LAlert
	default:
		log.Printf("warning: LOG_LEVEL=\"%s\" is empty or invalid, fallling back to \"INFO\".\n", level)
		return colog.LInfo
	}
}

func StringToInt(sThread string) int {
	thread, err := strconv.ParseInt(sThread, 0, 0)
	if err != nil {
		log.Printf("Parse string err due to %+v", err)
	}
	return int(thread)
}
