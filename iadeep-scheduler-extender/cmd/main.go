package main

import (
	"flag"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"gpushare-scheduler-extender/pkg/gpushare"
	"gpushare-scheduler-extender/pkg/methods"
	"gpushare-scheduler-extender/pkg/routes"
	"gpushare-scheduler-extender/pkg/scheduler"
	"gpushare-scheduler-extender/pkg/utils/signals"

	"github.com/comail/colog"
	"github.com/julienschmidt/httprouter"

	kubeinformers "k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/workqueue"
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
	if err != nil {
		log.Fatalf("Error building kubeconfig: %s", err.Error())
	}

	// create the clientset
	clientset, err = kubernetes.NewForConfig(restConfig)
	if err != nil {
		log.Fatalf("fatal: Failed to init rest config due to %v", err)
	}
}

func main() {
	// Call Parse() to avoid noisy logs
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
	// threadness := StringToInt(os.Getenv("THREADNESS"))
	threadness := StringToInt(os.Getenv("THREADNESS"))
	log.Print("THREADNESS was set to ", threadness)

	initKubeClient()
	methods.DeleteAllPodStatusContentByEtcd()
	// delete content for devId 1 of node cc232
	methods.DeletePodContentByEtcd("cc232", "1")
	methods.DeletePodContentByEtcd("vgg16-01", "")
	methods.DeletePodContentByEtcd("vgg16-02", "")
	methods.DeletePodContentByEtcd("squeezenet", "")

	port := os.Getenv("PORT")
	if _, err := strconv.Atoi(port); err != nil {
		port = "39999"
	}
	// Set up signals so we handle the first shutdown signal gracefully.
	stopCh := signals.SetupSignalHandler()

	tuningQueue := workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "tuningQueue")

	informerFactory := kubeinformers.NewSharedInformerFactory(clientset, resyncPeriod)
	controller, err := gpushare.NewController(clientset, informerFactory, tuningQueue, stopCh)
	if err != nil {
		log.Fatalf("Failed to start due to %v", err)
	}
	err = controller.BuildCache()
	if err != nil {
		log.Fatalf("Failed to start due to %v", err)
	}

	go controller.Run(threadness, stopCh)

	gpusharePredicate := scheduler.NewGPUsharePredicate(clientset, controller.GetSchedulerCache())
	gpusharePrioritize := scheduler.NewGPUsharePrioritize(clientset, controller.GetSchedulerCache())
	gpushareBind := scheduler.NewGPUShareBind(clientset, controller.GetSchedulerCache())
	gpushareInspect := scheduler.NewGPUShareInspect(controller.GetSchedulerCache())

	router := httprouter.New()

	routes.AddPProf(router)
	routes.AddVersion(router)
	routes.AddPredicate(router, gpusharePredicate)
	routes.AddPrioritize(router, gpusharePrioritize)
	routes.AddBind(router, gpushareBind)
	routes.AddInspect(router, gpushareInspect)

	log.Printf("info: server starting on the port :%s", port)
	if err := http.ListenAndServe(":"+port, router); err != nil {
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
	if len(sThread) == 0 {
		return 10
	}
	thread, err := strconv.ParseInt(sThread, 0, 0)
	if err != nil {
		log.Printf("Parse string err due to %+v", err)
	}
	return int(thread)
}
