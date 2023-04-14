package routes

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"

	"github.com/julienschmidt/httprouter"

	"gpushare-scheduler-extender/pkg/scheduler"

	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
)

const (
	versionPath       = "/version"
	apiPrefix         = "/gpushare-scheduler"
	bindPrefix        = apiPrefix + "/bind"
	predicatesPrefix  = apiPrefix + "/filter"
	prioritizePrefix  = apiPrefix + "/priorities"
	inspectPrefix     = apiPrefix + "/inspect/:nodename"
	inspectListPrefix = apiPrefix + "/inspect"
)

var (
	version = "0.1.0"
	// mu      sync.RWMutex
)

func checkBody(w http.ResponseWriter, r *http.Request) {
	if r.Body == nil {
		http.Error(w, "Please send a request body", 400)
		return
	}
}

func InspectRoute(inspect *scheduler.Inspect) httprouter.Handle {
	return func(w http.ResponseWriter, r *http.Request, ps httprouter.Params) {
		result := inspect.Handler(ps.ByName("nodename"))

		if resultBody, err := json.Marshal(result); err != nil {
			// panic(err)
			log.Printf("warn: Failed due to %v", err)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusInternalServerError)
			errMsg := fmt.Sprintf("{'error':'%s'}", err.Error())
			w.Write([]byte(errMsg))
		} else {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			w.Write(resultBody)
		}
	}
}

func PredicateRoute(predicate *scheduler.Predicate) httprouter.Handle {
	return func(w http.ResponseWriter, r *http.Request, _ httprouter.Params) {
		checkBody(w, r)

		// mu.RLock()
		// defer mu.RUnlock()

		var buf bytes.Buffer
		body := io.TeeReader(r.Body, &buf)
		// log.Print("info: ", predicate.Name, " ExtenderArgs = ", buf.String())

		var extenderArgs schedulerapi.ExtenderArgs
		var extenderFilterResult *schedulerapi.ExtenderFilterResult

		if err := json.NewDecoder(body).Decode(&extenderArgs); err != nil {

			log.Printf("warn: failed to parse request due to error %v", err)
			extenderFilterResult = &schedulerapi.ExtenderFilterResult{
				Nodes:       nil,
				FailedNodes: nil,
				Error:       err.Error(),
			}
		} else {
			// log.Printf("debug: gpusharingfilter ExtenderArgs =%v", extenderArgs)
			log.Printf("extenderArgs.nodeNames: %v", extenderArgs.NodeNames)

			extenderFilterResult = predicate.Handler(extenderArgs)
		}

		if resultBody, err := json.Marshal(extenderFilterResult); err != nil {
			// panic(err)
			log.Printf("warn: Failed due to %v", err)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusInternalServerError)
			errMsg := fmt.Sprintf("{'error':'%s'}", err.Error())
			w.Write([]byte(errMsg))
		} else {
			log.Print("info: ", predicate.Name, " extenderFilterResult = ", string(resultBody))
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			w.Write(resultBody)
		}
	}
}

//wychen: prioritize the nodes
func PrioritizeRoute(prioritize *scheduler.Prioritize) httprouter.Handle {
	log.Printf("PrioritizeRoute_ :")
	return func(w http.ResponseWriter, r *http.Request, _ httprouter.Params) {
		checkBody(w, r)
		// mu.RLock()
		// defer mu.RUnlock()
		log.Printf("PrioritizeRoute:")
		var buf bytes.Buffer
		body := io.TeeReader(r.Body, &buf)
		// log.Print("info: ", predicate.Name, " ExtenderArgs = ", buf.String())

		var extenderArgs schedulerapi.ExtenderArgs
		var hostPriorityList schedulerapi.HostPriorityList

		if err := json.NewDecoder(body).Decode(&extenderArgs); err != nil {

			log.Printf("warn: failed to parse request due to error %v", err)
			// hostPriorityList = make(map[*schedulerapi.HostPriority], 0)
		} else {
			// log.Printf("debug: gpusharingPrioritize ExtenderArgs =%v", extenderArgs)
			log.Printf("gpusharingPrioritize.nodeNames: %v", extenderArgs.NodeNames)
			hostPriorityList = prioritize.Handler(extenderArgs)
		}

		if resultBody, err := json.Marshal(hostPriorityList); err != nil {
			// panic(err)
			log.Printf("warn: Failed due to %v", err)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusInternalServerError)
			errMsg := fmt.Sprintf("{'error':'%s'}", err.Error())
			w.Write([]byte(errMsg))
		} else {
			log.Print("info: ", prioritize.Name, " hostPriorityList = ", string(resultBody))
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			w.Write(resultBody)
		}
	}
}

func BindRoute(bind *scheduler.Bind) httprouter.Handle {
	return func(w http.ResponseWriter, r *http.Request, _ httprouter.Params) {
		checkBody(w, r)

		// mu.Lock()
		// defer mu.Unlock()
		var buf bytes.Buffer
		body := io.TeeReader(r.Body, &buf)
		// log.Print("info: extenderBindingArgs = ", buf.String())

		var extenderBindingArgs schedulerapi.ExtenderBindingArgs
		var extenderBindingResult *schedulerapi.ExtenderBindingResult
		failed := false

		if err := json.NewDecoder(body).Decode(&extenderBindingArgs); err != nil {
			extenderBindingResult = &schedulerapi.ExtenderBindingResult{
				Error: err.Error(),
			}
			failed = true
		} else {
			log.Printf("debug: gpusharingBind ExtenderArgs =%v", extenderBindingArgs)
			extenderBindingResult = bind.Handler(extenderBindingArgs)
		}

		if len(extenderBindingResult.Error) > 0 {
			failed = true
		}

		if resultBody, err := json.Marshal(extenderBindingResult); err != nil {
			log.Printf("warn: Failed due to %v", err)
			// panic(err)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusInternalServerError)
			errMsg := fmt.Sprintf("{'error':'%s'}", err.Error())
			w.Write([]byte(errMsg))
		} else {
			log.Print("info: extenderBindingResult = ", string(resultBody))
			w.Header().Set("Content-Type", "application/json")
			if failed {
				w.WriteHeader(http.StatusInternalServerError)
			} else {
				w.WriteHeader(http.StatusOK)
			}

			w.Write(resultBody)
		}
	}
}

func VersionRoute(w http.ResponseWriter, r *http.Request, _ httprouter.Params) {
	fmt.Fprint(w, fmt.Sprint(version))
}

func AddVersion(router *httprouter.Router) {
	router.GET(versionPath, DebugLogging(VersionRoute, versionPath))
}

func DebugLogging(h httprouter.Handle, path string) httprouter.Handle {
	return func(w http.ResponseWriter, r *http.Request, p httprouter.Params) {
		// log.Print("debug: ", path, " request body = ", r.Body)
		h(w, r, p)
		// log.Print("debug: ", path, " response=", w)
	}
}

func AddPredicate(router *httprouter.Router, predicate *scheduler.Predicate) {
	path := predicatesPrefix + "/" + predicate.Name
	log.Printf("addPredicate: %v", path)
	router.POST(predicatesPrefix, DebugLogging(PredicateRoute(predicate), predicatesPrefix))
}

func AddPrioritize(router *httprouter.Router, prioritize *scheduler.Prioritize) {
	path := prioritizePrefix + "/" + prioritize.Name
	log.Printf("addPrioritize: %v", path)
	router.POST(prioritizePrefix, DebugLogging(PrioritizeRoute(prioritize), prioritizePrefix))
	log.Print("end1")
}

func AddBind(router *httprouter.Router, bind *scheduler.Bind) {
	if handle, _, _ := router.Lookup("POST", bindPrefix); handle != nil {
		log.Print("warning: AddBind was called more then once!")
	} else {
		router.POST(bindPrefix, DebugLogging(BindRoute(bind), bindPrefix))
	}
}

func AddInspect(router *httprouter.Router, inspect *scheduler.Inspect) {
	router.GET(inspectPrefix, DebugLogging(InspectRoute(inspect), inspectPrefix))
	router.GET(inspectListPrefix, DebugLogging(InspectRoute(inspect), inspectListPrefix))
}
