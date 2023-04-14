package methods

import (
	"encoding/csv"
	"fmt"
	"gpushare-scheduler-extender/pkg/utils"
	"io"
	"log"
	"os"
	"strconv"
	"strings"

	"github.com/fxsjy/RF.go/RF/Regression"
	"github.com/go-gota/gota/dataframe"
	"github.com/go-gota/gota/series"
)

func OnlineLoadcsv(filepath string, setupReader func(*csv.Reader), nOutputs int) ([][]interface{}, []float64, error) {
	f, err := os.Open(os.Getenv("CSV_FOLDER") + "/online_interference.csv")
	Check(err)
	defer f.Close()
	r := csv.NewReader(f)
	if setupReader != nil {
		setupReader(r)
	}
	cells, err := r.ReadAll()
	length := float64(len(cells))
	cells = cells[1:int(length)]
	// log.Printf("cells are %+v", cells)
	Check(err)
	nSamples, nFeatures := len(cells), len(cells[0])-nOutputs-1
	X := make([][]interface{}, nSamples)
	Y := make([]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		var x_arr []interface{}
		for j := 0; j < nFeatures; j++ {
			x, err := strconv.ParseFloat(cells[i][j], 64)
			// log.Printf("x is %+v\n", x)
			Check(err)
			x_arr = append(x_arr, x)
		}
		X = append(X, x_arr)
		// log.Printf("i is %+v, x_arr is %+v\n", i, x_arr)
		y, err := strconv.ParseFloat(cells[i][nFeatures-1], 64)
		// print(y)
		Check(err)
		Y = append(Y, y)
	}
	// log.Printf("X is %+v\n", X)
	// log.Printf("Y is %+v\n", Y)
	return X, Y, err
}

func OnlineCreatePredRecord(jobs map[string]int) []interface{} {
	Y := make([]interface{}, 16)
	columns := []string{"vgg16", "squeezenet", "resnet50", "neumf", "adgcl", "lstm", "bert", "yolov5"}
	for i, column := range columns {
		if _, ok := jobs[column]; !ok {
			Y[i] = 0
			Y[i+len(columns)] = 00
		} else {
			Y[i] = 1
			value := strconv.FormatInt(int64(jobs[column]), 2)
			val, err := strconv.ParseFloat(value, 64)
			Check(err)
			Y[i+len(columns)] = val
		}
	}
	return Y
}

func OnlineGetInterferenceScoreFromCsv_(base_jobs []string, new_job string) []interface{} {
	log.Printf("OnlineGetInterferenceScoreFromCsv_")
	Y := make([]interface{}, 16)
	csvFile, err := os.Open(os.Getenv("CSV_FOLDER") + "/online_interference.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer csvFile.Close()

	csvDf := dataframe.ReadCSV(csvFile)

	jobs := CreateJobKV(base_jobs, new_job)
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
	if csvDf.Nrow() > 0 {
		log.Printf("the value one is %v", csvDf.Nrow())
		return Y
	} else {
		log.Printf("the value two is %v", csvDf.Nrow())
		log.Printf("jobs are: %+v", jobs)

		online_record := OnlineCreatePredRecord(jobs)
		log.Printf("OnlineCreatePredRecord is %v", online_record)
		// pred_interference, _ := OnlinePredByRFRegressor(OnlineCreatePredRecord(jobs))
		// return pred_interference, nil
		return online_record

	}
}

func OnlinePredByRFRegressor(X_test []interface{}) float64 {

	pred := 0.0

	X_train, Y_train, err := OnlineLoadcsv(os.Getenv("CSV_FOLDER")+"/online_interference.csv", nil, 1)
	Check(err)

	// foreset := Regression.BuildForest(X_train, Y_train, 100, len(X_train), len(X_train[0]))
	foreset := Regression.BuildForest(X_train, Y_train, len(X_train), len(X_train), len(X_train[0]))

	pred = foreset.Predicate(X_test)
	log.Printf("x_test is %+v, and pred is %+v", X_test, pred)

	return pred
}

func GetInterferenceFromFile(base_jobs []string, new_job string, total_records int) (float64, error) {

	csvFile, err := os.Open(os.Getenv("CSV_FOLDER") + "/online_interference.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer csvFile.Close()

	csvDf := dataframe.ReadCSV(csvFile)

	jobs := CreateJobKV(base_jobs, new_job)
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
	if csvDf.Nrow() > 0 {
		return csvDf.Elem(0, 16).Float(), nil

	} else {
		if total_records > utils.TotalRecords {
			pred_interference, _ := PredByRFRegressor(CreatePredRecord(jobs))
			return pred_interference, nil
			// pred_interference_str, _ := GetInterferenceFromRecord(CreatePredRecord(jobs), total_records)
			// pred_interference_float, _ := strconv.ParseFloat(pred_interference_str, 32)
			// return pred_interference_float, nil

		} else {
			return 0, nil
		}

	}
}

func GetInterferenceFromRecord(X_test []interface{}, total_records int) (string, error) {
	csvFile, err := os.Open(os.Getenv("CSV_FOLDER") + "/online_interference.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer csvFile.Close()

	if err != nil {
		log.Fatalln("failed to open file", err)
	}
	job_record := make([]string, 0)
	for _, v := range X_test {
		switch v_out := v.(type) {
		case int:
			job_record = append(job_record, strconv.Itoa(v_out))
		case float64:
			job_record = append(job_record, fmt.Sprintf("%v", v_out))
		}
	}
	r := csv.NewReader(csvFile)

	for i := 0; i < total_records; i++ {
		record, err := r.Read()
		isEqual := true
		for index, content := range job_record {
			if strings.Compare(record[index], content) != 0 {
				isEqual = false
			}
		}
		if isEqual {
			return record[16], nil

		}

		if err == io.EOF {
			break
		}

	}
	return "", nil
}
