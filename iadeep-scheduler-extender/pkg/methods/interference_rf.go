package methods

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"

	"github.com/fxsjy/RF.go/RF/Regression"
	"github.com/go-gota/gota/dataframe"
	"github.com/go-gota/gota/series"
)

func Loadcsv(filepath string, setupReader func(*csv.Reader), nOutputs int) ([][]interface{}, []float64, error) {
	f, err := os.Open(filepath)
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

func CreatePredRecord(jobs map[string]int) []interface{} {
	Y := make([]interface{}, 16) //interface 接口
	columns := []string{"vgg16", "vgg19", "squeezenet", "googlenet", "alexnet", "resnet152", "neumf", "adgcl"}
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

func GetInterferenceScoreFromCsv_(base_jobs []string, new_job string) (float64, error) {

	csvFile, err := os.Open(os.Getenv("CSV_FOLDER") + "/interference.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer csvFile.Close()

	csvDf := dataframe.ReadCSV(csvFile)

	jobs := CreateJobKV(base_jobs, new_job)
	for k, v := range jobs {
		num := strconv.FormatInt(int64(v), 2)
		fil := csvDf.Filter(
			dataframe.F{Colname: k, Comparator: series.Eq, Comparando: v},
			dataframe.F{Colname: k + "_num", Comparator: series.Eq, Comparando: num},
		)
		csvDf = fil
	}
	if csvDf.Nrow() > 0 {
		return csvDf.Elem(0, 16).Float(), nil
	} else {
		pred_interference, _ := PredByRFRegressor(CreatePredRecord(jobs))
		return pred_interference, nil
	}
}

func PredByRFRegressor(X_test []interface{}) (float64, error) {

	pred := 0.0

	X_train, Y_train, err := Loadcsv(os.Getenv("CSV_FOLDER")+"/interference.csv", nil, 1)
	Check(err)

	// foreset := Regression.BuildForest(X_train, Y_train, 100, len(X_train), len(X_train[0]))
	foreset := Regression.BuildForest(X_train, Y_train, len(X_train), len(X_train), len(X_train[0]))

	pred = foreset.Predicate(X_test)
	log.Printf("x_test is %+v, and pred is %+v", X_test, pred)

	return pred, nil
}

func ReadRecord() int {
	csvFile, err := os.Open(os.Getenv("CSV_FOLDER") + "/online_interference.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer csvFile.Close()

	filedata, err := csv.NewReader(csvFile).ReadAll()

	if err != nil {
		log.Fatal(err)
	}
	defer csvFile.Close()
	total_jobs := len(filedata)
	log.Printf("total_jobs are : %+v", total_jobs)
	return total_jobs
}

func WriteRecord(var1 []interface{}, interference float64) {
	csvFile, err := os.OpenFile(os.Getenv("CSV_FOLDER")+"/online_interference.csv", os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0644)
	if err != nil {
		log.Fatal(err)
	}
	defer csvFile.Close()

	if err != nil {
		log.Fatalln("failed to open file", err)
	}
	interference_record := fmt.Sprintf("%v", interference)
	log.Printf("interference_record is : %+v", interference_record)

	job_record := make([]string, 0)

	for _, v := range var1 {
		switch v_out := v.(type) {
		case int:
			job_record = append(job_record, strconv.Itoa(v_out))
		case float64:
			job_record = append(job_record, fmt.Sprintf("%v", v_out))
		}
	}
	log.Printf("job_record is : %+v", job_record)

	whole_record := append(job_record, interference_record)
	log.Printf("whole_record is : %+v", whole_record)

	var records [][]string

	records = append(records, whole_record)
	log.Printf("records is : %+v", records)
	w := csv.NewWriter(csvFile)
	w.WriteAll(records)

	if err != nil {
		log.Fatal(err)
	}

}

// get interference from util csv

func GetInterferenceScoreFromUtilCsv(base_jobs []string, new_job string) (float64, error) {

	csvFile, err := os.Open("/csv/interference_util.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer csvFile.Close()

	csvDf := dataframe.ReadCSV(csvFile)

	jobs := CreateJobKV(base_jobs, new_job)
	for k, v := range jobs {
		num := strconv.FormatInt(int64(v), 2)
		fil := csvDf.Filter(
			dataframe.F{Colname: k, Comparator: series.Eq, Comparando: v},
			dataframe.F{Colname: k + "_num", Comparator: series.Eq, Comparando: num},
		)
		csvDf = fil
	}
	if csvDf.Nrow() > 0 {
		return csvDf.Elem(0, 16).Float(), nil
	} else {
		pred_interference, _ := PredByRFRegressor(CreatePredUtilRecord(jobs))
		return pred_interference, nil
	}
}

func CreatePredUtilRecord(jobs map[string]int) []interface{} {
	Y := make([]interface{}, 16)
	columns := []string{"vgg16", "vgg19", "squeezenet", "googlenet", "alexnet", "lstm", "neumf", "adgcl"}
	// for i, column := range columns {
	// 	if _, ok := jobs[column]; !ok {
	// 		Y[i] = 0
	// 		Y[i+1] = 0
	// 	} else {
	// 		Y[i] = jobs_util[i].SM_util
	// 		Y[i+1] = jobs_util[i].Mem_util
	// 	}
	// }
	for i, column := range columns {
		if _, ok := jobs[column]; !ok {
			Y[2*i] = 0
			Y[2*i+1] = 0
		} else {
			if column == "vgg16" {
				Y[2*i] = 53.44736842105264
				Y[2*i+1] = 26.730263157894736
			}
			if column == "vgg19" {
				Y[2*i] = 60.84324324324324
				Y[2*i+1] = 34.69189189189189
			}
			if column == "squeezenet" {
				Y[2*i] = 10.24271844660194
				Y[2*i+1] = 2.9805825242718447
			}
			if column == "googlenet" {
				Y[2*i] = 24.29545454545455
				Y[2*i+1] = 8.227272727272727
			}
			if column == "alexnet" {
				Y[2*i] = 7.531732418524871
				Y[2*i+1] = 2.497427101200686
			}
			if column == "lstm" {
				Y[2*i] = 57.08552631578947
				Y[2*i+1] = 28.697368421052627
			}
			if column == "neumf" {
				Y[2*i] = 3.425287356321839
				Y[2*i+1] = 0.5057471264367817
			}
			if column == "adgcl" {
				Y[2*i] = 10.933070866141732
				Y[2*i+1] = 3.8385826771653533
			}
		}
	}
	log.Printf("Y is %+v", Y)
	return Y
}
