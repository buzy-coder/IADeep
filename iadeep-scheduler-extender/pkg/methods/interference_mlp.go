package methods

import (
	"encoding/csv"
	"log"
	"os"
	"strconv"

	"github.com/go-gota/gota/dataframe"
	"github.com/go-gota/gota/series"
	"gonum.org/v1/gonum/mat"

	neuralnetwork "github.com/pa-m/sklearn/neural_network"
)

func Check(err error) {
	if err != nil {
		panic(err)
	}
}

func GetAdditionalInterferenceScoreFromCsv(base_jobs []string, new_job string) (float64, error) {

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

func CreateJobKV(base_jobs []string, new_job string) map[string]int {
	jobs := make(map[string]int)
	base_jobs = append(base_jobs, new_job)
	for _, base_job := range base_jobs {
		if _, ok := jobs[base_job]; ok {
			jobs[base_job] = jobs[base_job] + 1
		} else {
			jobs[base_job] = 1
		}
	}
	return jobs
}

func createPredData(jobs map[string]int) *mat.Dense {
	Y := mat.NewDense(1, 16, nil)
	columns := []string{"vgg16", "vgg19", "squeezenet", "googlenet", "alexnet", "resnet152", "neumf", "adgcl"}
	for i, column := range columns {
		if _, ok := jobs[column]; !ok {
			Y.Set(0, i, 0)
			Y.Set(0, i+len(columns), 0)
		} else {
			Y.Set(0, i, 1)
			value := strconv.FormatInt(int64(jobs[column]), 2)
			val, _ := strconv.ParseFloat(value, 64)
			Y.Set(0, i+len(columns), val)
		}
	}
	return Y
}

func LoadCSV(filepath string, setupReader func(*csv.Reader), nOutputs int) (X, Y *mat.Dense) {
	f, err := os.Open(filepath)
	Check(err)
	defer f.Close()
	r := csv.NewReader(f)
	if setupReader != nil {
		setupReader(r)
	}
	cells, err := r.ReadAll()
	length := len(cells)
	cells = cells[1:length]
	// log.Printf("cells are %+v", cells)
	Check(err)
	nSamples, nFeatures := len(cells), len(cells[0])-nOutputs
	X = mat.NewDense(nSamples, nFeatures, nil)
	X.Apply(func(i, j int, _ float64) float64 { x, err := strconv.ParseFloat(cells[i][j], 64); Check(err); return x }, X)
	Y = mat.NewDense(nSamples, nOutputs, nil)
	Y.Apply(func(i, o int, _ float64) float64 {
		y, err := strconv.ParseFloat(cells[i][nFeatures], 64)
		Check(err)
		return y
	}, Y)
	// log.Printf("X is %+v", X)
	// log.Printf("Y is %+v", Y)
	return X, Y
}

func PredByMLPRegressor(X_test *mat.Dense) (pred float64, err error) {

	X_train, Y_train := LoadCSV(os.Getenv("CSV_FOLDER")+"/interference.csv", nil, 1)
	// Y = ReadCSV(root_path + "/dataset/interference_pre.csv")
	r, c := X_train.Dims()
	log.Printf("X type is %v %v", r, c)
	r, c = Y_train.Dims()
	log.Printf("Y type is %v %v", r, c)

	mlp := neuralnetwork.NewMLPRegressor([]int{5, 5, 5}, "relu", "adam", 0)
	mlp = mlp.PredicterClone().(*neuralnetwork.MLPRegressor) // for coverage
	mlp.IsClassifier()                                       // for coverage
	// mlp.RandomState = base.NewLockedSource(1)
	mlp.LearningRateInit = .001
	mlp.WeightDecay = .01
	mlp.Shuffle = true
	mlp.BatchSize = 10
	mlp.MaxIter = 500
	mlp.Fit(X_train, Y_train)
	y_pred := mlp.Predict(X_test, nil)
	log.Printf("y_pred is: %v", y_pred)
	return y_pred.At(0, 0), nil
}
