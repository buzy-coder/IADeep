package methods

import (
	"bufio"
	"encoding/csv"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"reflect"
	"sort"
	"strconv"
	"strings"

	matrix "github.com/skelterjohn/go.matrix"
)

func GetInterferenceScore(base_job []string, new_job string) float64 {

	// csvFile, err := os.Open("./interference.csv")
	csvFile, err := os.Open(os.Getenv("CSV_FOLDER") + "/interference.csv")
	if err != nil {
		log.Fatal(err)
	}
	reader := csv.NewReader(bufio.NewReader(csvFile))
	defer csvFile.Close()
	var interference [][]string
	for {
		line, err := reader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			log.Fatal(err)
		}
		interference = append(interference, line)
	}

	header := interference[0]

	interference_value := interference[1:]

	log.Printf("interference_value: %v", interference_value)

	log.Printf("header: %v", header)
	log.Printf("header_type: %v", reflect.TypeOf(header))
	log.Printf("values: %v", interference_value)
	r := matrix.Zeros(len(interference_value), len(interference_value))
	// log.Printf("old_r: %v", r)
	for i := 0; i < len(interference_value); i++ {
		for j := 0; j < len(interference_value[i]); j++ {
			v, err := strconv.ParseFloat(interference_value[i][j], 64)
			if err != nil {
				log.Fatal(err)
			}
			// log.Printf("eachr: %v, %v, %v", i, j, r)
			r.Set(i, j, v)
		}
	}
	// log.Printf("new_r: %v", r)

	n := r.Rows()
	m := r.Cols()
	k := 2

	// rand init
	p := matrix.Zeros(n, k)
	randInit(p)
	// log.Printf("debug: matrixp is %v", p)

	q := matrix.Zeros(m, k)
	randInit(q)
	// log.Printf("debug: matrixq is %v", q)

	nP, nQ := sgd(r, p, q, k, 0.0002, 0.02, 50)
	// nP, nQ := sgd(r, p, q, k, 0.0002, 0.02, 1)
	// log.Printf("nP is %v", nP)
	// log.Printf("nQ is %v", nQ)
	res := matrix.Product(nP, nQ.Transpose())

	log.Printf("base_job: %v", base_job)
	base_job_str := ""
	for i, _ := range base_job {
		base_job[i] = rmu0000(string(base_job[i]))
		base_job_str = base_job_str + "_" + base_job[i]
	}
	base_job_str = strings.Trim(base_job_str, "_")
	// log.Printf("base_jobs_type: %v", reflect.TypeOf(base_jobs))
	log.Printf("base_jobs: %v, %v", len(base_job), base_job)

	id_base, find_base := Find(header, base_job_str)
	log.Printf("id_base:%v", id_base)
	log.Printf("find_base:%v", find_base)

	id_new, find_new := Find(header, new_job)
	log.Printf("id_new:%v", id_new)
	log.Printf("find_new:%v", find_new)
	val := 0.0
	if find_new && find_base {
		log.Printf("debug: res is %+v", res)
		log.Printf("res.get %v", res.Get(id_new, id_base))
		if r.Get(id_new, id_base) > 0 {
			val = r.Get(id_new, id_base)
		} else {
			val = res.Get(id_new, id_base)
		}
		log.Printf("val: %v", val)
	}
	return val
}

// delete 0 when use json to string
func rmu0000(s string) string {
	str := make([]rune, 0, len(s))
	for _, v := range []rune(s) {
		if v == 0 {
			continue
		}
		str = append(str, v)
	}
	return string(str)
}

func dot(a matrix.Matrix, b matrix.Matrix) float64 {
	var r = 0.0
	for i := 0; i < a.Rows(); i++ {
		for j := 0; j < a.Cols(); j++ {
			r += a.Get(i, j) * b.Get(i, j)
		}
	}
	return r
}

func sgd(r matrix.Matrix, p *matrix.DenseMatrix, q *matrix.DenseMatrix, k int, alpha, beta float64, steps int) (*matrix.DenseMatrix, *matrix.DenseMatrix) {
	q = q.Transpose()
	log.Printf("debug: sgd q is %+v", q)
	for step := 0; step < steps; step++ {
		for i := 0; i < r.Rows(); i++ {
			for j := 0; j < r.Cols(); j++ {
				if r.Get(i, j) > 0 {
					eij := r.Get(i, j) - dot(p.GetRowVector(i), q.GetColVector(j).Transpose())
					// log.Printf("debug: eij is %v", eij)
					for z := 0; z < k; z++ {
						p.Set(i, z, p.Get(i, z)+alpha*(2*eij*q.Get(z, j)-beta*p.Get(i, z)))
						q.Set(z, j, q.Get(z, j)+alpha*(2*eij*p.Get(i, z)-beta*q.Get(z, j)))
					}
				}
			}
		}
		var e = 0.0
		for i := 0; i < r.Rows(); i++ {
			for j := 0; j < r.Cols(); j++ {
				if r.Get(i, j) > 0 {
					e += math.Pow(r.Get(i, j)-dot(p.GetRowVector(i), q.GetColVector(j).Transpose()), 2)
					for z := 0; z < k; z++ {
						e += (beta / 2) * (math.Pow(p.Get(i, z), 2) + math.Pow(q.Get(z, j), 2))
					}
				}
			}
		}
		if e < 0.001 {
			break
		}
	}
	log.Println("-------------------------------------")
	// log.Printf("debug1: p is %+v", p)
	// log.Printf("debug1: q is %+v", q.Transpose())
	return p, q.Transpose()
}

func randInit(m matrix.Matrix) {
	for i := 0; i < m.Rows(); i++ {
		for j := 0; j < m.Cols(); j++ {
			m.Set(i, j, rand.Float64())
		}
	}
}

func sortString(str string) string {
	split := strings.Split(str, "")
	sort.Strings(split)
	return strings.Join(split, "")
}

func Find(slice []string, val string) (int, bool) {
	log.Printf("header slice:%v", slice)
	log.Printf("find val:%v", val)
	val = sortString(val)
	for i, item := range slice {
		item = sortString(item)
		if item == val {
			return i, true
		}
	}
	return -1, false
}

func GetMinValue(arr []float64) (int, float64) {
	minVal := 0.0
	index := 0
	if len(arr) == 0 {
		return -1, minVal
	} else if len(arr) == 1 {
		return 0, arr[0]
	} else {
		for idx, item := range arr {
			if item <= minVal {
				index = idx
				minVal = item
			}
		}
	}
	return index, minVal
}
