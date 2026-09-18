package main

import (
	"fmt"
	"math"
	"math/rand"
)

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func sigmoidDeriv(x float64) float64 {
	return x * (1 - x)
}

func main() {
	wHidden := make([]float64, 250)
	for i := range wHidden {
		wHidden[i] = rand.Float64()*2 - 1
	}

	bHidden := make([]float64, 10)
	for j := range bHidden {
		bHidden[j] = rand.Float64()*2 - 1
	}

	wOutput := make([]float64, 100)
	for i := range wOutput {
		wOutput[i] = rand.Float64()*2 - 1
	}
	bOutput := make([]float64, 10)
	for j := range bOutput {
		bOutput[j] = rand.Float64()*2 - 1
	}

	lr := 4.5

	one := [][]int{
		{0, 1, 1, 1, 0},
		{0, 1, 0, 1, 0},
		{1, 0, 0, 1, 0},
		{0, 1, 0, 1, 0},
		{0, 1, 1, 1, 0},
	}

	/*one := [][]int{
		{0, 1, 1, 1, 0},
		{0, 1, 0, 1, 0},
		{0, 1, 0, 1, 0},
		{0, 1, 0, 1, 0},
		{0, 1, 1, 1, 0},
	}*/

	/*one := [][]int{
		{0, 0, 1, 0, 0},
		{0, 1, 1, 0, 0},
		{0, 0, 1, 0, 0},
		{0, 0, 1, 0, 0},
		{0, 1, 1, 1, 0},
	}*/

	target := make([]float64, 10)
	target[0] = 1

	yTop := make([]float64, 10)

	for epoch := 1; epoch < 20000; epoch++ {
		xTop := make([]float64, 0, 25)
		for i := 0; i < 5; i++ {
			for j := 0; j < 5; j++ {
				xTop = append(xTop, float64(one[i][j]))
			}
		}

		sumH := make([]float64, 10)
		for h := 0; h < 10; h++ {
			for i := 0; i < 25; i++ {
				sumH[h] += float64(xTop[i]) * wHidden[h*25+i]
			}
			sumH[h] += bHidden[h]
		}

		hTop := make([]float64, 10)
		for j := 0; j < 10; j++ {
			hTop[j] = sigmoid(sumH[j])
		}

		sumY := make([]float64, 10)
		for i2 := 0; i2 < 10; i2++ {
			for i1 := 0; i1 < 10; i1++ {
				sumY[i2] += hTop[i1] * wOutput[i2*10+i1]
			}
			sumY[i2] += bOutput[i2]
		}

		for i3 := 0; i3 < 10; i3++ {
			yTop[i3] = sigmoid(sumY[i3])
		}

		errY := make([]float64, 10)
		for i4 := 0; i4 < 10; i4++ {
			errY[i4] = target[i4] - yTop[i4]
		}

		deltaY := make([]float64, 10)
		for i5 := 0; i5 < 10; i5++ {
			deltaY[i5] = errY[i5] * sigmoidDeriv(yTop[i5])
		}

		errH := make([]float64, 10)
		for h := 0; h < 10; h++ {
			errH[h] = 0.0
			for out := 0; out < 10; out++ {
				errH[h] += deltaY[out] * wOutput[out*10+h]
			}
		}

		deltaH := make([]float64, 10)
		for h := 0; h < 10; h++ {
			deltaH[h] = errH[h] * sigmoidDeriv(hTop[h])
		}

		for h := 0; h < 10; h++ {
			for i := 0; i < 25; i++ {
				wHidden[h*25+i] += lr * deltaH[h] * xTop[i]
			}
			bHidden[h] += lr * deltaH[h]
		}

		for out := 0; out < 10; out++ {
			for h := 0; h < 10; h++ {
				wOutput[out*10+h] += lr * deltaY[out] * hTop[h]
			}
			bOutput[out] += lr * deltaY[out]
		}
	}

	maxIdx := 0
	maxVal := yTop[0]
	for i := 1; i < 10; i++ {
		if yTop[i] > maxVal {
			maxVal = yTop[i]
			maxIdx = i
		}
	}
	fmt.Printf("Ответ: %d (уверенность: %.2f%%)\n", maxIdx, maxVal*100)

	fmt.Printf("Ожидалось: ")
	for i, v := range target {
		if v == 1 {
			fmt.Printf("%d\n", i)
		}
	}
}
