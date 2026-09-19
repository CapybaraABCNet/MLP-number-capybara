package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
)

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func sigmoidDeriv(x float64) float64 {
	return x * (1 - x)
}

func saveWeights(wHidden, bHidden, wOutput, bOutput []float64) {
	file, err := os.Create("weights.txt")
	if err != nil {
		fmt.Println("Ошибка: ", err)
		return
	}

	defer file.Close()

	for _, w := range wHidden {
		fmt.Fprintf(file, "%f\n", w)
	}

	for _, b := range bHidden {
		fmt.Fprintf(file, "%f\n", b)
	}

	for _, wO := range wOutput {
		fmt.Fprintf(file, "%f\n", wO)
	}

	for _, bO := range bOutput {
		fmt.Fprintf(file, "%f\n", bO)
	}

	fmt.Println("✅ Веса сохранены")
}

func loadWeights(wHidden, bHidden, wOutput, bOutput []float64) bool {
	file, err := os.Open("weights.txt")
	if err != nil {
		return false
	}

	defer file.Close()

	for w := range wHidden {
		fmt.Fscanf(file, "%f\n", &wHidden[w])
	}

	for i := range bHidden {
		fmt.Fscanf(file, "%f\n", &bHidden[i])
	}

	for j := range wOutput {
		fmt.Fscanf(file, "%f\n", &wOutput[j])
	}

	for b := range bOutput {
		fmt.Fscanf(file, "%f\n", &bOutput[b])
	}

	fmt.Println("✅ Веса загружены")
	return true
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

	yTop := make([]float64, 10)

	digits := [][][]int{
		{ // 0
			{1, 1, 1, 1, 1},
			{1, 0, 0, 0, 1},
			{1, 0, 0, 0, 1},
			{1, 0, 0, 0, 1},
			{1, 1, 1, 1, 1},
		},
		{ // 1
			{0, 0, 1, 0, 0},
			{0, 1, 1, 0, 0},
			{0, 0, 1, 0, 0},
			{0, 0, 1, 0, 0},
			{0, 1, 1, 1, 0},
		},
		{ // 2
			{1, 1, 1, 0, 0},
			{0, 0, 0, 1, 0},
			{0, 0, 1, 0, 0},
			{0, 1, 0, 0, 0},
			{1, 1, 1, 1, 1},
		},
		{ // 3
			{1, 1, 1, 1, 0},
			{0, 0, 0, 0, 1},
			{0, 1, 1, 1, 0},
			{0, 0, 0, 0, 1},
			{1, 1, 1, 1, 0},
		},
		{ // 4
			{0, 0, 0, 1, 0},
			{0, 0, 1, 1, 0},
			{0, 1, 0, 1, 0},
			{1, 1, 1, 1, 1},
			{0, 0, 0, 1, 0},
		},
		{ // 5
			{1, 1, 1, 1, 1},
			{1, 0, 0, 0, 0},
			{1, 1, 1, 1, 0},
			{0, 0, 0, 0, 1},
			{1, 1, 1, 1, 0},
		},
		{ // 6
			{0, 0, 1, 1, 0},
			{0, 1, 0, 0, 0},
			{1, 1, 1, 1, 0},
			{1, 0, 0, 0, 1},
			{0, 1, 1, 1, 0},
		},
		{ // 7
			{1, 1, 1, 1, 1},
			{0, 0, 0, 0, 1},
			{0, 0, 0, 1, 0},
			{0, 0, 1, 0, 0},
			{0, 1, 0, 0, 0},
		},
		{ // 8
			{0, 1, 1, 1, 0},
			{1, 0, 0, 0, 1},
			{0, 1, 1, 1, 0},
			{1, 0, 0, 0, 1},
			{0, 1, 1, 1, 0},
		},
		{ // 9
			{0, 1, 1, 1, 0},
			{1, 0, 0, 0, 1},
			{0, 1, 1, 1, 1},
			{0, 0, 0, 1, 0},
			{0, 1, 1, 0, 0},
		},
	}

	if !loadWeights(wHidden, bHidden, wOutput, bOutput) {
		fmt.Println("🔄 Обучение...")

		for epoch := 1; epoch < 20000; epoch++ {
			for d := 0; d < 10; d++ {
				xTop := make([]float64, 0, 25)
				for i := 0; i < 5; i++ {
					for j := 0; j < 5; j++ {
						xTop = append(xTop, float64(digits[d][i][j]))
					}
				}

				target := make([]float64, 10)
				target[d] = 1

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
		}

		saveWeights(wHidden, bHidden, wOutput, bOutput)
	}

	fmt.Println("\n🧪 Проверка:")
	for d := 0; d < 10; d++ {
		xTop := make([]float64, 0, 25)
		for i := 0; i < 5; i++ {
			for j := 0; j < 5; j++ {
				xTop = append(xTop, float64(digits[d][i][j]))
			}
		}

		sumH := make([]float64, 10)
		for h := 0; h < 10; h++ {
			for i := 0; i < 25; i++ {
				sumH[h] += xTop[i] * wHidden[h*25+i]
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

		yTop := make([]float64, 10)
		for i3 := 0; i3 < 10; i3++ {
			yTop[i3] = sigmoid(sumY[i3])
		}

		maxIdx := 0
		maxVal := yTop[0]
		for i := 1; i < 10; i++ {
			if yTop[i] > maxVal {
				maxVal = yTop[i]
				maxIdx = i
			}
		}

		fmt.Printf("Цифра %d → ответ: %d (%.2f%%)\n", d, maxIdx, maxVal*100)
	}
}
