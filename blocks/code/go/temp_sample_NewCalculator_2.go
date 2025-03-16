// Original file: temp_sample.go
// Block type: function
// Name: NewCalculator

func NewCalculator(initialValue float64) *Calculator {
    return &Calculator{
        value: initialValue,
    }
}