package main

import (
	"github.com/averseabfun/curiosity-ai/mnist"
	"github.com/averseabfun/logger"
)

func main() {
	logger.Log(logger.LogInfo, "Curiosity AI v0.1.0")
	var data1, err = mnist.GetData("mnist/train-labels-idx1-ubyte", mnist.GetDataOptions{MaxIterations: 5})
	data2, err := mnist.GetData("mnist/train-images-idx3-ubyte", mnist.GetDataOptions{MaxIterations: 5})
	var data = data1.Combine(data2)
	logger.Logf(logger.LogDebug, "err: %w, data:\n%s", err, data)
}
