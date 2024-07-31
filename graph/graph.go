package graph

import (
	"fmt"
	"slices"
	"strconv"
	"strings"
)

type SimpleData struct {
	data map[string]any
}

func (data SimpleData) GetData(str string) (bool, any) {
	var out, ok = data.data[str]
	return ok, out
}

func (data SimpleData) GetAllData() map[string]any {
	return data.data
}

func (data *SimpleData) SetData(str string, dta any) {
	if data.data == nil {
		data.data = make(map[string]any)
	}
	data.data[str] = dta
}

func CreateSimpleData() SimpleData {
	return SimpleData{data: make(map[string]any)}
}

type ActivationFunc struct {
	Call func(float64) float64
}

type NNData struct {
	Activation_func  ActivationFunc
	Output_value     float64
	Input_value      float64
	Bias             float64
	Layer_idx        uint64
	Node_idx         uint64
	Global_cost_func func(float64) float64
	Gradient         float64
	Parent_weights   map[uint64]float64
}

func (data NNData) GetAllData() string {
	return fmt.Sprintf("%#v", data)
}

func CreateNNData(activation_func ActivationFunc) NNData {
	return NNData{Activation_func: activation_func}
}

func (node *Node) AddChild(child *Node, weight float64) {
	node.Children = append(node.Children, child)
	child.Parents = append(child.Parents, node)
	child.Data.Parent_weights[node.Id] = weight
	child.Parent_map[node.Id] = node
}

type Node struct {
	Children   []*Node
	Parents    []*Node
	Parent_map map[uint64]*Node
	Id         uint64
	Data       NNData
}

func (node Node) CalculateOutput() {
	node.calculateInput()
	node.Data.Output_value = node.Data.Activation_func.Call(node.Data.Input_value)
}

func (node Node) calculateInput() {
	for key, val := range node.Data.Parent_weights {
		node.Parent_map[key].CalculateOutput()
		var output = node.Parent_map[key].Data.Output_value * val
		node.Data.Input_value += output
	}
	node.Data.Input_value += node.Data.Bias
}

func (node Node) String() string {
	var children = "["
	for _, val := range node.Children {
		children += strconv.Itoa(int(val.Id)) + ", "
	}
	children = strings.TrimSuffix(children, ", ")
	children += "]"

	var data = node.Data.GetAllData()
	var dataOut = "[" + data + "]"
	return fmt.Sprintf("\nNode %d: Children: %s, Data: %s", node.Id, children, dataOut)
}

var seen = []uint64{}

func (node Node) RecursiveString() string {
	var str = ""
	if !slices.Contains(seen, node.Id) {
		str = node.String()
	}
	for _, val := range node.Children {
		if slices.Contains(seen, val.Id) {
			continue
		}
		var newStr = val.RecursiveString()
		seen = append(seen, val.Id)
		str += newStr
	}
	return str
}

type AINetwork struct {
	InputLayer   []*Node
	HiddenLayers [][]*Node
	OutputLayer  []*Node
	maxID        uint64
}

func (network *AINetwork) CreateNode(activation_func ActivationFunc) *Node {
	return &Node{Id: network.GetID(), Data: CreateNNData(activation_func)}
}

func (network *AINetwork) GetID() uint64 {
	network.maxID++
	return network.maxID
}

func (network AINetwork) String() string {
	var str = "["
	for _, val := range network.InputLayer {
		str += val.String()
	}
	str += "]"
	return str
}

func (network AINetwork) RecursiveString() string {
	seen = []uint64{}
	var str = "["
	for _, val := range network.InputLayer {
		var newStr = val.RecursiveString()
		str += newStr
	}
	str += "]"
	return str
}

func CreateNetwork(inputLayerNum uint32, outputLayerNum uint32, hiddenLayerX uint32, hiddenLayerY uint32, defaultWeight float64, activation_func ActivationFunc) *AINetwork {
	var newNetwork = &AINetwork{maxID: 0}
	for i := 0; i < int(inputLayerNum); i++ {
		newNetwork.InputLayer = append(newNetwork.InputLayer, newNetwork.CreateNode(activation_func))
	}
	for i := 0; i < int(outputLayerNum); i++ {
		newNetwork.OutputLayer = append(newNetwork.OutputLayer, newNetwork.CreateNode(activation_func))
	}

	var addTo = newNetwork.InputLayer
	for x := 0; x < int(hiddenLayerX); x++ {
		newNetwork.HiddenLayers = append(newNetwork.HiddenLayers, []*Node{})
		var newLayer = newNetwork.HiddenLayers[len(newNetwork.HiddenLayers)-1]
		var tempLayer []*Node
		for y := 0; y < int(hiddenLayerY); y++ {
			var newNode = newNetwork.CreateNode(activation_func)
			newLayer = append(newLayer, newNode)
			for _, val := range addTo {
				val.AddChild(newNode, defaultWeight)
			}
			if x == int(hiddenLayerX)-1 {
				for _, val := range newNetwork.OutputLayer {
					newNode.AddChild(val, defaultWeight)
				}
			}
			tempLayer = append(tempLayer, newNode)
		}
		addTo = tempLayer
	}
	return newNetwork
}
