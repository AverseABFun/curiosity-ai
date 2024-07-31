package mnist

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
)

const asciiChars = " |+/?lpx9$#"

type MNISTType int

const (
	MNIST_TYPE_LABEL    = 0x00000801
	MNIST_TYPE_IMAGE    = 0x00000803
	MNIST_TYPE_COMBINED = 0x00000805
)

type MNISTObject struct {
	File       *os.File
	Type       MNISTType
	NumEntries uint32
	Rows       uint32
	Columns    uint32
	LabelData  []byte
	ImageData  [][][]byte
	Invalid    bool
}

func (object MNISTObject) Combine(obj2 MNISTObject) MNISTObject {
	if object.Invalid || obj2.Invalid {
		panic("passed invalid objects to Combine")
	}
	if object.Type == obj2.Type {
		panic("tried to Combine two objects of the same kind")
	}
	if len(object.LabelData) != len(obj2.ImageData) {
		panic(fmt.Sprintf("not same number of entries in objects(%d in object1, %d in object2)", len(object.LabelData), len(obj2.ImageData)))
	}
	var newObject = object
	newObject.File = nil
	newObject.NumEntries = 0
	if newObject.Type == MNIST_TYPE_IMAGE {
		newObject.LabelData = obj2.LabelData
	} else {
		newObject.ImageData = obj2.ImageData
		newObject.Rows = obj2.Rows
		newObject.Columns = obj2.Columns
	}
	newObject.Type = MNIST_TYPE_COMBINED
	return newObject
}

func (object MNISTObject) String() string {
	var out = ""
	switch object.Type {
	case MNIST_TYPE_IMAGE:
		var switchedImageData = make([][][]byte, len(object.ImageData))
		for i, val := range object.ImageData {
			switchedImageData[i] = make([][]byte, object.Columns)
			for y, val2 := range val {
				switchedImageData[i][y] = make([]byte, object.Rows)
				copy(switchedImageData[i][y], val2)
			}
		}
		for i, val := range switchedImageData {
			out += fmt.Sprintf("Image %d:\n", i)
			for y, val2 := range val {
				out += fmt.Sprintf("Y %02d: ", y)
				for _, val3 := range val2 {
					out += calculateDensity(val3)
				}
				out += "\n"
			}
		}
	case MNIST_TYPE_COMBINED:
		var switchedImageData = make([][][]byte, len(object.ImageData))
		for i, val := range object.ImageData {
			switchedImageData[i] = make([][]byte, object.Columns)
			for y, val2 := range val {
				switchedImageData[i][y] = make([]byte, object.Rows)
				copy(switchedImageData[i][y], val2)
			}
		}
		for i, val := range switchedImageData {
			out += fmt.Sprintf("Image %d: Label %d\n", i, object.LabelData[i])
			for y, val2 := range val {
				out += fmt.Sprintf("Y %02d: ", y)
				for _, val3 := range val2 {
					out += calculateDensity(val3)
				}
				out += "\n"
			}
		}
	default:
		for i, num := range object.LabelData {
			out += fmt.Sprintf("Image %d: %d\n", i, num)
		}
	}
	return out
}

func calculateDensity(value byte) string {
	var val = float64(value)
	val = math.Round(val / 25.5)
	return string(asciiChars[int(val)])
}

type GetDataOptions struct {
	MaxIterations uint32
}

func GetData(file string, options GetDataOptions) (MNISTObject, error) {
	var openFile, err = os.Open(file)
	if err != nil {
		return MNISTObject{Invalid: true}, err
	}
	bytetype, err := ReadBytes(openFile, 4)
	if err != nil {
		return MNISTObject{Invalid: true}, err
	}

	var actualtype = binary.BigEndian.Uint32(bytetype)
	if actualtype != MNIST_TYPE_IMAGE && actualtype != MNIST_TYPE_LABEL {
		return MNISTObject{Invalid: true}, fmt.Errorf("file had neither type MNIST_TYPE_IMAGE nor type MNIST_TYPE_LABEL")
	}

	var object = MNISTObject{}
	object.File = openFile
	object.Type = MNISTType(actualtype)

	byteentries, err := ReadBytes(openFile, 4)
	if err != nil {
		return MNISTObject{Invalid: true}, err
	}

	object.NumEntries = binary.BigEndian.Uint32(byteentries)

	if object.Type == MNIST_TYPE_IMAGE {
		byterows, err := ReadBytes(openFile, 4)
		if err != nil {
			return MNISTObject{Invalid: true}, err
		}
		object.Rows = binary.BigEndian.Uint32(byterows)

		bytecolumns, err := ReadBytes(openFile, 4)
		if err != nil {
			return MNISTObject{Invalid: true}, err
		}
		object.Columns = binary.BigEndian.Uint32(bytecolumns)

		bytedata, err := ReadBytes(openFile, 1)
		var i = uint32(0)
		for err != io.EOF {
			if i > options.MaxIterations && options.MaxIterations > 0 {
				break
			}
			object.ImageData = append(object.ImageData, [][]byte{})
			for x := uint32(0); x < object.Rows; x++ {
				object.ImageData[len(object.ImageData)-1] = append(object.ImageData[len(object.ImageData)-1], []byte{})
				for y := uint32(0); y < object.Columns; y++ {
					object.ImageData[len(object.ImageData)-1][len(object.ImageData[len(object.ImageData)-1])-1] = append(object.ImageData[len(object.ImageData)-1][len(object.ImageData[len(object.ImageData)-1])-1], bytedata[0])
					bytedata, err = ReadBytes(openFile, 1)
				}
			}
			i++
		}
	} else {
		object.Rows = 0
		object.Columns = 0

		bytelabel, err := ReadBytes(openFile, 1)
		var i = uint32(0)
		for err != io.EOF {
			if i > options.MaxIterations && options.MaxIterations > 0 {
				break
			}
			object.LabelData = append(object.LabelData, bytelabel...)
			bytelabel, err = ReadBytes(openFile, 1)
			i++
		}
	}

	return object, nil
}

func ReadBytes(file io.Reader, numBytes uint) ([]byte, error) {
	var bytes = make([]byte, numBytes)
	var _, err = file.Read(bytes)
	if err != nil {
		return nil, err
	}

	return bytes, nil
}
