/*
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
*/

package annoy_test

import (
	"math"
	"math/rand"
	"os"
	"testing"

	"github.com/spotify/annoy"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

type AnnoyTestSuite struct {
	suite.Suite
}

func Round(f float64) float64 {
	return math.Floor(f + 0.5)
}

func RoundPlus(f float64, places int) float64 {
	shift := math.Pow(10, float64(places))
	return Round(f*shift) / shift
}

func (suite *AnnoyTestSuite) SetupTest() {
}

func (suite *AnnoyTestSuite) TestFileHandling() {
	index := annoy.NewAnnoyIndexAngular(3)
	index.AddItem(0, []float32{0, 0, 1})
	index.AddItem(1, []float32{0, 1, 0})
	index.AddItem(2, []float32{1, 0, 0})
	index.Build(10)

	index.Save("go_test.ann")

	info, err := os.Stat("go_test.ann")
	if err != nil {
		assert.Fail(suite.T(), "Failed to create file, file not found")
	}
	if info.Size() == 0 {
		assert.Fail(suite.T(), "Failed to create file, file size zero")
	}

	annoy.DeleteAnnoyIndexAngular(index)

	index = annoy.NewAnnoyIndexAngular(3)
	if ret := index.Load("go_test.ann"); ret == false {
		assert.Fail(suite.T(), "Failed to load file")
	}

	os.Remove("go_test.ann")
	index.Save("go_test2.ann", false)

	info, err = os.Stat("go_test2.ann")
	if err != nil {
		assert.Fail(suite.T(), "Failed to create file without prefault, file not found")
	}
	if info.Size() == 0 {
		assert.Fail(suite.T(), "Failed to create file without prefault, file size zero")
	}

	annoy.DeleteAnnoyIndexAngular(index)

	index = annoy.NewAnnoyIndexAngular(3)
	if ret := index.Load("go_test2.ann", false); ret == false {
		assert.Fail(suite.T(), "Failed to load file without prefault")
	}

	os.Remove("go_test2.ann")
	index.Save("go_test3.ann", true)

	info, err = os.Stat("go_test3.ann")
	if err != nil {
		assert.Fail(suite.T(), "Failed to create file allowing prefault, file not found")
	}
	if info.Size() == 0 {
		assert.Fail(suite.T(), "Failed to create file allowing prefault, file size zero")
	}

	annoy.DeleteAnnoyIndexAngular(index)

	index = annoy.NewAnnoyIndexAngular(3)
	if ret := index.Load("go_test3.ann", true); ret == false {
		assert.Fail(suite.T(), "Failed to load file allowing prefault")
	}
	annoy.DeleteAnnoyIndexAngular(index)

	os.Remove("go_test3.ann")
}

func (suite *AnnoyTestSuite) TestOnDiskBuild() {
	index := annoy.NewAnnoyIndexAngular(3)
	index.OnDiskBuild("go_test.ann")

	info, err := os.Stat("go_test.ann")
	if err != nil {
		assert.Fail(suite.T(), "Failed to create file, file not found")
	}
	if info.Size() == 0 {
		assert.Fail(suite.T(), "Failed to create file, file size zero")
	}

	index.AddItem(0, []float32{0, 0, 1})
	index.AddItem(1, []float32{0, 1, 0})
	index.AddItem(2, []float32{1, 0, 0})
	index.Build(10)

	index.Unload()
	index.Load("go_test.ann")

	result := annoy.NewAnnoyVectorInt()
	defer result.Free()

	index.GetNnsByVector([]float32{3, 2, 1}, 3, -1, result)
	assert.Equal(suite.T(), []int32{2, 1, 0}, result.ToSlice())

	index.GetNnsByVector([]float32{1, 2, 3}, 3, -1, result)
	assert.Equal(suite.T(), []int32{0, 1, 2}, result.ToSlice())

	index.GetNnsByVector([]float32{2, 0, 1}, 3, -1, result)
	assert.Equal(suite.T(), []int32{2, 0, 1}, result.ToSlice())

	annoy.DeleteAnnoyIndexAngular(index)

	os.Remove("go_test.ann")
}

func (suite *AnnoyTestSuite) TestGetNnsByVector() {
	t := suite.T()
	index := annoy.NewAnnoyIndexAngular(3)
	index.AddItem(0, []float32{0, 0, 1})
	index.AddItem(1, []float32{0, 1, 0})
	index.AddItem(2, []float32{1, 0, 0})
	index.Build(10)

	t.Run("regular", func(t *testing.T) {
		result := annoy.NewAnnoyVectorInt()
		defer result.Free()

		index.GetNnsByVector([]float32{3, 2, 1}, 3, -1, result)
		assert.Equal(t, []int32{2, 1, 0}, result.ToSlice())

		index.GetNnsByVector([]float32{1, 2, 3}, 3, -1, result)
		assert.Equal(t, []int32{0, 1, 2}, result.ToSlice())

		index.GetNnsByVector([]float32{2, 0, 1}, 3, -1, result)
		assert.Equal(t, []int32{2, 0, 1}, result.ToSlice())
	})

	t.Run("with copying", func(t *testing.T) {
		result := annoy.NewAnnoyVectorInt()
		defer result.Free()

		var notAllocated []int32
		index.GetNnsByVector([]float32{3, 2, 1}, 3, -1, result)
		result.Copy(&notAllocated)
		assert.Equal(t, []int32{2, 1, 0}, notAllocated)

		// to make sure it will be overwritten
		var alreadyAllocated = make([]int32, 10)
		for i := 0; i < len(alreadyAllocated); i++ {
			alreadyAllocated[i] = -1
		}
		index.GetNnsByVector([]float32{3, 2, 1}, 3, -1, result)
		result.Copy(&alreadyAllocated)
		assert.Equal(t, []int32{2, 1, 0}, alreadyAllocated)

		var alreadyAllocatedCap = make([]int32, 0, 00)
		index.GetNnsByVector([]float32{3, 2, 1}, 3, -1, result)
		result.Copy(&alreadyAllocatedCap)
		assert.Equal(t, []int32{2, 1, 0}, alreadyAllocatedCap)
	})

	t.Run("with inner array", func(t *testing.T) {
		result := annoy.NewAnnoyVectorInt()
		defer result.Free()

		index.GetNnsByVector([]float32{3, 2, 1}, 3, -1, result)
		assert.Equal(t, []int32{2, 1, 0}, result.InnerArray())
	})

	annoy.DeleteAnnoyIndexAngular(index)
}

func (suite *AnnoyTestSuite) TestGetNnsByItem() {
	index := annoy.NewAnnoyIndexAngular(3)
	index.AddItem(0, []float32{2, 1, 0})
	index.AddItem(1, []float32{1, 2, 0})
	index.AddItem(2, []float32{0, 0, 1})
	index.Build(10)

	var result = annoy.NewAnnoyVectorInt()
	defer result.Free()

	index.GetNnsByItem(0, 3, -1, result)
	assert.Equal(suite.T(), []int32{0, 1, 2}, result.ToSlice())

	index.GetNnsByItem(1, 3, -1, result)
	assert.Equal(suite.T(), []int32{1, 0, 2}, result.ToSlice())

	annoy.DeleteAnnoyIndexAngular(index)
}

func (suite *AnnoyTestSuite) TestGetItem() {
	index := annoy.NewAnnoyIndexAngular(3)
	index.AddItem(0, []float32{2, 1, 0})
	index.AddItem(1, []float32{1, 2, 0})
	index.AddItem(2, []float32{0, 0, 1})
	index.Build(10)

	var result = annoy.NewAnnoyVectorFloat()
	defer result.Free()

	index.GetItem(0, result)
	assert.Equal(suite.T(), []float32{2, 1, 0}, result.ToSlice())

	index.GetItem(1, result)
	assert.Equal(suite.T(), []float32{1, 2, 0}, result.ToSlice())

	index.GetItem(2, result)
	assert.Equal(suite.T(), []float32{0, 0, 1}, result.ToSlice())

	annoy.DeleteAnnoyIndexAngular(index)
}

func (suite *AnnoyTestSuite) TestGetDistance() {
	index := annoy.NewAnnoyIndexAngular(2)
	index.AddItem(0, []float32{0, 1})
	index.AddItem(1, []float32{1, 1})
	index.Build(10)

	assert.Equal(suite.T(), RoundPlus(math.Pow(2*(1.0-math.Pow(2, -0.5)), 0.5), 3), RoundPlus(float64(index.GetDistance(0, 1)), 3))

	annoy.DeleteAnnoyIndexAngular(index)
}

func (suite *AnnoyTestSuite) TestGetDotProductDistance() {
	index := annoy.NewAnnoyIndexDotProduct(2)
	index.AddItem(0, []float32{0, 1})
	index.AddItem(1, []float32{1, 1})
	index.Build(10)

	assert.True(suite.T(),
		math.Abs(1.0-float64(index.GetDistance(0, 1))) < 0.00001)

	annoy.DeleteAnnoyIndexDotProduct(index)
}

func (suite *AnnoyTestSuite) TestLargeEuclideanIndex() {
	index := annoy.NewAnnoyIndexEuclidean(10)

	for j := 0; j < 10000; j += 2 {
		p := make([]float32, 0, 10)
		for i := 0; i < 10; i++ {
			p = append(p, rand.Float32())
		}
		x := make([]float32, 0, 10)
		for i := 0; i < 10; i++ {
			x = append(x, 1+p[i]+rand.Float32()*1e-2)
		}
		y := make([]float32, 0, 10)
		for i := 0; i < 10; i++ {
			y = append(y, 1+p[i]+rand.Float32()*1e-2)
		}
		index.AddItem(j, x)
		index.AddItem(j+1, y)
	}
	index.Build(10)
	result := annoy.NewAnnoyVectorInt()
	defer result.Free()
	for j := 0; j < 10000; j += 2 {
		index.GetNnsByItem(j, 2, -1, result)

		require.Equal(suite.T(), result.ToSlice(), []int32{int32(j), int32(j + 1)})

		index.GetNnsByItem(j+1, 2, -1, result)
		require.Equal(suite.T(), result.ToSlice(), []int32{int32(j) + 1, int32(j)})
	}
	annoy.DeleteAnnoyIndexEuclidean(index)
}

func TestAnnoyTestSuite(t *testing.T) {
	suite.Run(t, new(AnnoyTestSuite))
}
