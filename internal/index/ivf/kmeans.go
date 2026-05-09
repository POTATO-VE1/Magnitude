package ivf

import (
	"math"
	"math/rand"
	"runtime"
	"sync"

	"github.com/POTATO-VE1/Magnitude/internal/distance"
)

// KMeans implements K-Means++ clustering with parallel assignment.
// Used by IVFIndex to partition vectors into Voronoi cells.
type KMeans struct {
	K         int         // number of clusters
	Dim       int         // vector dimension
	MaxIter   int         // maximum iterations (default: 200)
	Centroids []float32   // shape [K * Dim], row-major
	distFn    distance.DistanceFunc
	rng       *rand.Rand
}

// NewKMeans creates a new K-Means engine.
func NewKMeans(k, dim int, distFn distance.DistanceFunc, seed int64) *KMeans {
	return &KMeans{
		K:       k,
		Dim:     dim,
		MaxIter: 200,
		distFn:  distFn,
		rng:     rand.New(rand.NewSource(seed)),
	}
}

// Fit runs K-Means++ on the given vectors and returns cluster assignments.
// vectors is a flat [n * dim] row-major slice. n is the vector count.
// Returns assignments where assignments[i] is the cluster index for vector i.
func (km *KMeans) Fit(vectors []float32, n int) []int {
	if n <= km.K {
		// Degenerate case: fewer vectors than clusters
		km.Centroids = make([]float32, n*km.Dim)
		copy(km.Centroids, vectors[:n*km.Dim])
		assignments := make([]int, n)
		for i := range assignments {
			assignments[i] = i
		}
		return assignments
	}

	// K-Means++ initialization
	km.initPlusPlus(vectors, n)

	assignments := make([]int, n)
	prevChanges := n // force at least one iteration

	for iter := 0; iter < km.MaxIter; iter++ {
		// Assignment step (parallel)
		changes := km.assignParallel(vectors, n, assignments)

		// Convergence check: < 0.1% of vectors changed assignment
		if float64(changes)/float64(n) < 0.001 && iter > 0 {
			break
		}
		if changes == 0 && prevChanges == 0 {
			break
		}
		prevChanges = changes

		// Update step: recompute centroids
		km.updateCentroids(vectors, n, assignments)
	}

	return assignments
}

// initPlusPlus implements K-Means++ initialization (D² sampling).
// Produces well-spaced initial centroids that dramatically reduce final variance.
func (km *KMeans) initPlusPlus(vectors []float32, n int) {
	km.Centroids = make([]float32, km.K*km.Dim)

	// Pick the first centroid uniformly at random
	first := km.rng.Intn(n)
	copy(km.Centroids[0:km.Dim], vectors[first*km.Dim:(first+1)*km.Dim])

	// D² sampling for remaining centroids
	minDists := make([]float32, n) // min distance from each vector to nearest centroid
	for i := range minDists {
		minDists[i] = math.MaxFloat32
	}

	for c := 1; c < km.K; c++ {
		// Update min distances to include the newly added centroid (c-1)
		prevCentroid := km.Centroids[(c-1)*km.Dim : c*km.Dim]
		var totalWeight float64
		for i := 0; i < n; i++ {
			v := vectors[i*km.Dim : (i+1)*km.Dim]
			d := km.distFn(v, prevCentroid)
			if d < minDists[i] {
				minDists[i] = d
			}
			totalWeight += float64(minDists[i])
		}

		// Sample proportional to D²
		target := km.rng.Float64() * totalWeight
		var cumWeight float64
		chosen := 0
		for i := 0; i < n; i++ {
			cumWeight += float64(minDists[i])
			if cumWeight >= target {
				chosen = i
				break
			}
		}
		copy(km.Centroids[c*km.Dim:(c+1)*km.Dim], vectors[chosen*km.Dim:(chosen+1)*km.Dim])
	}
}

// assignParallel assigns each vector to its nearest centroid using goroutines.
// Returns the number of assignment changes (for convergence detection).
func (km *KMeans) assignParallel(vectors []float32, n int, assignments []int) int {
	numWorkers := runtime.NumCPU()
	if numWorkers > n {
		numWorkers = n
	}
	chunkSize := (n + numWorkers - 1) / numWorkers

	changes := make([]int, numWorkers)
	var wg sync.WaitGroup

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(workerID, start int) {
			defer wg.Done()
			end := start + chunkSize
			if end > n {
				end = n
			}
			localChanges := 0
			for i := start; i < end; i++ {
				v := vectors[i*km.Dim : (i+1)*km.Dim]
				nearest := km.nearestCentroid(v)
				if nearest != assignments[i] {
					assignments[i] = nearest
					localChanges++
				}
			}
			changes[workerID] = localChanges
		}(w, w*chunkSize)
	}
	wg.Wait()

	total := 0
	for _, c := range changes {
		total += c
	}
	return total
}

// nearestCentroid returns the index of the centroid closest to v.
func (km *KMeans) nearestCentroid(v []float32) int {
	best := 0
	bestDist := float32(math.MaxFloat32)
	for c := 0; c < km.K; c++ {
		centroid := km.Centroids[c*km.Dim : (c+1)*km.Dim]
		d := km.distFn(v, centroid)
		if d < bestDist {
			bestDist = d
			best = c
		}
	}
	return best
}

// updateCentroids recomputes centroid positions as the mean of assigned vectors.
// Empty clusters are replaced with the vector farthest from its current centroid.
func (km *KMeans) updateCentroids(vectors []float32, n int, assignments []int) {
	newCentroids := make([]float32, km.K*km.Dim)
	counts := make([]int, km.K)

	// Accumulate sums
	for i := 0; i < n; i++ {
		c := assignments[i]
		counts[c]++
		src := vectors[i*km.Dim : (i+1)*km.Dim]
		dst := newCentroids[c*km.Dim : (c+1)*km.Dim]
		for j := range src {
			dst[j] += src[j]
		}
	}

	// Divide by count to get mean
	for c := 0; c < km.K; c++ {
		if counts[c] > 0 {
			centroid := newCentroids[c*km.Dim : (c+1)*km.Dim]
			invCount := 1.0 / float32(counts[c])
			for j := range centroid {
				centroid[j] *= invCount
			}
		}
	}

	// Handle empty clusters: replace with the farthest vector from its centroid
	for c := 0; c < km.K; c++ {
		if counts[c] == 0 {
			farthestIdx := km.findFarthestVector(vectors, n, assignments, newCentroids)
			copy(newCentroids[c*km.Dim:(c+1)*km.Dim], vectors[farthestIdx*km.Dim:(farthestIdx+1)*km.Dim])
			assignments[farthestIdx] = c
		}
	}

	km.Centroids = newCentroids
}

// findFarthestVector finds the vector with the greatest distance to its assigned centroid.
func (km *KMeans) findFarthestVector(vectors []float32, n int, assignments []int, centroids []float32) int {
	maxDist := float32(-1)
	maxIdx := 0
	for i := 0; i < n; i++ {
		c := assignments[i]
		v := vectors[i*km.Dim : (i+1)*km.Dim]
		centroid := centroids[c*km.Dim : (c+1)*km.Dim]
		d := km.distFn(v, centroid)
		if d > maxDist {
			maxDist = d
			maxIdx = i
		}
	}
	return maxIdx
}

// NearestCentroid is exported for use by IVFIndex to assign new vectors.
func (km *KMeans) NearestCentroid(v []float32) int {
	return km.nearestCentroid(v)
}

// NearestKCentroids returns the indices and distances of the k nearest centroids.
func (km *KMeans) NearestKCentroids(v []float32, k int) []CentroidDist {
	if k > km.K {
		k = km.K
	}
	dists := make([]CentroidDist, km.K)
	for c := 0; c < km.K; c++ {
		centroid := km.Centroids[c*km.Dim : (c+1)*km.Dim]
		dists[c] = CentroidDist{Index: c, Distance: km.distFn(v, centroid)}
	}
	// Partial sort: get top-k (simple selection for small K)
	for i := 0; i < k; i++ {
		minIdx := i
		for j := i + 1; j < len(dists); j++ {
			if dists[j].Distance < dists[minIdx].Distance {
				minIdx = j
			}
		}
		dists[i], dists[minIdx] = dists[minIdx], dists[i]
	}
	return dists[:k]
}

// CentroidDist holds a centroid index and its distance to a query.
type CentroidDist struct {
	Index    int
	Distance float32
}
