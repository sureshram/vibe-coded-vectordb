
package vector

import (
	"encoding/gob"
	"fmt"
	"hash/fnv"
	"math"
	"os"
	"sort"

	"github.com/fogfish/hnsw"
	hnsw_vector "github.com/fogfish/hnsw/vector"
	surface "github.com/kshard/vector"
)

// Vector represents a single vector with its associated data.
type Vector struct {
	ID   string
	Vec  []float32
	Text string
}

// DB is the in-memory vector database.
type DB struct {
	Vectors map[string]Vector
	HNSW    *hnsw.HNSW[hnsw_vector.VF32]
}

// NewDB creates a new vector database.
func NewDB() *DB {
	return &DB{
		Vectors: make(map[string]Vector),
		HNSW:    hnsw.New[hnsw_vector.VF32](hnsw_vector.SurfaceVF32(surface.Cosine())),
	}
}

// Add adds a vector to the database.
func (db *DB) Add(v Vector) error {
	if _, exists := db.Vectors[v.ID]; exists {
		return fmt.Errorf("vector with ID %s already exists", v.ID)
	}
	db.Vectors[v.ID] = v
	db.HNSW.Insert(hnsw_vector.VF32{Key: hashString(v.ID), Vec: v.Vec})
	return nil
}

// KNNResult represents a single result from a k-NN search.
type KNNResult struct {
	ID       string
	Distance float32
}

// KNN performs a k-Nearest Neighbors search.
func (db *DB) KNN(query []float32, k int) ([]KNNResult, error) {
	if k <= 0 {
		return nil, fmt.Errorf("k must be a positive integer")
	}

	var results []KNNResult

	for _, v := range db.Vectors {
		dist, err := cosineSimilarity(query, v.Vec)
		if err != nil {
			return nil, err
		}
		results = append(results, KNNResult{ID: v.ID, Distance: dist})
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Distance > results[j].Distance // Cosine similarity: higher is better
	})

	if len(results) > k {
		return results[:k], nil
	}

	return results, nil
}

// ANN performs an Approximate Nearest Neighbor search.
func (db *DB) ANN(query []float32, k int) ([]KNNResult, error) {
	if k <= 0 {
		return nil, fmt.Errorf("k must be a positive integer")
	}

	queryVF32 := hnsw_vector.VF32{Key: hashString("query"), Vec: query}
	// The efSearch parameter (10) is the size of the dynamic list for the search.
	matches := db.HNSW.Search(queryVF32, k, 10)

	var results []KNNResult
	for _, item := range matches {
		// Find the original string ID from the Vectors map using the hashed ID
		var originalID string
		for _, v := range db.Vectors {
			if hashString(v.ID) == item.Key {
				originalID = v.ID
				break
			}
		}
		results = append(results, KNNResult{ID: originalID, Distance: db.HNSW.Distance(item, queryVF32)})
	}

	return results, nil
}

// Save saves the database to a file.
func (db *DB) Save(path string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	return encoder.Encode(db.Vectors)
}

// Load loads the database from a file.
func (db *DB) Load(path string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	return decoder.Decode(&db.Vectors)
}

// cosineSimilarity calculates the cosine similarity between two vectors.
func cosineSimilarity(a, b []float32) (float32, error) {
	if len(a) != len(b) {
		return 0, fmt.Errorf("vectors must have the same dimension")
	}

	var dotProduct float32
	var normA, normB float32

	for i := 0; i < len(a); i++ {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0, fmt.Errorf("cannot compute similarity with a zero vector")
	}

	return dotProduct / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB)))), nil
}

// hashString converts a string to a uint32 hash.
func hashString(s string) uint32 {
	h := fnv.New32a()
	h.Write([]byte(s))
	return h.Sum32()
}
