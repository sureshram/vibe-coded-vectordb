package embedding

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// Embedder is the interface for text embedding.
type Embedder interface {
	Embed(text string) ([]float32, error)
}

// GloVe is a pre-trained GloVe model.
type GloVe struct {
	wordVectors map[string][]float32
	dimension   int
}

// NewGloVe creates a new GloVe embedder from a file.
func NewGloVe(path string) (*GloVe, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	wordVectors := make(map[string][]float32)
	var dimension int

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Split(line, " ")
		word := parts[0]

		if dimension == 0 {
			dimension = len(parts) - 1
		}

		var vector []float32
		for _, s := range parts[1:] {
			f, err := strconv.ParseFloat(s, 32)
			if err != nil {
				return nil, err
			}
			vector = append(vector, float32(f))
		}

		wordVectors[word] = vector
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return &GloVe{wordVectors: wordVectors, dimension: dimension}, nil
}

// Embed generates an embedding for a given text.
func (g *GloVe) Embed(text string) ([]float32, error) {
	words := strings.Fields(strings.ToLower(text))
	if len(words) == 0 {
		return nil, fmt.Errorf("cannot embed empty text")
	}

	embedding := make([]float32, g.dimension)
	var count int

	for _, word := range words {
		if vector, ok := g.wordVectors[word]; ok {
			for i, v := range vector {
				embedding[i] += v
			}
			count++
		}
	}

	if count == 0 {
		return nil, fmt.Errorf("no words in the text found in the model")
	}

	for i := range embedding {
		embedding[i] /= float32(count)
	}

	// Pad the embedding to the next multiple of 4
	padding := 4 - (len(embedding) % 4)
	if padding != 4 {
		for i := 0; i < padding; i++ {
			embedding = append(embedding, 0.0)
		}
	}

	return embedding, nil
}