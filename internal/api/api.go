
package api

import (
	"encoding/json"
	"net/http"

	"github.com/google/uuid"
	"github.com/suresh/kanaja/internal/embedding"
	"github.com/suresh/kanaja/internal/vector"
)

// API holds the dependencies for the API.
type API struct {
	DB       *vector.DB
	Embedder embedding.Embedder
}

// StoreRequest is the request body for the /store endpoint.
type StoreRequest struct {
	Text string `json:"text"`
}

// StoreResponse is the response body for the /store endpoint.
type StoreResponse struct {
	ID string `json:"id"`
}

// SearchRequest is the request body for the /search endpoint.
type SearchRequest struct {
	Text      string `json:"text"`
	K         int    `json:"k"`
	Algorithm string `json:"algorithm"`
}

// SearchResponse is the response body for the /search endpoint.
type SearchResponse struct {
	Results []vector.KNNResult `json:"results"`
}

// Store handles the /store endpoint.
func (a *API) Store(w http.ResponseWriter, r *http.Request) {
	var req StoreRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	vec, err := a.Embedder.Embed(req.Text)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	id := uuid.New().String()
	if err := a.DB.Add(vector.Vector{ID: id, Vec: vec, Text: req.Text}); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(StoreResponse{ID: id})
}

// Search handles the /search endpoint.
func (a *API) Search(w http.ResponseWriter, r *http.Request) {
	var req SearchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	vec, err := a.Embedder.Embed(req.Text)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	var results []vector.KNNResult
	if req.Algorithm == "knn" {
		results, err = a.DB.KNN(vec, req.K)
	} else if req.Algorithm == "ann" {
		results, err = a.DB.ANN(vec, req.K)
	} else {
		http.Error(w, "invalid algorithm", http.StatusBadRequest)
		return
	}

	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(SearchResponse{Results: results})
}
