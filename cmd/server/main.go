
package main

import (
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/suresh/kanaja/internal/api"
	"github.com/suresh/kanaja/internal/embedding"
	"github.com/suresh/kanaja/internal/vector"
)

func main() {
	// Initialize the vector database
	db := vector.NewDB()

	// Load the GloVe model
	embedder, err := embedding.NewGloVe("glove.6B.50d.txt")
	if err != nil {
		log.Fatalf("Failed to load GloVe model: %v", err)
	}

	// Initialize the API
	api := &api.API{
		DB:       db,
		Embedder: embedder,
	}

	// Define the HTTP routes
	http.HandleFunc("/store", api.Store)
	http.HandleFunc("/search", api.Search)

	// Start the server
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	fmt.Printf("Server listening on port %s...\n", port)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}
