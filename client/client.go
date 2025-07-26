package main

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"
)

func main() {
	// Store a vector
	storeURL := "http://localhost:8080/store"
	storeBody := []byte(`{"text": "hello world"}`)
	req, err := http.NewRequest("POST", storeURL, bytes.NewBuffer(storeBody))
	if err != nil {
		fmt.Println("Error creating request:", err)
		return
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		fmt.Println("Error sending request:", err)
		return
	}
	defer resp.Body.Close()

	body, _ := ioutil.ReadAll(resp.Body)
	fmt.Println("Store Response:", string(body))

	// Search for a vector using KNN
	searchURL := "http://localhost:8080/search"
	searchBody := []byte(`{"text": "greetings earthling", "k": 1, "algorithm": "knn"}`)
	req, err = http.NewRequest("POST", searchURL, bytes.NewBuffer(searchBody))
	if err != nil {
		fmt.Println("Error creating request:", err)
		return
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err = client.Do(req)
	if err != nil {
		fmt.Println("Error sending request:", err)
		return
	}
	defer resp.Body.Close()

	body, _ = ioutil.ReadAll(resp.Body)
	fmt.Println("Search Response (KNN):", string(body))

	// Search for a vector using ANN
	searchBody = []byte(`{"text": "greetings earthling", "k": 1, "algorithm": "ann"}`)
	req, err = http.NewRequest("POST", searchURL, bytes.NewBuffer(searchBody))
	if err != nil {
		fmt.Println("Error creating request:", err)
		return
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err = client.Do(req)
	if err != nil {
		fmt.Println("Error sending request:", err)
		return
	}
	defer resp.Body.Close()

	body, _ = ioutil.ReadAll(resp.Body)
	fmt.Println("Search Response (ANN):", string(body))
}