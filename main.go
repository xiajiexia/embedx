package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

var (
	defaultPort    = getEnv("EMBEDX_PORT", "11434")
	defaultModel   = getEnv("EMBEDX_MODEL", "BAAI/bge-small-zh-v1.5")
	embedxPort     int
	pythonProcess  *exec.Cmd
	pythonStdin    io.WriteCloser
	pythonStdout  *bufio.Reader
)

// Ollama-compatible request/response types
type EmbedRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	Stream bool   `json:"stream"`
}

type EmbedResponse struct {
	Embedding []float32 `json:"embedding"`
}

type FastEmbedRequest struct {
	Texts []string `json:"texts"`
}

type FastEmbedResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
}

type FastEmbedError struct {
	Error string `json:"error"`
}

func getEnv(key, defaultVal string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return defaultVal
}

func getEnvInt(key string, defaultVal int) int {
	if v := os.Getenv(key); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			return i
		}
	}
	return defaultVal
}

func startPythonBackend() error {
	model := defaultModel
	pythonProcess = exec.Command("python3", "-u", "embed.py", model)
	
	// Set working directory to the executable's directory
	if dir := getExeDir(); dir != "." {
		pythonProcess.Dir = dir
	}
	
	var err error
	pythonStdin, err = pythonProcess.StdinPipe()
	if err != nil {
		return fmt.Errorf("create stdin pipe: %w", err)
	}
	
	pythonProcess.Stderr = os.Stderr
	
	stdout, err := pythonProcess.StdoutPipe()
	if err != nil {
		return fmt.Errorf("create stdout pipe: %w", err)
	}
	pythonStdout = bufio.NewReader(stdout)
	
	if err := pythonProcess.Start(); err != nil {
		return fmt.Errorf("start python: %w", err)
	}
	
	// Wait for model to load (stderr will say "Model loaded, ready for input")
	for i := 0; i < 60; i++ {
		line, err := pythonStdout.ReadString('\n')
		if err != nil {
			return fmt.Errorf("waiting for model load: %w", err)
		}
		if strings.Contains(line, "READY") {
			break
		}
	}
	
	log.Printf("Python backend ready")
	return nil
}

func getExeDir() string {
	exe, err := os.Executable()
	if err != nil {
		return "."
	}
	idx := strings.LastIndex(exe, "/")
	if idx > 0 {
		return exe[:idx]
	}
	return "."
}

func embed(ctx context.Context, text string) ([]float32, error) {
	reqBody := FastEmbedRequest{Texts: []string{text}}
	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}
	
	// Set deadline for this operation
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	
	// Write request
	if _, err := fmt.Fprintf(pythonStdin, "%s\n", body); err != nil {
		return nil, fmt.Errorf("write to python: %w", err)
	}
	
	// Read response
	respCh := make(chan []byte, 1)
	errCh := make(chan error, 1)
	
	go func() {
		line, err := pythonStdout.ReadString('\n')
		if err != nil {
			errCh <- err
			return
		}
		respCh <- []byte(line)
	}()
	
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case err := <-errCh:
		return nil, fmt.Errorf("read from python: %w", err)
	case respBytes := <-respCh:
		var result FastEmbedResponse
		if err := json.Unmarshal(respBytes, &result); err != nil {
			// Check if it's an error response
			var errResp FastEmbedError
			if json.Unmarshal(respBytes, &errResp) == nil {
				return nil, fmt.Errorf("fastembed error: %s", errResp.Error)
			}
			return nil, fmt.Errorf("decode response: %w", err)
		}
		
		if len(result.Embeddings) == 0 {
			return nil, fmt.Errorf("no embeddings returned")
		}
		
		return result.Embeddings[0], nil
	}
}

func embedHandler(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/api/embeddings" {
		http.Error(w, "not found", http.StatusNotFound)
		return
	}

	var req EmbedRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	ctx := r.Context()

	embedding, err := embed(ctx, req.Prompt)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	response := EmbedResponse{Embedding: embedding}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func modelsHandler(w http.ResponseWriter, r *http.Request) {
	models := []map[string]interface{}{
		{
			"name":  defaultModel,
			"model": defaultModel,
			"size":  0,
		},
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"models": models,
	})
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
}

func main() {
	embedxPort = getEnvInt("EMBEDX_PORT", 11434)

	if err := startPythonBackend(); err != nil {
		log.Fatalf("Failed to start Python backend: %v", err)
	}
	defer pythonProcess.Process.Kill()

	mux := http.NewServeMux()
	mux.HandleFunc("/api/embeddings", embedHandler)
	mux.HandleFunc("/api/tags", modelsHandler)
	mux.HandleFunc("/models", modelsHandler)
	mux.HandleFunc("/health", healthHandler)

	addr := fmt.Sprintf("0.0.0.0:%d", embedxPort)
	log.Printf("embedx starting on %s", addr)
	log.Printf("Default model: %s", defaultModel)

	server := &http.Server{
		Addr:    addr,
		Handler: mux,
	}

	if err := server.ListenAndServe(); err != nil {
		log.Fatalf("Server error: %v", err)
	}
}
