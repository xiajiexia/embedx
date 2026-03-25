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
	"sync"
	"time"
)

// Environment config
var (
	defaultPort  = getEnv("EMBEDX_PORT", "11434")
	defaultModel = getEnv("EMBEDX_MODEL", "BAAI/bge-small-zh-v1.5")
	embedxPort  int
)

var (
	pythonProcess *exec.Cmd
	pythonStdin   io.WriteCloser
	pythonStdout  *bufio.Reader
	mu            sync.Mutex // protects python stdin writes
)

// --- Protocol types ---

type PyRequest struct {
	Command  string `json:"command"`
	Model    string `json:"model,omitempty"`
	ModelName string `json:"model_name,omitempty"`
	Texts    []string `json:"texts,omitempty"`
}

type PyResponse struct {
	Type      string `json:"type,omitempty"`
	Event     string `json:"event,omitempty"`
	Status    string `json:"status,omitempty"`
	Model     string `json:"model,omitempty"`
	Dimensions int    `json:"dimensions,omitempty"`
	MaxLength int    `json:"max_length,omitempty"`
	Error     string `json:"error,omitempty"`
	Cached    []string `json:"cached,omitempty"`
	Embeddings [][]float32 `json:"embeddings,omitempty"`
}

// Ollama-compatible request/response types
type EmbedRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	Stream bool   `json:"stream"`
}

type EmbedResponse struct {
	Embedding []float32 `json:"embedding"`
}

// Pull request (Ollama-compatible)
type PullRequest struct {
	Name   string `json:"name"`
	Stream bool   `json:"stream"`
}

// Show request
type ShowRequest struct {
	Name string `json:"name"`
}

// Model response
type ModelInfo struct {
	Name        string `json:"name"`
	Size       int64  `json:"size"`
	Digest     string `json:"digest,omitempty"`
	Dimensions  int    `json:"dimensions,omitempty"`
}

// ============================================================================
// Python backend lifecycle
// ============================================================================

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
	pythonProcess = exec.Command("python3", "-u", "embed.py", defaultModel)

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

	// Read ready event
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	for {
		line, err := pythonStdout.ReadString('\n')
		if err != nil {
			if ctx.Err() != nil {
				return fmt.Errorf("timeout waiting for python ready: %w", ctx.Err())
			}
			return fmt.Errorf("read python stdout: %w", err)
		}

		var resp PyResponse
		if err := json.Unmarshal([]byte(line), &resp); err != nil {
			continue // skip non-JSON lines (like stderr prints)
		}

		if resp.Event == "ready" {
			if resp.Error != "" {
				log.Printf("Python backend ready with error: %s", resp.Error)
			} else {
				log.Printf("Python backend ready: model=%s dims=%d", resp.Model, resp.Dimensions)
			}
			break
		}
	}

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

// ============================================================================
// Python communication
// ============================================================================

func pythonCall(ctx context.Context, req PyRequest) (*PyResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	ctx, cancel := context.WithTimeout(ctx, 120*time.Second)
	defer cancel()

	mu.Lock()
	if _, err := fmt.Fprintf(pythonStdin, "%s\n", body); err != nil {
		mu.Unlock()
		return nil, fmt.Errorf("write to python: %w", err)
	}
	mu.Unlock()

	respCh := make(chan *PyResponse, 1)
	errCh := make(chan error, 1)

	go func() {
		line, err := pythonStdout.ReadString('\n')
		if err != nil {
			errCh <- err
			return
		}
		var resp PyResponse
		if err := json.Unmarshal([]byte(line), &resp); err != nil {
			errCh <- fmt.Errorf("decode python response: %w", err)
			return
		}
		respCh <- &resp
	}()

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case err := <-errCh:
		return nil, fmt.Errorf("read from python: %w", err)
	case resp := <-respCh:
		return resp, nil
	}
}

func pythonEmbed(ctx context.Context, text string, modelName string) ([]float32, error) {
	req := PyRequest{
		Command:  "embed",
		ModelName: modelName,
		Texts:    []string{text},
	}

	resp, err := pythonCall(ctx, req)
	if err != nil {
		return nil, err
	}

	if resp.Error != "" {
		return nil, fmt.Errorf("python error: %s", resp.Error)
	}

	if len(resp.Embeddings) == 0 {
		return nil, fmt.Errorf("no embeddings returned")
	}

	return resp.Embeddings[0], nil
}

func pythonLoadModel(ctx context.Context, modelName string) (*PyResponse, error) {
	req := PyRequest{
		Command:  "load",
		ModelName: modelName,
	}
	return pythonCall(ctx, req)
}

func pythonPullModel(ctx context.Context, modelName string) (*PyResponse, error) {
	req := PyRequest{
		Command:  "pull",
		ModelName: modelName,
	}
	return pythonCall(ctx, req)
}

func pythonListCached(ctx context.Context) (*PyResponse, error) {
	req := PyRequest{Command: "list_cached"}
	return pythonCall(ctx, req)
}

// ============================================================================
// HTTP Handlers
// ============================================================================

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

	modelName := req.Model
	if modelName == "" {
		modelName = defaultModel
	}

	ctx := r.Context()

	embedding, err := pythonEmbed(ctx, req.Prompt, modelName)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(EmbedResponse{Embedding: embedding})
}

func pullHandler(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/api/pull" {
		http.Error(w, "not found", http.StatusNotFound)
		return
	}

	var req PullRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if req.Name == "" {
		http.Error(w, "model name is required", http.StatusBadRequest)
		return
	}

	ctx := r.Context()

	if req.Stream {
		// SSE streaming response
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "streaming not supported", http.StatusInternalServerError)
			return
		}

		// Start pull and stream status
		done := make(chan error, 1)
		go func() {
			resp, err := pythonPullModel(ctx, req.Name)
			if err != nil {
				done <- err
				return
			}
			if resp.Error != "" {
				done <- fmt.Errorf("%s", resp.Error)
				return
			}
			done <- nil
		}()

		// Send initial progress
		fmt.Fprintf(w, "event: status\ndata: {\"status\":\"pulling\",\"model\":\"%s\"}\n\n", req.Name)
		flusher.Flush()

		select {
		case err := <-done:
			if err != nil {
				fmt.Fprintf(w, "event: error\ndata: {\"error\":\"%s\"}\n\n", err)
			} else {
				fmt.Fprintf(w, "event: done\ndata: {\"status\":\"success\",\"model\":\"%s\"}\n\n", req.Name)
			}
			flusher.Flush()
		case <-r.Context().Done():
			return
		}
	} else {
		// Non-streaming response
		resp, err := pythonPullModel(ctx, req.Name)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if resp.Error != "" {
			http.Error(w, resp.Error, http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status":      "success",
			"model":       req.Name,
			"dimensions":   resp.Dimensions,
		})
	}
}

func showHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req ShowRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	ctx := r.Context()

	// First try to load the model
	resp, err := pythonLoadModel(ctx, req.Name)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	if resp.Error != "" {
		http.Error(w, resp.Error, http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"name":       resp.Model,
		"dimensions": resp.Dimensions,
		"max_length": resp.MaxLength,
	})
}

func modelsHandler(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()

	resp, err := pythonListCached(ctx)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	models := []map[string]interface{}{}
	if resp.Cached != nil {
		for _, name := range resp.Cached {
			models = append(models, map[string]interface{}{
				"name":  name,
				"model": name,
				"size":  0,
			})
		}
	} else {
		// Fallback to default model
		models = []map[string]interface{}{
			{
				"name":  defaultModel,
				"model": defaultModel,
				"size":  0,
			},
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"models": models,
	})
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
}

func createHandler(w http.ResponseWriter, r *http.Request) {
	// Ollama-compatible create endpoint (for model loading)
	var req PullRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	ctx := r.Context()
	resp, err := pythonLoadModel(ctx, req.Name)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	if resp.Error != "" {
		http.Error(w, resp.Error, http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status": "success",
		"model":  resp.Model,
	})
}

// ============================================================================
// Main
// ============================================================================

func main() {
	embedxPort = getEnvInt("EMBEDX_PORT", 11434)

	if err := startPythonBackend(); err != nil {
		log.Fatalf("Failed to start Python backend: %v", err)
	}
	defer func() {
		if pythonProcess != nil && pythonProcess.Process != nil {
			pythonProcess.Process.Kill()
		}
	}()

	mux := http.NewServeMux()

	// Ollama-compatible embedding endpoint
	mux.HandleFunc("/api/embeddings", embedHandler)

	// Ollama-compatible pull endpoint (download model)
	mux.HandleFunc("/api/pull", pullHandler)

	// Ollama-compatible create endpoint (load model into memory)
	mux.HandleFunc("/api/create", createHandler)

	// Ollama-compatible show endpoint (get model info)
	mux.HandleFunc("/api/show", showHandler)

	// List models
	mux.HandleFunc("/api/tags", modelsHandler)
	mux.HandleFunc("/models", modelsHandler)

	// Health check
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
