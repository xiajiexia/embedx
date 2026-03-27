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
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

var (
	defaultPort    = getEnv("EMBEDX_PORT", "11434")
	defaultModel   = getEnv("EMBEDX_MODEL", "BAAI/bge-small-zh-v1.5")
	embedxPort     int
	requestCounter uint64
)

var (
	pythonProcess *exec.Cmd
	pythonStdin   io.WriteCloser
	pythonStdout  *bufio.Reader
	pythonFile    *os.File
	mu            sync.Mutex
	pythonAlive   int32 = 1
)

type PyRequest struct {
	Command   string   `json:"command"`
	Model     string   `json:"model,omitempty"`
	ModelName string   `json:"model_name,omitempty"`
	Texts     []string `json:"texts,omitempty"`
}

type PyResponse struct {
	Type       string     `json:"type,omitempty"`
	Event      string     `json:"event,omitempty"`
	Status     string     `json:"status,omitempty"`
	Model      string     `json:"model,omitempty"`
	Dimensions int        `json:"dimensions,omitempty"`
	MaxLength  int        `json:"max_length,omitempty"`
	Error      string     `json:"error,omitempty"`
	Cached     []string   `json:"cached,omitempty"`
	Embeddings [][]float32 `json:"embeddings,omitempty"`
}

type EmbedRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	Stream bool   `json:"stream"`
}

type EmbedResponse struct {
	Embedding []float32 `json:"embedding"`
}

type PullRequest struct {
	Name   string `json:"name"`
	Stream bool   `json:"stream"`
}

type ShowRequest struct {
	Name string `json:"name"`
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

func recoverPanic(name string) {
	if r := recover(); r != nil {
		log.Printf("[ERROR] PANIC recovered in %s: value=%v\n%s", name, r, string(debug.Stack()))
	}
}

func logf(level, msg string, kv ...interface{}) {
	log.Printf("[%s] [%s] %s %v", time.Now().Format("2006-01-02 15:04:05.000"), level, msg, kv)
}

func infof(msg string, kv ...interface{})  { logf("INFO", msg, kv...) }
func warnf(msg string, kv ...interface{})   { logf("WARN", msg, kv...) }
func errorf(msg string, kv ...interface{})  { logf("ERROR", msg, kv...) }
func debugf(msg string, kv ...interface{}) {
	if os.Getenv("EMBEDX_DEBUG") != "" {
		logf("DEBUG", msg, kv...)
	}
}

func startPythonBackend() error {
	infof("Starting Python backend", "model", defaultModel, "exe_dir", getExeDir())

	pythonProcess = exec.Command("python3", "-u", "embed.py", defaultModel)

	if dir := getExeDir(); dir != "." {
		pythonProcess.Dir = dir
	}

	var err error
	pythonStdin, err = pythonProcess.StdinPipe()
	if err != nil {
		return fmt.Errorf("create stdin pipe: %w", err)
	}

	stderr, err := pythonProcess.StderrPipe()
	if err != nil {
		return fmt.Errorf("create stderr pipe: %w", err)
	}
	go func() {
		scanner := bufio.NewScanner(stderr)
		for scanner.Scan() {
			line := scanner.Text()
			if line != "" {
				debugf("python stderr", "line", line)
			}
		}
	}()

	stdout, err := pythonProcess.StdoutPipe()
	if err != nil {
		return fmt.Errorf("create stdout pipe: %w", err)
	}
	// Keep raw file reference for SetReadDeadline
	if f, ok := stdout.(*os.File); ok {
		pythonFile = f
	}
	pythonStdout = bufio.NewReader(stdout)

	if err := pythonProcess.Start(); err != nil {
		return fmt.Errorf("start python: %w", err)
	}

	infof("Python process started", "pid", pythonProcess.Process.Pid)
	atomic.StoreInt32(&pythonAlive, 1)

	go func() {
		defer recoverPanic("pythonProcess.Wait")
		err := pythonProcess.Wait()
		atomic.StoreInt32(&pythonAlive, 0)
		if err != nil {
			errorf("Python process exited with error", "err", err)
		} else {
			infof("Python process exited normally")
		}
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	for attempt := 1; ; attempt++ {
		line, err := readLineWithTimeout(30 * time.Second)
		if err != nil {
			if err == context.DeadlineExceeded {
				errorf("Timeout reading python stdout", "attempt", attempt, "timeout", "30s")
				continue
			}
			if ctx.Err() != nil {
				return fmt.Errorf("timeout waiting for python ready: %w", ctx.Err())
			}
			return fmt.Errorf("read python stdout: %w", err)
		}

		if len(strings.TrimSpace(line)) == 0 {
			continue
		}

		var resp PyResponse
		if err := json.Unmarshal([]byte(line), &resp); err != nil {
			warnf("Non-JSON line from python, skipping", "line", strings.TrimSpace(line)[:200], "err", err)
			continue
		}

		infof("Python backend event", "event", resp.Event, "model", resp.Model, "dims", resp.Dimensions, "error", resp.Error)

		if resp.Event == "ready" {
			if resp.Error != "" {
				warnf("Python backend ready with error", "error", resp.Error)
			} else {
				infof("Python backend ready", "model", resp.Model, "dimensions", resp.Dimensions, "max_length", resp.MaxLength, "cached", resp.Cached)
			}
			break
		}
	}

	return nil
}

// readLineWithTimeout reads a newline-delimited line with a timeout
func readLineWithTimeout(timeout time.Duration) (string, error) {
	lineCh := make(chan string, 1)
	errCh := make(chan error, 1)

	go func() {
		line, err := pythonStdout.ReadString('\n')
		if err != nil {
			errCh <- err
			return
		}
		lineCh <- line
	}()

	select {
	case line := <-lineCh:
		return line, nil
	case err := <-errCh:
		return "", err
	case <-time.After(timeout):
		return "", context.DeadlineExceeded
	}
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

func pythonCall(ctx context.Context, req PyRequest) (*PyResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	reqID := atomic.AddUint64(&requestCounter, 1)
	debugf("pythonCall START", "req_id", reqID, "command", req.Command, "model_name", req.ModelName)

	ctx, cancel := context.WithTimeout(ctx, 120*time.Second)
	defer cancel()

	mu.Lock()
	if atomic.LoadInt32(&pythonAlive) == 0 {
		mu.Unlock()
		errorf("pythonCall: python process dead", "req_id", reqID)
		return nil, fmt.Errorf("python process is not running")
	}
	if _, err := fmt.Fprintf(pythonStdin, "%s\n", body); err != nil {
		mu.Unlock()
		errorf("pythonCall: write failed", "req_id", reqID, "err", err)
		return nil, fmt.Errorf("write to python: %w", err)
	}
	mu.Unlock()

	respCh := make(chan *PyResponse, 1)
	errCh := make(chan error, 1)

	go func() {
		defer recoverPanic("pythonCall ReadString")
		line, err := pythonStdout.ReadString('\n')
		if err != nil {
			errCh <- err
			return
		}
		var resp PyResponse
		if err := json.Unmarshal([]byte(line), &resp); err != nil {
			errCh <- fmt.Errorf("decode python response: %w (line: %s)", err, line[:min(len(line), 100)])
			return
		}
		respCh <- &resp
	}()

	select {
	case <-ctx.Done():
		errorf("pythonCall TIMEOUT", "req_id", reqID, "command", req.Command, "model_name", req.ModelName)
		return nil, ctx.Err()
	case err := <-errCh:
		errorf("pythonCall READ ERROR", "req_id", reqID, "err", err)
		return nil, fmt.Errorf("read from python: %w", err)
	case resp := <-respCh:
		debugf("pythonCall DONE", "req_id", reqID, "resp_type", resp.Type, "resp_event", resp.Event, "resp_error", resp.Error)
		return resp, nil
	}
}

func pythonEmbed(ctx context.Context, text string, modelName string) ([]float32, error) {
	req := PyRequest{
		Command:  "embed",
		ModelName: modelName,
		Texts:    []string{text},
	}

	start := time.Now()
	resp, err := pythonCall(ctx, req)
	elapsed := time.Since(start)

	if err != nil {
		errorf("pythonEmbed FAILED", "text_len", len(text), "model", modelName, "err", err, "elapsed", elapsed.String())
		return nil, err
	}

	if resp.Error != "" {
		errorf("pythonEmbed python error", "text_len", len(text), "model", modelName, "error", resp.Error, "elapsed", elapsed.String())
		return nil, fmt.Errorf("python error: %s", resp.Error)
	}

	if len(resp.Embeddings) == 0 {
		errorf("pythonEmbed no embeddings", "text_len", len(text), "model", modelName)
		return nil, fmt.Errorf("no embeddings returned")
	}

	debugf("pythonEmbed OK", "text_len", len(text), "model", modelName, "embedding_dim", len(resp.Embeddings[0]), "elapsed", elapsed.String())
	return resp.Embeddings[0], nil
}

func pythonLoadModel(ctx context.Context, modelName string) (*PyResponse, error) {
	req := PyRequest{Command: "load", ModelName: modelName}
	start := time.Now()
	resp, err := pythonCall(ctx, req)
	if err != nil {
		errorf("pythonLoadModel FAILED", "model", modelName, "err", err, "elapsed", time.Since(start).String())
	} else {
		infof("pythonLoadModel OK", "model", modelName, "dimensions", resp.Dimensions, "elapsed", time.Since(start).String())
	}
	return resp, err
}

func pythonPullModel(ctx context.Context, modelName string) (*PyResponse, error) {
	req := PyRequest{Command: "pull", ModelName: modelName}
	start := time.Now()
	resp, err := pythonCall(ctx, req)
	if err != nil {
		errorf("pythonPullModel FAILED", "model", modelName, "err", err, "elapsed", time.Since(start).String())
	} else {
		infof("pythonPullModel OK", "model", modelName, "dimensions", resp.Dimensions, "elapsed", time.Since(start).String())
	}
	return resp, err
}

func pythonListCached(ctx context.Context) (*PyResponse, error) {
	return pythonCall(ctx, PyRequest{Command: "list_cached"})
}

func withRecover(h func(http.ResponseWriter, *http.Request)) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		defer recoverPanic(r.URL.Path)
		h(w, r)
	}
}

func embedHandler(w http.ResponseWriter, r *http.Request) {
	reqID := atomic.AddUint64(&requestCounter, 1)
	start := time.Now()

	debugf("embedHandler START", "req_id", reqID, "path", r.URL.Path, "method", r.Method)

	if r.URL.Path != "/api/embeddings" {
		http.Error(w, "not found", http.StatusNotFound)
		return
	}

	var req EmbedRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		errorf("embedHandler bad request", "req_id", reqID, "err", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	modelName := req.Model
	if modelName == "" {
		modelName = defaultModel
	}

	if atomic.LoadInt32(&pythonAlive) == 0 {
		errorf("embedHandler python dead", "req_id", reqID)
		http.Error(w, "python backend not running", http.StatusInternalServerError)
		return
	}

	embedding, err := pythonEmbed(r.Context(), req.Prompt, modelName)
	if err != nil {
		errorf("embedHandler FAILED", "req_id", reqID, "model", modelName, "prompt_len", len(req.Prompt), "err", err, "elapsed", time.Since(start).String())
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	debugf("embedHandler OK", "req_id", reqID, "model", modelName, "prompt_len", len(req.Prompt), "embedding_dim", len(embedding), "elapsed", time.Since(start).String())
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(EmbedResponse{Embedding: embedding})
}

func pullHandler(w http.ResponseWriter, r *http.Request) {
	reqID := atomic.AddUint64(&requestCounter, 1)
	start := time.Now()

	debugf("pullHandler START", "req_id", reqID, "path", r.URL.Path, "method", r.Method)

	if r.URL.Path != "/api/pull" {
		http.Error(w, "not found", http.StatusNotFound)
		return
	}

	var req PullRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		errorf("pullHandler bad request", "req_id", reqID, "err", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if req.Name == "" {
		http.Error(w, "model name is required", http.StatusBadRequest)
		return
	}

	if req.Stream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "streaming not supported", http.StatusInternalServerError)
			return
		}

		infof("pullHandler streaming START", "req_id", reqID, "model", req.Name)

		done := make(chan error, 1)
		go func() {
			defer recoverPanic("pullHandler pythonPullModel")
			resp, err := pythonPullModel(r.Context(), req.Name)
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

		fmt.Fprintf(w, "event: status\ndata: {\"status\":\"pulling\",\"model\":\"%s\"}\n\n", req.Name)
		flusher.Flush()

		select {
		case err := <-done:
			if err != nil {
				errorf("pullHandler streaming ERROR", "req_id", reqID, "err", err)
				fmt.Fprintf(w, "event: error\ndata: {\"error\":\"%s\"}\n\n", err)
			} else {
				infof("pullHandler streaming DONE", "req_id", reqID, "model", req.Name, "elapsed", time.Since(start).String())
				fmt.Fprintf(w, "event: done\ndata: {\"status\":\"success\",\"model\":\"%s\"}\n\n", req.Name)
			}
			flusher.Flush()
		case <-r.Context().Done():
			infof("pullHandler streaming CANCELLED", "req_id", reqID)
			return
		}
	} else {
		resp, err := pythonPullModel(r.Context(), req.Name)
		if err != nil {
			errorf("pullHandler FAILED", "req_id", reqID, "model", req.Name, "err", err, "elapsed", time.Since(start).String())
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if resp.Error != "" {
			errorf("pullHandler python error", "req_id", reqID, "model", req.Name, "error", resp.Error)
			http.Error(w, resp.Error, http.StatusInternalServerError)
			return
		}

		infof("pullHandler OK", "req_id", reqID, "model", req.Name, "dimensions", resp.Dimensions, "elapsed", time.Since(start).String())
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{"status": "success", "model": req.Name, "dimensions": resp.Dimensions})
	}
}

func showHandler(w http.ResponseWriter, r *http.Request) {
	reqID := atomic.AddUint64(&requestCounter, 1)
	start := time.Now()

	debugf("showHandler START", "req_id", reqID, "method", r.Method)

	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req ShowRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		errorf("showHandler bad request", "req_id", reqID, "err", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	resp, err := pythonLoadModel(r.Context(), req.Name)
	if err != nil {
		errorf("showHandler FAILED", "req_id", reqID, "model", req.Name, "err", err, "elapsed", time.Since(start).String())
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	if resp.Error != "" {
		errorf("showHandler python error", "req_id", reqID, "model", req.Name, "error", resp.Error)
		http.Error(w, resp.Error, http.StatusInternalServerError)
		return
	}

	infof("showHandler OK", "req_id", reqID, "model", req.Name, "dimensions", resp.Dimensions, "elapsed", time.Since(start).String())
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{"name": resp.Model, "dimensions": resp.Dimensions, "max_length": resp.MaxLength})
}

func modelsHandler(w http.ResponseWriter, r *http.Request) {
	reqID := atomic.AddUint64(&requestCounter, 1)
	start := time.Now()

	resp, err := pythonListCached(r.Context())
	if err != nil {
		errorf("modelsHandler FAILED", "req_id", reqID, "err", err, "elapsed", time.Since(start).String())
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	models := []map[string]interface{}{}
	if resp.Cached != nil {
		for _, name := range resp.Cached {
			models = append(models, map[string]interface{}{"name": name, "model": name, "size": 0})
		}
	} else {
		models = []map[string]interface{}{{"name": defaultModel, "model": defaultModel, "size": 0}}
	}

	debugf("modelsHandler OK", "req_id", reqID, "num_models", len(models), "elapsed", time.Since(start).String())
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{"models": models})
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	reqID := atomic.AddUint64(&requestCounter, 1)
	status := "ok"
	if atomic.LoadInt32(&pythonAlive) == 0 {
		status = "python_dead"
	}
	debugf("healthHandler", "req_id", reqID, "status", status, "python_alive", atomic.LoadInt32(&pythonAlive))
	w.WriteHeader(http.StatusOK)
}

func createHandler(w http.ResponseWriter, r *http.Request) {
	reqID := atomic.AddUint64(&requestCounter, 1)
	start := time.Now()

	debugf("createHandler START", "req_id", reqID, "method", r.Method)

	var req PullRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		errorf("createHandler bad request", "req_id", reqID, "err", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	resp, err := pythonLoadModel(r.Context(), req.Name)
	if err != nil {
		errorf("createHandler FAILED", "req_id", reqID, "model", req.Name, "err", err, "elapsed", time.Since(start).String())
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	if resp.Error != "" {
		errorf("createHandler python error", "req_id", reqID, "model", req.Name, "error", resp.Error)
		http.Error(w, resp.Error, http.StatusInternalServerError)
		return
	}

	infof("createHandler OK", "req_id", reqID, "model", req.Name, "dimensions", resp.Dimensions, "elapsed", time.Since(start).String())
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{"status": "success", "model": resp.Model})
}

func main() {
	embedxPort = getEnvInt("EMBEDX_PORT", 11434)

	if os.Getenv("EMBEDX_DEBUG") != "" {
		log.SetOutput(os.Stderr)
	}

	infof("embedx starting",
		"port", embedxPort,
		"default_model", defaultModel,
		"go_version", runtime.Version(),
		"pid", os.Getpid(),
	)

	if err := startPythonBackend(); err != nil {
		errorf("startPythonBackend FAILED", "err", err)
		log.Fatalf("Failed to start Python backend: %v", err)
	}
	defer func() {
		if pythonProcess != nil && pythonProcess.Process != nil {
			infof("Killing python process", "pid", pythonProcess.Process.Pid)
			pythonProcess.Process.Kill()
		}
	}()

	mux := http.NewServeMux()
	mux.HandleFunc("/api/embeddings", withRecover(embedHandler))
	mux.HandleFunc("/api/pull", withRecover(pullHandler))
	mux.HandleFunc("/api/create", withRecover(createHandler))
	mux.HandleFunc("/api/show", withRecover(showHandler))
	mux.HandleFunc("/api/tags", withRecover(modelsHandler))
	mux.HandleFunc("/models", withRecover(modelsHandler))
	mux.HandleFunc("/health", withRecover(healthHandler))

	addr := fmt.Sprintf("0.0.0.0:%d", embedxPort)
	infof("embedx server starting", "addr", addr)

	server := &http.Server{
		Addr:         addr,
		Handler:      mux,
		ReadTimeout:  120 * time.Second,
		WriteTimeout: 120 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	if err := server.ListenAndServe(); err != nil {
		errorf("Server.ListenAndServe error", "err", err)
		log.Fatalf("Server error: %v", err)
	}
}
