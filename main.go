package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
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

// pythonReqMu serialises all requests to the Python subprocess — only one request
// can be in-flight at a time because stdin/stdout are a single sequential channel.
var pythonReqMu sync.Mutex

// restartCh signals that Python backend needs to restart. Buffered so non-blocking.
var restartCh  = make(chan struct{}, 1)
var pythonCond = sync.NewCond(&mu)
var pythonReady = true  // Python starts alive (set by main before spawning goroutines)

// ensurePythonRunning ensures the Python backend is fully ready before any request writes to it.
// All requests wait on the same restart goroutine via pythonCond, guaranteeing no request proceeds
// until Python has emitted the "ready" event — eliminating the race where pythonAlive=1 but not ready.
func ensurePythonRunning() error {
	mu.Lock()
	if atomic.LoadInt32(&pythonAlive) == 1 {
		mu.Unlock()
		return nil
	}
	// Signal restart goroutine to start Python. Non-blocking send (buffered channel).
	select {
	case restartCh <- struct{}{}:
	default:
	}
	// Wait until pythonReady==true (set by restart goroutine after "ready" event).
	// Cond.Wait() unlocks mu atomically, so restartGoroutine can acquire it.
	for !pythonReady {
		pythonCond.Wait()
	}
	mu.Unlock()
	return nil
}

// restartGoroutine runs for the lifetime of the process. It serialises all restart attempts
// and guarantees Python is fully ready (ready event seen) before any request goroutine
// is unblocked to write to stdin.
func restartGoroutine() {
	for range restartCh {
		mu.Lock()
		infof("Python backend dead, restarting...")
		if err := startPythonBackendLocked(); err != nil {
			errorf("restart python backend failed", "err", err)
			// Keep pythonAlive=0 so next request retries; Broadcast so blocked callers
			// re-check the loop and either retry (if pythonAlive==0 triggers another restart)
			// or give up. They will return nil on pythonAlive==1.
			pythonCond.Broadcast()
			mu.Unlock()
			continue
		}
		// pythonAlive=1 is set inside startPythonBackendLocked after the ready event.
		// Now wake all blocked callers.
		pythonReady = true
		pythonCond.Broadcast()
		mu.Unlock()
	}
}

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

// Chroma types
type ChromaCreateRequest struct {
	Name      string `json:"name"`
	Dimension int    `json:"dimension"`
	Metric    string `json:"metric"`
}

type ChromaAddRequest struct {
	IDs        []string                 `json:"ids"`
	Embeddings [][]float32               `json:"embeddings"`
	Metadatas  []map[string]interface{} `json:"metadatas,omitempty"`
}

type ChromaQueryRequest struct {
	QueryEmbeddings [][]float32 `json:"query_embeddings"`
	NResults        int          `json:"n_results"`
	Include         []string     `json:"include,omitempty"`
}

type ChromaDeleteRequest struct {
	IDs []string `json:"ids,omitempty"`
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
	mu.Lock()
	defer mu.Unlock()
	return startPythonBackendLocked()
}

// startPythonBackendLocked starts or restarts the Python backend. Caller must hold mu.
func startPythonBackendLocked() error {
	// Clean up any stale process
	if pythonProcess != nil && pythonProcess.Process != nil {
		pythonProcess.Process.Kill()
		pythonProcess.Wait()
	}
	// Mark not-ready so any goroutines woken by a prior crash广播 re-check and wait for us
	pythonReady = false

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
	if f, ok := stdout.(*os.File); ok {
		pythonFile = f
	}
	// Use 256KB buffer to prevent bufio panic when Python outputs large blocks (Go 1.26)
	pythonStdout = bufio.NewReaderSize(stdout, 256*1024)

	if err := pythonProcess.Start(); err != nil {
		return fmt.Errorf("start python: %w", err)
	}
	infof("Python process started", "pid", pythonProcess.Process.Pid)

	go func() {
		defer recoverPanic("pythonProcess.Wait")
		err := pythonProcess.Wait()
		// Log once when transitioning alive -> dead (SwapInt32 prevents duplicate logs)
		if atomic.SwapInt32(&pythonAlive, 0) == 1 {
			if err != nil {
				errorf("Python process exited with error", "err", err)
			} else {
				infof("Python process exited normally")
			}
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
			// Python is fully initialised — mark alive so new requests can proceed
			atomic.StoreInt32(&pythonAlive, 1)
			break
		}
	}

	return nil
}


// readLineFromBufio reads a newline-delimited line from a bufio.Reader,
// avoiding bufio.Reader's ReadString/ReadBytes which call collectFragments
// and can panic with "slice bounds out of range" on certain large reads.
func readLineFromBufio(br *bufio.Reader) (string, error) {
	var buf bytes.Buffer
	for {
		b, err := br.ReadByte()
		if err != nil {
			return buf.String(), err
		}
		if b == '\n' {
			buf.WriteByte('\n')
			return buf.String(), nil
		}
		buf.WriteByte(b)
	}
}

// signalHandler listens for SIGTERM/SIGINT (graceful HTTP drain) and SIGUSR1 (Python backend restart).
func signalHandler(server *http.Server) {
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGTERM, syscall.SIGINT, syscall.SIGUSR1)
	for sig := range sigCh {
		if sig == syscall.SIGUSR1 {
			infof("Received SIGUSR1, restarting Python backend")
			mu.Lock()
			if atomic.LoadInt32(&pythonAlive) == 0 {
				infof("Python backend already restarting")
				mu.Unlock()
				continue
			}
			atomic.StoreInt32(&pythonAlive, 0)
			pythonReady = false
			select {
			case restartCh <- struct{}{}:
			default:
			}
			mu.Unlock()
			infof("Python backend restart signalled")
			continue
		}
		// SIGTERM / SIGINT: graceful HTTP server shutdown
		infof("Received %s, graceful shutdown starting", sig)
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		if err := server.Shutdown(ctx); err != nil {
			errorf("Graceful shutdown error", "err", err)
		} else {
			infof("Graceful shutdown complete, all requests drained")
		}
		cancel()
		signal.Stop(sigCh)
		return
	}
}

// readLineWithTimeout reads a newline-delimited line with a timeout
func readLineWithTimeout(timeout time.Duration) (string, error) {
	lineCh := make(chan string, 1)
	errCh := make(chan error, 1)

	go func() {
		line, err := readLineFromBufio(pythonStdout)
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

func pythonCall(ctx context.Context, req interface{}) (*PyResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	// Extract command for logging
	cmd := ""
	if m, ok := req.(map[string]interface{}); ok {
		cmd, _ = m["command"].(string)
	}

	reqID := atomic.AddUint64(&requestCounter, 1)
	debugf("pythonCall START", "req_id", reqID, "command", cmd)

	ctx, cancel := context.WithTimeout(ctx, 120*time.Second)
	defer cancel()

	// Auto-restart Python backend if it has crashed
	if err := ensurePythonRunning(); err != nil {
		errorf("pythonCall: cannot restart python backend", "req_id", reqID, "err", err)
		return nil, fmt.Errorf("python backend unavailable: %w", err)
	}

	// Serialise write+read so Python sees one JSON request at a time.
	pythonReqMu.Lock()
	if _, err := fmt.Fprintf(pythonStdin, "%s\n", body); err != nil {
		pythonReqMu.Unlock()
		errorf("pythonCall: write failed", "req_id", reqID, "err", err)
		return nil, fmt.Errorf("write to python: %w", err)
	}

	respCh := make(chan *PyResponse, 1)
	errCh := make(chan error, 1)

	go func() {
		defer recoverPanic("pythonCall readLine")
		// pythonReqMu is held by parent; pythonStdout is only accessed here.
		line, err := readLineFromBufio(pythonStdout)
		if err != nil {
			errCh <- err
			return
		}
		var resp PyResponse
		if err := json.Unmarshal([]byte(line), &resp); err != nil {
			previewLen := len(line)
			if previewLen > 100 {
				previewLen = 100
			}
			errCh <- fmt.Errorf("decode python response: %w (line: %s)", err, line[:previewLen])
			return
		}
		respCh <- &resp
	}()

	select {
	case <-ctx.Done():
		// Keep pythonReqMu locked on this goroutine; another request waiting
		// on pythonReqMu will block, which is correct — Python is still busy.
		errorf("pythonCall TIMEOUT", "req_id", reqID, "command", cmd)
		pythonReqMu.Unlock()
		return nil, ctx.Err()
	case err := <-errCh:
		errorf("pythonCall READ ERROR", "req_id", reqID, "err", err)
		pythonReqMu.Unlock()
		return nil, fmt.Errorf("read from python: %w", err)
	case resp := <-respCh:
		debugf("pythonCall DONE", "req_id", reqID, "resp_type", resp.Type, "resp_event", resp.Event, "resp_error", resp.Error)
		pythonReqMu.Unlock()
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

	if err := ensurePythonRunning(); err != nil {
		errorf("embedHandler python unavailable", "req_id", reqID, "err", err)
		http.Error(w, "python backend unavailable: "+err.Error(), http.StatusInternalServerError)
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

// Chroma handlers

func collectionsHandler(w http.ResponseWriter, r *http.Request) {
	reqID := atomic.AddUint64(&requestCounter, 1)
	start := time.Now()

	debugf("collectionsHandler START", "req_id", reqID, "method", r.Method)

	if r.Method == http.MethodGet {
		// List collections
		resp, err := pythonCall(r.Context(), map[string]interface{}{"command": "chroma_list"})
		if err != nil {
			errorf("collectionsHandler GET FAILED", "req_id", reqID, "err", err)
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		debugf("collectionsHandler GET OK", "req_id", reqID, "elapsed", time.Since(start).String())
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
		return
	}

	if r.Method == http.MethodPost {
		// Create collection
		var req ChromaCreateRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			errorf("collectionsHandler POST bad request", "req_id", reqID, "err", err)
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if req.Name == "" {
			http.Error(w, "collection name is required", http.StatusBadRequest)
			return
		}
		dimension := req.Dimension
		if dimension == 0 {
			dimension = 768
		}
		metric := req.Metric
		if metric == "" {
			metric = "cosine"
		}
		resp, err := pythonCall(r.Context(), map[string]interface{}{
			"command":   "chroma_create",
			"name":      req.Name,
			"dimension": dimension,
			"metric":    metric,
		})
		if err != nil {
			errorf("collectionsHandler POST FAILED", "req_id", reqID, "err", err)
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		debugf("collectionsHandler POST OK", "req_id", reqID, "elapsed", time.Since(start).String())
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
		return
	}

	http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
}

func collectionHandler(w http.ResponseWriter, r *http.Request) {
	reqID := atomic.AddUint64(&requestCounter, 1)
	start := time.Now()

	// Extract collection name from URL: /collections/{name}
	parts := strings.SplitN(r.URL.Path, "/", 4)
	name := ""
	if len(parts) >= 3 {
		name = parts[2]
	}

	debugf("collectionHandler START", "req_id", reqID, "method", r.Method, "name", name)

	if r.Method == http.MethodDelete {
		resp, err := pythonCall(r.Context(), map[string]interface{}{
			"command": "chroma_delete",
			"name":    name,
		})
		if err != nil {
			errorf("collectionHandler DELETE FAILED", "req_id", reqID, "err", err)
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		debugf("collectionHandler DELETE OK", "req_id", reqID, "elapsed", time.Since(start).String())
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
		return
	}

	http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
}

func collectionVectorsHandler(w http.ResponseWriter, r *http.Request) {
	reqID := atomic.AddUint64(&requestCounter, 1)
	start := time.Now()

	// Extract collection name from URL: /collections/{name}/vectors
	parts := strings.SplitN(r.URL.Path, "/", 5)
	name := ""
	if len(parts) >= 4 {
		name = parts[2]
	}

	debugf("collectionVectorsHandler START", "req_id", reqID, "method", r.Method, "name", name)

	if r.Method == http.MethodPost {
		// Add vectors
		var req ChromaAddRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			errorf("collectionVectorsHandler POST bad request", "req_id", reqID, "err", err)
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		resp, err := pythonCall(r.Context(), map[string]interface{}{
			"command":    "chroma_add",
			"name":       name,
			"ids":        req.IDs,
			"embeddings": req.Embeddings,
			"metadatas":  req.Metadatas,
		})
		if err != nil {
			errorf("collectionVectorsHandler POST FAILED", "req_id", reqID, "err", err)
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		debugf("collectionVectorsHandler POST OK", "req_id", reqID, "elapsed", time.Since(start).String())
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
		return
	}

	if r.Method == http.MethodDelete {
		// Delete vectors
		var req ChromaDeleteRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			errorf("collectionVectorsHandler DELETE bad request", "req_id", reqID, "err", err)
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		resp, err := pythonCall(r.Context(), map[string]interface{}{
			"command": "chroma_delete",
			"name":    name,
			"ids":     req.IDs,
		})
		if err != nil {
			errorf("collectionVectorsHandler DELETE FAILED", "req_id", reqID, "err", err)
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		debugf("collectionVectorsHandler DELETE OK", "req_id", reqID, "elapsed", time.Since(start).String())
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
		return
	}

	http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
}

func collectionQueryHandler(w http.ResponseWriter, r *http.Request) {
	reqID := atomic.AddUint64(&requestCounter, 1)
	start := time.Now()

	// Extract collection name from URL: /collections/{name}/query
	parts := strings.SplitN(r.URL.Path, "/", 5)
	name := ""
	if len(parts) >= 4 {
		name = parts[2]
	}

	debugf("collectionQueryHandler START", "req_id", reqID, "method", r.Method, "name", name)

	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req ChromaQueryRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		errorf("collectionQueryHandler bad request", "req_id", reqID, "err", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	nResults := req.NResults
	if nResults == 0 {
		nResults = 10
	}
	include := req.Include
	if include == nil {
		include = []string{"metadatas", "distances"}
	}

	resp, err := pythonCall(r.Context(), map[string]interface{}{
		"command":          "chroma_query",
		"name":             name,
		"query_embeddings": req.QueryEmbeddings,
		"n_results":       nResults,
		"include":          include,
	})
	if err != nil {
		errorf("collectionQueryHandler FAILED", "req_id", reqID, "err", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	debugf("collectionQueryHandler OK", "req_id", reqID, "elapsed", time.Since(start).String())
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func collectionResetHandler(w http.ResponseWriter, r *http.Request) {
	reqID := atomic.AddUint64(&requestCounter, 1)
	start := time.Now()

	// Extract collection name from URL: /collections/{name}/reset
	parts := strings.SplitN(r.URL.Path, "/", 5)
	name := ""
	if len(parts) >= 4 {
		name = parts[2]
	}

	debugf("collectionResetHandler START", "req_id", reqID, "method", r.Method, "name", name)

	if r.Method != http.MethodDelete {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	resp, err := pythonCall(r.Context(), map[string]interface{}{
		"command": "chroma_reset",
		"name":    name,
	})
	if err != nil {
		errorf("collectionResetHandler FAILED", "req_id", reqID, "err", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	debugf("collectionResetHandler OK", "req_id", reqID, "elapsed", time.Since(start).String())
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
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
	// Single goroutine serialises all crash restarts; guarantees Python is ready before requests proceed
	go restartGoroutine()
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

	// Chroma vector store routes
	mux.HandleFunc("/collections", withRecover(collectionsHandler))
	mux.HandleFunc("/collections/", func(w http.ResponseWriter, r *http.Request) {
		path := r.URL.Path
		if strings.HasSuffix(path, "/vectors") {
			collectionVectorsHandler(w, r)
		} else if strings.HasSuffix(path, "/query") {
			collectionQueryHandler(w, r)
		} else if strings.HasSuffix(path, "/reset") {
			collectionResetHandler(w, r)
		} else {
			collectionHandler(w, r)
		}
	})

	addr := fmt.Sprintf("0.0.0.0:%d", embedxPort)
	infof("embedx server starting", "addr", addr)

	server := &http.Server{
		Addr:         addr,
		Handler:      mux,
		ReadTimeout:  120 * time.Second,
		WriteTimeout: 120 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	go signalHandler(server)

	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		errorf("Server.ListenAndServe error", "err", err)
		log.Fatalf("Server error: %v", err)
	}
}
