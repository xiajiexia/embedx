.PHONY: build run test clean install start stop uninstall

# Build the binary
build:
	go build -o embedx .

# Run locally
run:
	./embedx

# Test the API
test:
	curl -X POST http://localhost:11434/api/embeddings \
		-H "Content-Type: application/json" \
		-d '{"model": "BAAI/bge-small-zh-v1.5", "prompt": "你好世界"}'

# Test pull
test-pull:
	curl -X POST http://localhost:11434/api/pull \
		-H "Content-Type: application/json" \
		-d '{"name": "BAAI/bge-base-en-v1.5"}'

# Test list models
test-tags:
	curl http://localhost:11434/api/tags

# Health check
health:
	curl http://localhost:11434/health

# Clean build artifacts
clean:
	rm -f embedx

# Install to /opt/embedx
install: build
	sudo mkdir -p /opt/embedx
	sudo mkdir -p /opt/embedx/models
	sudo chown -R $$USER /opt/embedx
	sudo cp embedx /opt/embedx/
	sudo cp embed.py /opt/embedx/
	sudo chmod +x /opt/embedx/embedx

	sudo tee /etc/systemd/system/embedx.service > /dev/null <<EOF
[Unit]
Description=embedx - FastEmbed with Ollama-compatible API
After=network.target

[Service]
Type=simple
User=$$USER
WorkingDirectory=/opt/embedx
ExecStart=/opt/embedx/embedx
Restart=always
RestartSec=5
Environment=EMBEDX_PORT=11434
Environment=EMBEDX_MODEL=BAAI/bge-small-zh-v1.5
Environment=HTTP_PROXY=http://127.0.0.1:1087
Environment=HTTPS_PROXY=http://127.0.0.1:1087
Environment=NO_PROXY=localhost,127.0.0.1
Environment=HF_HUB_CACHE=/opt/embedx/models
Environment=TRANSFORMERS_CACHE=/opt/embedx/models

[Install]
WantedBy=multi-user.target
EOF
	sudo systemctl daemon-reload
	sudo systemctl enable embedx

# Start the service
start:
	sudo systemctl start embedx

# Stop the service
stop:
	sudo systemctl stop embedx

# View logs
logs:
	sudo journalctl -u embedx -f

# Uninstall
uninstall:
	sudo systemctl stop embedx || true
	sudo systemctl disable embedx || true
	sudo rm -f /etc/systemd/system/embedx.service
	sudo systemctl daemon-reload
	sudo rm -rf /opt/embedx
