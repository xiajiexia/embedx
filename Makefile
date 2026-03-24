.PHONY: build run test clean install uninstall

# Build the binary
build:
	go build -o embedx .

# Run locally (requires FastEmbed model)
run:
	./embedx

# Run with custom port
dev:
	EMBEDX_PORT=11450 EMBEDX_MODEL=BAAI/bge-small-zh-v1.5 ./embedx

# Test the API
test:
	curl -X POST http://localhost:11434/api/embeddings \
		-H "Content-Type: application/json" \
		-d '{"model": "BAAI/bge-small-zh-v1.5", "prompt": "你好世界"}'

# Health check
health:
	curl http://localhost:11434/health

# List models
models:
	curl http://localhost:11434/api/tags

# Clean build artifacts
clean:
	rm -f embedx

# Install to /opt/embedx
install: build
	sudo mkdir -p /opt/embedx
	sudo cp embedx /opt/embedx/
	sudo cp embed.py /opt/embedx/
	sudo chmod +x /opt/embedx/embedx

# Install systemd service
service-install:
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

[Install]
WantedBy=multi-user.target
EOF
	sudo systemctl daemon-reload
	sudo systemctl enable embedx

# Start the service
service-start:
	sudo systemctl start embedx

# Stop the service
service-stop:
	sudo systemctl stop embedx

# Service status
service-status:
	sudo systemctl status embedx

# View service logs
service-logs:
	journalctl -u embedx -f

# Uninstall
uninstall:
	sudo systemctl stop embedx
	sudo systemctl disable embedx
	sudo rm /etc/systemd/system/embedx.service
	sudo systemctl daemon-reload
	sudo rm -rf /opt/embedx
