.PHONY: build run test test-pull test-tags health clean install start stop restart restart-py logs uninstall deploy help

# Build the binary
build:
	@echo "🔨 Building embedx..."
	@go build -o embedx .
	@echo "✅ Build complete"

# Run locally
run:
	@echo "🚀 Starting embedx locally..."
	@./embedx

# Test the API
test:
	@echo "🧪 Testing embedding API..."
	@curl -s -X POST http://localhost:11434/api/embeddings \
		-H "Content-Type: application/json" \
		-d '{"model": "BAAI/bge-small-zh-v1.5", "prompt": "你好世界"}' | head -c 200
	@echo "..."

# Test pull model
test-pull:
	@echo "📥 Pulling model..."
	@curl -X POST http://localhost:11434/api/pull \
		-H "Content-Type: application/json" \
		-d '{"name": "BAAI/bge-base-en-v1.5"}'

# Test list models
test-tags:
	@echo "📋 Listing models..."
	@curl -s http://localhost:11434/api/tags | python3 -m json.tool 2>/dev/null || curl -s http://localhost:11434/api/tags

# Health check
health:
	@echo "🏥 Health check..."
	@curl -s http://localhost:11434/health && echo ""

# Clean build artifacts
clean:
	@echo "🧹 Cleaning..."
	@rm -f embedx
	@echo "✅ Clean complete"

# Install to /opt/embedx and enable systemd service
install: build
	@CURRENT_USER=$$(whoami) && \
	CURRENT_GROUP=$$(id -gn) && \
	DEPLOY_PATH="/opt/embedx" && \
	echo "📦 Installing to $$DEPLOY_PATH (user: $$CURRENT_USER)..." && \
	sed "s/{{USER}}/$$CURRENT_USER/g; s/{{GROUP}}/$$CURRENT_GROUP/g; s|{{DEPLOY_PATH}}|$$DEPLOY_PATH|g" \
		deploy/embedx.service | sudo tee /etc/systemd/system/embedx.service > /dev/null && \
	sudo mkdir -p $$DEPLOY_PATH/models && \
	sudo chown -R $$CURRENT_USER:$$CURRENT_GROUP $$DEPLOY_PATH && \
	sudo cp embedx $$DEPLOY_PATH/ && \
	sudo cp embed.py $$DEPLOY_PATH/ && \
	sudo chmod +x $$DEPLOY_PATH/embedx && \
	sudo systemctl daemon-reload && \
	sudo systemctl enable embedx
	@echo "✅ Installed to $$DEPLOY_PATH"

# Start the service
start:
	@echo "▶️  Starting embedx service..."
	@sudo systemctl start embedx
	@sleep 1
	@systemctl is-active embedx && echo "✅ Service started" || echo "❌ Start failed"

# Stop the service
stop:
	@echo "⏹️  Stopping embedx service..."
	@sudo systemctl stop embedx
	@echo "✅ Service stopped"

# Restart the service
restart:
	@echo "🔄 Restarting embedx service..."
	@sudo systemctl restart embedx
	@sleep 1
	@systemctl is-active embedx && echo "✅ Service restarted" || echo "❌ Restart failed"

# Restart only the Python backend (hot reload model without restarting Go server)
restart-py:
	@echo "🔄 Restarting Python backend (model hot-reload)..."
	@sudo kill -SIGUSR1 $$(systemctl show --property MainPID --value embedx) 2>/dev/null || echo "SIGUSR1 not supported, use restart instead"
	@sleep 1
	@systemctl is-active embedx && echo "✅ Python backend restarted" || echo "❌ Restart failed"

# View logs
logs:
	@sudo journalctl -u embedx -f

# View recent logs
logs-recent:
	@sudo journalctl -u embedx -n 50 --no-pager

# Uninstall
uninstall:
	@echo "🗑️  Uninstalling embedx service..."
	@sudo systemctl stop embedx 2>/dev/null || true
	@sudo systemctl disable embedx 2>/dev/null || true
	@sudo rm -f /etc/systemd/system/embedx.service
	@sudo systemctl daemon-reload
	@echo "✅ Service uninstalled"
	@echo "⚠️  Note: /opt/embedx not removed (contains models). Remove manually if needed:"
	@echo "    sudo rm -rf /opt/embedx"

# Deploy: build -> stop -> deploy -> restart
deploy: build
	@CURRENT_USER=$$(whoami) && \
	CURRENT_GROUP=$$(id -gn) && \
	DEPLOY_PATH="/opt/embedx" && \
	echo "📦 Deploying to $$DEPLOY_PATH (user: $$CURRENT_USER)..." && \
	sed "s/{{USER}}/$$CURRENT_USER/g; s/{{GROUP}}/$$CURRENT_GROUP/g; s|{{DEPLOY_PATH}}|$$DEPLOY_PATH|g" \
		deploy/embedx.service | sudo tee /etc/systemd/system/embedx.service > /dev/null && \
	echo "⏹️  Stopping old service..." && \
	sudo systemctl stop embedx 2>/dev/null || true && \
	sudo mkdir -p $$DEPLOY_PATH && \
	sudo mkdir -p $$DEPLOY_PATH/models && \
	sudo chown -R $$CURRENT_USER:$$CURRENT_GROUP $$DEPLOY_PATH && \
	sudo cp embedx $$DEPLOY_PATH/ && \
	sudo cp embed.py $$DEPLOY_PATH/ && \
	sudo chmod +x $$DEPLOY_PATH/embedx && \
	sudo systemctl daemon-reload && \
	echo "🔄 Restarting service..." && \
	sudo systemctl restart embedx && \
	sleep 2 && \
	systemctl is-active embedx && echo "✅ Deploy complete" || echo "❌ Start failed"

# Help
help:
	@echo "embedx - FastEmbed with Ollama-compatible API"
	@echo ""
	@echo "  make build        Build the binary"
	@echo "  make run          Run locally (foreground)"
	@echo "  make test         Test embedding API"
	@echo "  make test-pull    Pull a model"
	@echo "  make test-tags    List available models"
	@echo "  make health       Health check"
	@echo "  make install      Install to /opt/embedx + enable systemd"
	@echo "  make deploy       Build + stop + deploy + restart (one command)"
	@echo "  make start        Start systemd service"
	@echo "  make stop         Stop systemd service"
	@echo "  make restart      Restart systemd service"
	@echo "  make restart-py   Hot-reload Python backend (model swap)"
	@echo "  make logs         Tail service logs (-f)"
	@echo "  make logs-recent  Show recent logs"
	@echo "  make uninstall    Stop and disable systemd service"
	@echo "  make clean        Remove build artifacts"
	@echo ""
	@echo "Deploy path: /opt/embedx (fixed)"
	@echo "Port: 11434"
	@echo "Default model: BAAI/bge-small-zh-v1.5"
