#!/bin/bash
#
# Development Environment Setup Script
# Sets up the enhanced gamification dashboard for development
#

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
COMPOSE_FILE="docker-compose.dev.yml"

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

# Setup development environment
setup_dev() {
    log "Setting up development environment..."

    # Create necessary directories
    mkdir -p data/datasets logs mlruns mlartifacts model_registry results notebooks

    # Build and start development containers
    log "Building development Docker images..."
    docker-compose -f $COMPOSE_FILE build

    log "Starting development environment..."
    docker-compose -f $COMPOSE_FILE up -d

    success "Development environment started!"
}

# Show development status
show_dev_status() {
    log "Development Environment Status:"
    echo
    docker-compose -f $COMPOSE_FILE ps
    echo

    log "Development URLs:"
    echo "🎮 Enhanced Gamification Dashboard: http://localhost:8503"
    echo "📊 Standard Dashboard: http://localhost:8501"
    echo "📱 Modern Dashboard: http://localhost:8502"
    echo "📓 Jupyter Lab: http://localhost:8888 (token: dev-token-12345)"
    echo "🔬 MLflow: http://localhost:5000"
    echo "🗄️  PostgreSQL: localhost:5432 (user: sandbox, password: sandbox123)"
    echo "📦 Redis: localhost:6379"
    echo

    log "Hot reload is enabled - changes will be reflected automatically!"
}

# Main function
main() {
    case "${1:-start}" in
        "start"|"up")
            setup_dev
            show_dev_status
            ;;
        "stop"|"down")
            log "Stopping development environment..."
            docker-compose -f $COMPOSE_FILE down
            success "Development environment stopped"
            ;;
        "restart")
            log "Restarting development environment..."
            docker-compose -f $COMPOSE_FILE restart
            success "Development environment restarted"
            ;;
        "logs")
            docker-compose -f $COMPOSE_FILE logs -f "${2:-}"
            ;;
        "status")
            show_dev_status
            ;;
        "shell")
            log "Opening shell in development container..."
            docker-compose -f $COMPOSE_FILE exec sandbox-dev bash
            ;;
        "test")
            log "Running tests in development environment..."
            docker-compose -f $COMPOSE_FILE exec sandbox-dev python -m pytest tests/ -v
            ;;
        *)
            echo "Usage: $0 {start|stop|restart|logs|status|shell|test}"
            echo
            echo "Commands:"
            echo "  start   - Start development environment (default)"
            echo "  stop    - Stop development environment"
            echo "  restart - Restart development environment"
            echo "  logs    - Show logs (optionally specify service)"
            echo "  status  - Show environment status"
            echo "  shell   - Open shell in development container"
            echo "  test    - Run tests"
            exit 1
            ;;
    esac
}

log "🛠️ Data Science Sandbox - Development Environment"
echo
main "$@"
