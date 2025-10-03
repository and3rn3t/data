#!/bin/bash
#
# Production Deployment Script for Data Science Sandbox
# Deploys the enhanced gamification dashboard with all components
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="data-science-sandbox"
COMPOSE_FILE="docker-compose.prod.yml"
BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"

# Functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."

    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        error "Docker Compose is not installed"
        exit 1
    fi

    success "Prerequisites check passed"
}

# Backup existing data
backup_data() {
    if [ -d "./data" ] || [ -d "./mlruns" ] || [ -d "./logs" ]; then
        log "Creating backup..."
        mkdir -p "$BACKUP_DIR"

        if [ -d "./data" ]; then
            cp -r ./data "$BACKUP_DIR/" 2>/dev/null || true
        fi

        if [ -d "./mlruns" ]; then
            cp -r ./mlruns "$BACKUP_DIR/" 2>/dev/null || true
        fi

        if [ -d "./logs" ]; then
            cp -r ./logs "$BACKUP_DIR/" 2>/dev/null || true
        fi

        success "Backup created at $BACKUP_DIR"
    else
        log "No existing data to backup"
    fi
}

# Build and deploy
deploy() {
    log "Building Docker images..."
    docker-compose -f $COMPOSE_FILE build --no-cache

    log "Stopping existing containers..."
    docker-compose -f $COMPOSE_FILE down --remove-orphans

    log "Starting production deployment..."
    docker-compose -f $COMPOSE_FILE up -d

    success "Deployment started!"
}

# Health check
health_check() {
    log "Performing health checks..."

    local max_attempts=30
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        log "Health check attempt $attempt/$max_attempts"

        # Check enhanced gamification dashboard
        if curl -f -s http://localhost:8503/_stcore/health > /dev/null 2>&1; then
            success "Enhanced gamification dashboard is healthy"
            break
        fi

        if [ $attempt -eq $max_attempts ]; then
            error "Health check failed after $max_attempts attempts"
            docker-compose -f $COMPOSE_FILE logs gamification-dashboard
            return 1
        fi

        sleep 10
        ((attempt++))
    done

    # Additional service checks
    sleep 5

    if curl -f -s http://localhost:5000/health > /dev/null 2>&1; then
        success "MLflow tracking server is healthy"
    else
        warning "MLflow tracking server may not be ready yet"
    fi

    success "Health checks completed"
}

# Show deployment status
show_status() {
    log "Deployment Status:"
    echo
    docker-compose -f $COMPOSE_FILE ps
    echo

    log "Service URLs:"
    echo "🎮 Enhanced Gamification Dashboard: http://localhost:8503"
    echo "📊 Standard Dashboard: http://localhost:8501"
    echo "📱 Modern Dashboard: http://localhost:8502"
    echo "🔬 MLflow Tracking: http://localhost:5000"
    echo "🌐 Nginx Proxy: http://localhost"
    echo

    log "Service Access via Nginx:"
    echo "🎮 Default (Gamification): http://localhost/"
    echo "📊 Standard: http://standard.localhost/"
    echo "📱 Modern: http://modern.localhost/"
    echo "🔬 MLflow: http://mlflow.localhost/"
}

# Main deployment flow
main() {
    log "🚀 Starting production deployment of Data Science Sandbox"
    echo

    check_prerequisites
    backup_data
    deploy
    health_check
    show_status

    echo
    success "🎉 Production deployment completed successfully!"
    echo
    log "Monitor logs with: docker-compose -f $COMPOSE_FILE logs -f"
    log "Stop deployment with: docker-compose -f $COMPOSE_FILE down"
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "stop")
        log "Stopping production deployment..."
        docker-compose -f $COMPOSE_FILE down
        success "Deployment stopped"
        ;;
    "restart")
        log "Restarting production deployment..."
        docker-compose -f $COMPOSE_FILE restart
        success "Deployment restarted"
        ;;
    "logs")
        docker-compose -f $COMPOSE_FILE logs -f "${2:-}"
        ;;
    "status")
        show_status
        ;;
    "backup")
        backup_data
        ;;
    *)
        echo "Usage: $0 {deploy|stop|restart|logs|status|backup}"
        echo
        echo "Commands:"
        echo "  deploy  - Deploy the application (default)"
        echo "  stop    - Stop the deployment"
        echo "  restart - Restart the deployment"
        echo "  logs    - Show logs (optionally specify service name)"
        echo "  status  - Show deployment status"
        echo "  backup  - Create backup of data"
        exit 1
        ;;
esac
