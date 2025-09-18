#!/bin/bash

# Data Science Sandbox - Deployment Script

set -e

echo "🚀 Deploying Data Science Sandbox"
echo "================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Default deployment type
DEPLOYMENT_TYPE=${1:-local}

case $DEPLOYMENT_TYPE in
    "local")
        print_status "Deploying locally..."
        
        # Run tests first
        ./scripts/test.sh
        
        # Generate sample data
        print_status "Generating sample datasets..."
        python data/generate_datasets.py
        
        # Start the dashboard
        print_status "Starting dashboard..."
        python main.py --mode dashboard
        ;;
        
    "docker")
        print_status "Building Docker container..."
        
        if ! command -v docker &> /dev/null; then
            print_error "Docker is not installed"
            exit 1
        fi
        
        # Build Docker image
        docker build -t data-science-sandbox .
        
        # Run container
        print_status "Starting container..."
        docker run -p 8501:8501 -p 8888:8888 data-science-sandbox
        ;;
        
    "heroku")
        print_status "Deploying to Heroku..."
        
        if ! command -v heroku &> /dev/null; then
            print_error "Heroku CLI is not installed"
            exit 1
        fi
        
        # Login to Heroku (if needed)
        heroku auth:whoami || heroku login
        
        # Create Heroku app (if not exists)
        heroku create data-science-sandbox || true
        
        # Deploy
        git push heroku main
        
        # Open app
        heroku open
        ;;
        
    "aws")
        print_status "Deploying to AWS..."
        print_warning "AWS deployment requires additional configuration"
        print_status "Please refer to docs/deployment/aws.md for instructions"
        ;;
        
    *)
        print_error "Unknown deployment type: $DEPLOYMENT_TYPE"
        echo "Available types: local, docker, heroku, aws"
        exit 1
        ;;
esac

print_success "Deployment completed!"