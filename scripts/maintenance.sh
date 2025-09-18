#!/bin/bash

# Data Science Sandbox - Maintenance Script

set -e

echo "🔧 Data Science Sandbox Maintenance"
echo "===================================="

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

TASK=${1:-help}

case $TASK in
    "clean")
        print_status "Cleaning temporary files..."
        
        # Remove Python cache files
        find . -name "*.pyc" -delete
        find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
        
        # Remove test artifacts
        rm -rf .pytest_cache/ htmlcov/ .coverage coverage.xml
        
        # Remove build artifacts
        rm -rf build/ dist/ *.egg-info/
        
        # Remove logs (but keep directory)
        rm -f logs/*.log 2>/dev/null || true
        
        # Remove temporary files
        rm -rf temp/* 2>/dev/null || true
        
        print_success "Cleanup completed"
        ;;
        
    "update")
        print_status "Updating dependencies..."
        
        # Activate virtual environment
        if [ -f "venv/bin/activate" ]; then
            source venv/bin/activate
        fi
        
        # Update pip
        pip install --upgrade pip
        
        # Update all packages
        pip install --upgrade -r requirements.txt
        
        # Update development tools
        pip install --upgrade black isort flake8 mypy pytest pytest-cov
        
        print_success "Dependencies updated"
        ;;
        
    "backup")
        print_status "Creating backup..."
        
        BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$BACKUP_DIR"
        
        # Backup important files
        cp -r sandbox/ "$BACKUP_DIR/"
        cp -r data/datasets/ "$BACKUP_DIR/"
        cp -r notebooks/ "$BACKUP_DIR/"
        cp -r docs/ "$BACKUP_DIR/"
        cp *.py "$BACKUP_DIR/"
        cp *.txt "$BACKUP_DIR/"
        cp *.md "$BACKUP_DIR/"
        
        # Create archive
        tar -czf "backup_$(date +%Y%m%d_%H%M%S).tar.gz" -C backups/ "$(basename $BACKUP_DIR)"
        
        print_success "Backup created: backup_$(date +%Y%m%d_%H%M%S).tar.gz"
        ;;
        
    "reset")
        print_status "Resetting user data..."
        print_warning "This will remove all progress data!"
        read -p "Are you sure? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -f progress.json
            rm -rf data/user_data/ 2>/dev/null || true
            print_success "User data reset"
        else
            print_status "Reset cancelled"
        fi
        ;;
        
    "logs")
        print_status "Displaying recent logs..."
        
        if [ -d "logs" ] && [ "$(ls -A logs/)" ]; then
            for logfile in logs/*.log; do
                if [ -f "$logfile" ]; then
                    print_status "=== $(basename $logfile) ==="
                    tail -20 "$logfile"
                    echo
                fi
            done
        else
            print_status "No log files found"
        fi
        ;;
        
    "stats")
        print_status "System statistics..."
        
        # File counts
        echo "📁 Project structure:"
        echo "  Python files: $(find . -name '*.py' | wc -l)"
        echo "  Notebooks: $(find . -name '*.ipynb' | wc -l)"
        echo "  Markdown files: $(find . -name '*.md' | wc -l)"
        echo "  Datasets: $(find data/datasets/ -name '*.csv' 2>/dev/null | wc -l)"
        
        # Code metrics
        if command -v cloc &> /dev/null; then
            echo ""
            echo "📊 Code metrics:"
            cloc --exclude-dir=venv,node_modules,__pycache__,.git .
        fi
        
        # Git stats (if available)
        if [ -d ".git" ]; then
            echo ""
            echo "📈 Git statistics:"
            echo "  Total commits: $(git rev-list --count HEAD)"
            echo "  Contributors: $(git shortlog -sn | wc -l)"
            echo "  Files tracked: $(git ls-files | wc -l)"
        fi
        ;;
        
    "help"|*)
        print_status "Available maintenance tasks:"
        echo "  clean   - Remove temporary files and cache"
        echo "  update  - Update all dependencies"
        echo "  backup  - Create backup of important files"
        echo "  reset   - Reset user progress data"
        echo "  logs    - Display recent log files"
        echo "  stats   - Show project statistics"
        echo ""
        echo "Usage: ./scripts/maintenance.sh [task]"
        ;;
esac