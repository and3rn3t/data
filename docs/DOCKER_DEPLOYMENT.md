# Docker Deployment Guide for Data Science Sandbox

## Overview

This document provides comprehensive instructions for deploying the Data Science Sandbox with enhanced gamification features using Docker in both development and production environments.

## 🏗️ Architecture

The application consists of multiple services:

### Production Services

- **Enhanced Gamification Dashboard** (Port 8503) - Main production interface with achievements, badges, and analytics
- **Standard Dashboard** (Port 8501) - Classic Streamlit interface
- **Modern Dashboard** (Port 8502) - iOS-style modern interface
- **MLflow Tracking Server** (Port 5000) - Model and experiment tracking
- **Nginx Reverse Proxy** (Port 80) - Load balancing and routing

### Development Services

- **Development Container** - Hot-reload enabled development environment
- **Jupyter Lab** (Port 8888) - Interactive notebook environment
- **PostgreSQL** (Port 5432) - Development database
- **Redis** (Port 6379) - Caching and session storage

## 🚀 Quick Start

### Production Deployment

#### Linux/macOS

```bash
# Make scripts executable
chmod +x scripts/deploy-prod.sh

# Deploy to production
./scripts/deploy-prod.sh

# Check status
./scripts/deploy-prod.sh status

# View logs
./scripts/deploy-prod.sh logs

# Stop deployment
./scripts/deploy-prod.sh stop
```

#### Windows

```cmd
# Deploy to production
scripts\deploy-prod.bat

# Check status
scripts\deploy-prod.bat status

# View logs
scripts\deploy-prod.bat logs

# Stop deployment
scripts\deploy-prod.bat stop
```

### Development Environment

#### Linux/macOS

```bash
# Make scripts executable
chmod +x scripts/dev-setup.sh

# Start development environment
./scripts/dev-setup.sh start

# Open shell in container
./scripts/dev-setup.sh shell

# Run tests
./scripts/dev-setup.sh test

# Stop development environment
./scripts/dev-setup.sh stop
```

#### Manual Docker Commands

```bash
# Development
docker-compose -f docker-compose.dev.yml up -d

# Production
docker-compose -f docker-compose.prod.yml up -d
```

## 🔧 Configuration

### Environment Variables

#### Production (.env.prod)

```bash
# Application
NODE_ENV=production
PYTHONPATH=/app

# Streamlit
STREAMLIT_SERVER_PORT=8503
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
STREAMLIT_SERVER_ENABLE_CORS=false
STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true

# MLflow
MLFLOW_TRACKING_URI=file:///app/mlruns
MLFLOW_ARTIFACT_ROOT=/app/mlartifacts

# Weights & Biases
WANDB_MODE=offline

# Security
STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true
```

#### Development (.env.dev)

```bash
# Application
NODE_ENV=development
PYTHONPATH=/app

# Streamlit Development
STREAMLIT_SERVER_PORT=8503
STREAMLIT_SERVER_RUN_ON_SAVE=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Database
POSTGRES_DB=sandbox_db
POSTGRES_USER=sandbox
POSTGRES_PASSWORD=sandbox123

# Jupyter
JUPYTER_ENABLE_LAB=yes
JUPYTER_TOKEN=dev-token-12345
```

### Volume Mounts

#### Production Volumes

- `./data:/app/data:ro` - Read-only data access
- `./logs:/app/logs` - Application logs
- `./mlruns:/app/mlruns` - MLflow experiment tracking
- `./mlartifacts:/app/mlartifacts` - MLflow artifacts
- `./model_registry:/app/model_registry` - Model registry
- `./results:/app/results` - Analysis results

#### Development Volumes

- `.:/app` - Full hot-reload access
- `./data:/app/data` - Read-write data access
- `./notebooks:/app/notebooks` - Jupyter notebooks

## 🌐 Service Access

### Production URLs

| Service                   | URL                   | Description                                 |
| ------------------------- | --------------------- | ------------------------------------------- |
| **Enhanced Gamification** | http://localhost:8503 | Main production dashboard with achievements |
| **Standard Dashboard**    | http://localhost:8501 | Classic Streamlit interface                 |
| **Modern Dashboard**      | http://localhost:8502 | iOS-style modern interface                  |
| **MLflow Tracking**       | http://localhost:5000 | Experiment and model tracking               |
| **Nginx Proxy**           | http://localhost      | Load balancer and routing                   |

### Nginx Routing

| Domain               | Target                | Description       |
| -------------------- | --------------------- | ----------------- |
| `localhost`          | Enhanced Gamification | Default route     |
| `standard.localhost` | Standard Dashboard    | Classic interface |
| `modern.localhost`   | Modern Dashboard      | Modern interface  |
| `mlflow.localhost`   | MLflow Server         | Model tracking    |

### Development URLs

| Service                   | URL                   | Credentials                             |
| ------------------------- | --------------------- | --------------------------------------- |
| **Enhanced Gamification** | http://localhost:8503 | -                                       |
| **Jupyter Lab**           | http://localhost:8888 | Token: `dev-token-12345`                |
| **PostgreSQL**            | localhost:5432        | User: `sandbox`, Password: `sandbox123` |
| **Redis**                 | localhost:6379        | -                                       |
| **MLflow**                | http://localhost:5000 | -                                       |

## 📊 Monitoring and Logging

### Health Checks

The application includes comprehensive health checks:

```bash
# Check enhanced gamification dashboard
curl -f http://localhost:8503/_stcore/health

# Check MLflow server
curl -f http://localhost:5000/health

# Check nginx proxy
curl -f http://localhost/health
```

### Log Access

```bash
# All services
docker-compose -f docker-compose.prod.yml logs -f

# Specific service
docker-compose -f docker-compose.prod.yml logs -f gamification-dashboard

# Follow logs with timestamps
docker-compose -f docker-compose.prod.yml logs -f -t
```

### Container Status

```bash
# Check running containers
docker-compose -f docker-compose.prod.yml ps

# Resource usage
docker stats

# Container inspection
docker inspect ds-sandbox-gamification
```

## 🔒 Security Features

### Production Security

1. **Non-root User**: Application runs as `appuser`
2. **XSRF Protection**: Enabled for Streamlit
3. **CORS Policy**: Restricted cross-origin requests
4. **Rate Limiting**: Nginx rate limiting (10 req/s)
5. **Security Headers**: X-Frame-Options, X-Content-Type-Options, etc.
6. **Read-only Mounts**: Data mounted as read-only in production

### SSL/TLS Configuration

For production deployment with SSL:

1. Place SSL certificates in `docker/ssl/`
2. Update `docker/nginx.conf` with SSL configuration
3. Update port mappings to include 443

```nginx
server {
    listen 443 ssl;
    ssl_certificate /etc/ssl/certs/cert.pem;
    ssl_certificate_key /etc/ssl/certs/key.pem;
    # ... rest of configuration
}
```

## 🚨 Troubleshooting

### Common Issues

#### Container Won't Start

```bash
# Check container logs
docker-compose -f docker-compose.prod.yml logs gamification-dashboard

# Check Docker daemon
sudo systemctl status docker

# Rebuild without cache
docker-compose -f docker-compose.prod.yml build --no-cache
```

#### Port Conflicts

```bash
# Check what's using the port
sudo lsof -i :8503  # Linux/macOS
netstat -ano | findstr :8503  # Windows

# Change port in docker-compose.yml if needed
```

#### Permission Issues

```bash
# Fix data directory permissions
sudo chown -R $USER:$USER ./data ./logs ./mlruns

# For development on Linux
sudo chown -R 1000:1000 ./data ./logs
```

#### Memory Issues

```bash
# Check Docker resources
docker system df

# Clean up unused containers/images
docker system prune -a

# Increase Docker memory limit in Docker Desktop
```

### Performance Optimization

#### Production Optimizations

1. **Resource Limits**: Add resource limits to docker-compose.yml

```yaml
services:
  gamification-dashboard:
    deploy:
      resources:
        limits:
          cpus: "0.50"
          memory: 512M
        reservations:
          memory: 256M
```

2. **Nginx Caching**: Enable caching for static assets
3. **Database Connection Pooling**: For production databases
4. **Container Health Monitoring**: Use tools like Prometheus

## 📦 Backup and Recovery

### Backup Strategy

```bash
# Create backup using script
./scripts/deploy-prod.sh backup

# Manual backup
docker run --rm -v $(pwd):/backup -v ds-data:/data alpine tar czf /backup/data-backup-$(date +%Y%m%d).tar.gz -C /data .
```

### Recovery Process

```bash
# Stop services
docker-compose -f docker-compose.prod.yml down

# Restore from backup
tar xzf data-backup-20241003.tar.gz -C ./data/

# Restart services
docker-compose -f docker-compose.prod.yml up -d
```

## 🔄 Deployment Pipeline

### CI/CD Integration

Example GitHub Actions workflow:

```yaml
name: Deploy to Production
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to production
        run: |
          ./scripts/deploy-prod.sh
          ./scripts/deploy-prod.sh status
```

### Rolling Updates

```bash
# Update specific service
docker-compose -f docker-compose.prod.yml up -d --no-deps gamification-dashboard

# Scale services
docker-compose -f docker-compose.prod.yml up -d --scale gamification-dashboard=2
```

## 📈 Scaling Considerations

### Horizontal Scaling

For high-traffic deployments:

1. **Load Balancer**: Use external load balancer (AWS ALB, nginx+)
2. **Database**: External database (PostgreSQL, MySQL)
3. **Storage**: Shared storage (AWS S3, NFS)
4. **Container Orchestration**: Kubernetes, Docker Swarm

### Kubernetes Deployment

See `k8s/` directory for Kubernetes manifests (if available).

## 🆘 Support

### Getting Help

1. **Logs**: Always check logs first
2. **Health Checks**: Verify service health endpoints
3. **Resource Usage**: Check memory and CPU usage
4. **Network**: Verify port accessibility

### Useful Commands

```bash
# Interactive debugging
docker-compose -f docker-compose.prod.yml exec gamification-dashboard bash

# Real-time logs with grep
docker-compose -f docker-compose.prod.yml logs -f | grep ERROR

# Container resource usage
docker stats --no-stream

# Network debugging
docker network ls
docker network inspect ds-sandbox-network
```
