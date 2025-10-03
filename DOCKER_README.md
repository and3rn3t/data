# 🐳 Docker Quick Start Guide

## Enhanced Gamification Dashboard - Production Ready

This guide helps you quickly deploy the Data Science Sandbox with enhanced gamification features using Docker.

## 🚀 Quick Deploy (Windows)

### Option 1: Windows Batch Script (Recommended)

```cmd
# Deploy to production
scripts\deploy-prod.bat

# Check status
scripts\deploy-prod.bat status

# Access the enhanced dashboard
# Open: http://localhost:8503
```

### Option 2: Docker Compose Manual

```cmd
# Build and deploy
docker compose -f docker-compose.prod.yml up -d

# Check running services
docker compose -f docker-compose.prod.yml ps

# View logs
docker compose -f docker-compose.prod.yml logs -f
```

## 🎮 What You Get

After deployment, you'll have access to:

- **🎮 Enhanced Gamification Dashboard**: <http://localhost:8503>

  - Achievement system with 19+ badges
  - Auto-validation of code challenges
  - Learning analytics and personalized recommendations
  - Progress tracking and skill radar

- **📊 Standard Dashboard**: <http://localhost:8501>
- **📱 Modern Dashboard**: <http://localhost:8502>
- **🔬 MLflow Tracking**: <http://localhost:5000>

## 🛠️ Development Environment

For development with hot-reload:

```cmd
# Start development environment
docker compose -f docker-compose.dev.yml up -d

# Access enhanced gamification dashboard
# Open: http://localhost:8503 (with hot reload)

# Access Jupyter Lab
# Open: http://localhost:8888 (token: dev-token-12345)
```

## 🔧 Key Features

### ✅ Production Ready

- **Security**: Non-root user, XSRF protection, rate limiting
- **Performance**: Nginx reverse proxy, optimized Python dependencies
- **Monitoring**: Health checks, comprehensive logging
- **Scalability**: Multi-service architecture

### ✅ Enhanced Gamification

- **19+ Achievement Badges** across 4 categories
- **Auto-validation System** with secure code execution
- **Learning Analytics** with style detection
- **Progressive Hint System** for challenges
- **Personalized Recommendations** engine

### ✅ Development Features

- **Hot Reload** for rapid development
- **Jupyter Lab** integration
- **PostgreSQL & Redis** for development
- **Comprehensive Testing** environment

## 🔄 Common Commands

```cmd
# Production deployment
scripts\deploy-prod.bat              # Deploy
scripts\deploy-prod.bat status       # Check status
scripts\deploy-prod.bat logs         # View logs
scripts\deploy-prod.bat stop         # Stop all services
scripts\deploy-prod.bat backup       # Create backup

# Development
docker compose -f docker-compose.dev.yml up -d     # Start dev
docker compose -f docker-compose.dev.yml down      # Stop dev
docker compose -f docker-compose.dev.yml logs -f   # Follow logs
```

## 🚨 Troubleshooting

### Port Conflicts

If ports are already in use, modify the ports in `docker-compose.prod.yml`:

```yaml
ports:
  - "8504:8503" # Change external port
```

### Container Won't Start

```cmd
# Check logs
docker compose -f docker-compose.prod.yml logs gamification-dashboard

# Rebuild without cache
docker compose -f docker-compose.prod.yml build --no-cache
```

### Access Issues

- Ensure Docker Desktop is running
- Check Windows Firewall/antivirus settings
- Try accessing via `http://127.0.0.1:8503` instead of `localhost`

## 📋 System Requirements

- **Docker Desktop**: Version 4.0+
- **RAM**: Minimum 4GB, Recommended 8GB+
- **Storage**: 5GB free space for images and data
- **OS**: Windows 10/11, macOS, or Linux

## 🔗 For More Details

See the complete deployment guide: [docs/DOCKER_DEPLOYMENT.md](docs/DOCKER_DEPLOYMENT.md)

## 🎯 Next Steps

1. **Deploy**: Run the deployment script
2. **Explore**: Visit <http://localhost:8503> for the enhanced dashboard
3. **Learn**: Try the gamified learning challenges
4. **Develop**: Use the development environment for customization

---

**🎉 Ready to experience gamified data science learning with Docker!**
