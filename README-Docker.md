# RAG Chunking Service - Docker Setup

This document explains how to run the RAG Chunking Service using Docker containers with Redis for production-ready concurrent user support.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Nginx Proxy   │    │  RAG API        │    │     Redis       │
│   (Production)  │────│  FastAPI + uv   │────│  Progress Store │
│   Port: 80/443  │    │  Port: 8000     │    │  Port: 6379     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### 1. Start in Development Mode

```bash
./docker-start.sh start dev
```

This includes Redis Commander UI at http://localhost:8081 (admin/secret)

### 2. Start in Production Mode

```bash
./docker-start.sh start prod
```

This includes Nginx reverse proxy with rate limiting

### 3. Check Service Health

```bash
./docker-start.sh health
```

## 📋 Available Commands

```bash
./docker-start.sh {start|stop|restart|logs|health|cleanup|status} [dev|prod]
```

- **start [dev|prod]** - Start services (dev includes Redis commander)
- **stop** - Stop all services
- **restart [dev|prod]** - Restart services
- **logs [prod]** - Show service logs
- **health** - Check service health
- **cleanup** - Clean up Docker resources
- **status** - Show service status

## 🔧 Configuration

### Environment Variables (.env.docker)

```bash
REDIS_URL=redis://redis:6379
MAX_CONCURRENT_TASKS=5      # Maximum parallel processing tasks
PROCESSING_QUEUE_SIZE=10    # Queue size for waiting tasks
MAX_FILE_SIZE=104857600     # 100MB file limit
MAX_PARALLEL_WORKERS=4      # Parallel processing workers
LOG_LEVEL=INFO
```

### Service Limits

- **Concurrent Users**: 5 (configurable)
- **Queue Size**: 10 waiting tasks
- **File Size**: 100MB maximum
- **Memory**: 2GB limit per container
- **CPU**: 1.0 core limit

## 🌐 Endpoints

- **API**: http://localhost:8000
- **Health Check**: http://localhost:8000/health
- **WebSocket Progress**: ws://localhost:8000/v1/operations/progress/{task_id}
- **Redis Commander** (dev only): http://localhost:8081 (admin/secret)

## 📊 Production Features

### Redis Progress Storage

- Persistent progress tracking across container restarts
- Automatic TTL (1 hour) for progress data
- Fallback to in-memory if Redis unavailable

### Queue Management

- Maximum 5 concurrent processing tasks
- Queue size of 10 for waiting requests
- Graceful handling of resource limits

### Rate Limiting (Production)

- 10 requests per second per IP
- Burst capacity of 20 requests
- WebSocket connections supported

### Health Monitoring

```bash
# Check API health
curl http://localhost:8000/health

# Check Redis health
docker exec rag-redis redis-cli ping
```

## 🐳 Docker Services

### Development Mode

- `redis` - Redis 7 with persistence
- `rag-api` - FastAPI application
- `redis-commander` - Redis management UI

### Production Mode

- `redis` - Redis 7 with optimized settings
- `rag-api` - FastAPI with resource limits
- `nginx` - Reverse proxy with rate limiting

## 📈 Scaling

### Horizontal Scaling

```bash
# Scale API service to 3 instances
docker-compose up --scale rag-api=3 -d
```

### Vertical Scaling

Update resource limits in `docker-compose.prod.yml`:

```yaml
deploy:
  resources:
    limits:
      memory: 4G # Increase memory
      cpus: "2.0" # Increase CPU
```

## 🔍 Monitoring

### View Logs

```bash
# All services
./docker-start.sh logs

# Specific service
docker-compose logs -f rag-api
```

### Resource Usage

```bash
# Container stats
docker stats

# Service status
docker-compose ps
```

## 🛠️ Troubleshooting

### Redis Connection Issues

```bash
# Check Redis logs
docker-compose logs redis

# Test Redis connection
docker exec rag-redis redis-cli ping
```

### API Performance Issues

```bash
# Check API logs
docker-compose logs rag-api

# Monitor resource usage
docker stats rag-chunking-api
```

### Redis Commander Issues

```bash
# Check if Redis Commander is running (dev mode only)
docker compose ps redis-commander

# Check Redis Commander logs
docker compose logs redis-commander

# Access Redis Commander
# URL: http://localhost:8081
# Username: admin
# Password: secret
```

### Queue Management

- **Queue Full**: Returns HTTP 429 (Too Many Requests)
- **Processing Limit**: Users wait in queue automatically
- **Failed Tasks**: Automatically removed from queue

## 🔐 Security

### Production Security Features

- Non-root container user
- Resource limits enforcement
- Rate limiting and request size limits
- Security headers via Nginx
- No exposed internal ports

### SSL Configuration (Production)

1. Place SSL certificates in `./ssl/` directory
2. Update `nginx.conf` for HTTPS
3. Restart services: `./docker-start.sh restart prod`

## 📝 Testing

### Load Testing

```bash
# Install testing tools
pip install httpx asyncio

# Run concurrent tests
python test_concurrent_users.py
```

### API Testing

```bash
# Test chunking endpoint
curl -X POST "http://localhost:8000/v1/operations/structural-block" \
  -H "Content-Type: application/json" \
  -d '{"text": "Test document for chunking", "chunk_size": 100}'

# Test WebSocket progress
# Use test_streaming.py or frontend WebSocket client
```

## 🚦 Service Status

### Healthy Services

```bash
# Expected output
✅ API is healthy
✅ Redis is healthy
```

### Unhealthy Services

Check logs and restart if needed:

```bash
./docker-start.sh logs
./docker-start.sh restart
```

## 📦 Deployment

### Development

```bash
git clone <repository>
cd rag-simulator
./docker-start.sh start dev
```

### Production

```bash
git clone <repository>
cd rag-simulator
cp .env.docker .env.production  # Customize as needed
./docker-start.sh start prod
```

The system is now production-ready for multiple concurrent users with Redis-backed progress storage and intelligent queue management! 🎉
