#!/bin/bash

set -e

echo "🚀 Starting RAG Chunking Service with Docker..."

# Function to check if Redis is running
check_redis() {
    echo "🔍 Checking if Redis is running..."
    if docker run --rm --network rag-network redis:7-alpine redis-cli -h redis ping > /dev/null 2>&1; then
        echo "✅ Redis is running"
        return 0
    else
        echo "❌ Redis is not running"
        return 1
    fi
}

# Function to start services
start_services() {
    echo "🐳 Building and starting Docker services..."
    
    # Development mode (with Redis commander)
    if [ "$1" = "dev" ]; then
        echo "🔧 Starting in development mode..."
        COMPOSE_PROFILES=debug docker compose up --build -d
    # Production mode
    elif [ "$1" = "prod" ]; then
        echo "🚀 Starting in production mode..."
        docker compose -f docker-compose.prod.yml up --build -d
    else
        echo "🔧 Starting in standard mode..."
        docker compose up --build -d
    fi
}

# Function to show logs
show_logs() {
    echo "📋 Showing service logs..."
    if [ "$1" = "prod" ]; then
        docker compose -f docker-compose.prod.yml logs -f
    else
        docker compose logs -f
    fi
}

# Function to stop services
stop_services() {
    echo "🛑 Stopping services..."
    if [ -f "docker-compose.prod.yml" ] && docker compose -f docker-compose.prod.yml ps -q > /dev/null 2>&1; then
        docker compose -f docker-compose.prod.yml down
    else
        docker compose down
    fi
}

# Function to restart services
restart_services() {
    echo "🔄 Restarting services..."
    stop_services
    start_services $1
}

# Function to check service health
check_health() {
    echo "🏥 Checking service health..."
    
    # Check API health
    echo "Checking API health..."
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ API is healthy"
    else
        echo "❌ API is not responding"
    fi
    
    # Check Redis health
    if check_redis; then
        echo "✅ Redis is healthy"
    else
        echo "❌ Redis is not healthy"
    fi
    
    # Check Redis Commander (if running)
    echo "Checking Redis Commander..."
    if curl -f http://localhost:8081 > /dev/null 2>&1; then
        echo "✅ Redis Commander is running at http://localhost:8081"
        echo "   Username: admin, Password: secret"
    else
        echo "ℹ️  Redis Commander is not running (use 'dev' mode to start)"
    fi
}

# Function to clean up Docker resources
cleanup() {
    echo "🧹 Cleaning up Docker resources..."
    docker system prune -f
    docker volume prune -f
}

# Main script logic
case "$1" in
    "start")
        start_services $2
        ;;
    "stop")
        stop_services
        ;;
    "restart")
        restart_services $2
        ;;
    "logs")
        show_logs $2
        ;;
    "health")
        check_health
        ;;
    "cleanup")
        cleanup
        ;;
    "status")
        docker compose ps
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|logs|health|cleanup|status} [dev|prod]"
        echo ""
        echo "Commands:"
        echo "  start [dev|prod] - Start services (dev includes Redis commander)"
        echo "  stop             - Stop all services"
        echo "  restart [dev|prod] - Restart services"
        echo "  logs [prod]      - Show service logs"
        echo "  health           - Check service health"
        echo "  cleanup          - Clean up Docker resources"
        echo "  status           - Show service status"
        echo ""
        echo "Examples:"
        echo "  $0 start dev     - Start in development mode"
        echo "  $0 start prod    - Start in production mode"
        echo "  $0 restart       - Restart in standard mode"
        echo "  $0 logs          - Show logs"
        exit 1
        ;;
esac