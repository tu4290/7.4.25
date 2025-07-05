# Uber Elite Database MCP - Technical Context

## Core Technologies

### AI/ML Framework Stack

#### PydanticAI
- **Version**: Latest stable
- **Purpose**: Structured AI agent development and data validation
- **Key Features**:
  - Type-safe AI agent construction
  - Structured data processing with validation
  - Integration with Pydantic models
  - Agent-based workflow orchestration
- **Usage**: Primary framework for structured AI operations and data validation

#### PyTorch
- **Version**: 2.1+
- **Purpose**: Deep learning model development and training
- **Key Features**:
  - Dynamic computation graphs
  - CUDA acceleration support
  - Extensive model zoo
  - TorchScript for production deployment
- **Usage**: Custom neural network development, transformer models, research-oriented AI

#### JAX
- **Version**: 0.4+
- **Purpose**: High-performance numerical computing with automatic differentiation
- **Key Features**:
  - JIT compilation with XLA
  - Automatic vectorization (vmap)
  - Functional programming paradigm
  - NumPy-compatible API
- **Usage**: Performance-critical computations, scientific computing, optimization algorithms

#### TensorFlow
- **Version**: 2.15+
- **Purpose**: Enterprise-grade machine learning and model serving
- **Key Features**:
  - TensorFlow Serving for production deployment
  - TensorBoard for visualization
  - Extensive ecosystem and tools
  - Mobile and edge deployment support
- **Usage**: Production model deployment, enterprise integrations, scalable serving

#### Candle (Rust-based ML)
- **Version**: Latest
- **Purpose**: High-performance, memory-safe machine learning
- **Key Features**:
  - Rust-based implementation for safety and performance
  - CUDA and Metal acceleration
  - Minimal dependencies
  - WebAssembly support
- **Usage**: Performance-critical inference, edge deployment, memory-constrained environments

### Core Python Stack

#### Python
- **Version**: 3.11+
- **Rationale**: Latest stable version with performance improvements and new features
- **Key Features**: Type hints, async/await, pattern matching, performance optimizations

#### FastAPI
- **Version**: 0.104+
- **Purpose**: High-performance web framework for API development
- **Key Features**:
  - Automatic OpenAPI documentation
  - Type validation with Pydantic
  - Async/await support
  - High performance (comparable to NodeJS)
- **Usage**: MCP protocol implementation, REST API endpoints, WebSocket support

#### Pydantic
- **Version**: 2.5+
- **Purpose**: Data validation and serialization
- **Key Features**:
  - Type validation and coercion
  - JSON schema generation
  - Performance optimizations
  - Integration with FastAPI
- **Usage**: Data model definitions, API validation, configuration management

#### AsyncIO
- **Purpose**: Asynchronous programming support
- **Key Features**:
  - Non-blocking I/O operations
  - Concurrent request handling
  - Event loop management
  - Integration with web frameworks
- **Usage**: Concurrent AI processing, database operations, external API calls

### Data Processing Stack

#### NumPy
- **Version**: 1.24+
- **Purpose**: Fundamental numerical computing
- **Key Features**: N-dimensional arrays, mathematical functions, broadcasting
- **Usage**: Core numerical operations, data preprocessing, mathematical computations

#### Pandas
- **Version**: 2.1+
- **Purpose**: Data manipulation and analysis
- **Key Features**: DataFrames, time series analysis, data cleaning, I/O operations
- **Usage**: Structured data processing, data transformation, analysis operations

#### Apache Arrow
- **Version**: 14.0+
- **Purpose**: High-performance columnar data processing
- **Key Features**: Zero-copy data sharing, cross-language compatibility, efficient serialization
- **Usage**: High-performance data exchange, memory-efficient operations, interoperability

#### Polars
- **Version**: 0.20+
- **Purpose**: Fast DataFrame library with lazy evaluation
- **Key Features**: Rust-based performance, lazy evaluation, memory efficiency
- **Usage**: Large dataset processing, performance-critical data operations

### Memory and Storage

#### Redis
- **Version**: 7.0+
- **Purpose**: In-memory data structure store
- **Key Features**:
  - High-performance caching
  - Data structure support (strings, hashes, lists, sets)
  - Pub/Sub messaging
  - Persistence options
- **Usage**: Short-term memory (1,000 active states), caching, session storage

#### PostgreSQL
- **Version**: 15+
- **Purpose**: Advanced relational database
- **Key Features**:
  - ACID compliance
  - JSON/JSONB support
  - Full-text search
  - Extensibility
- **Usage**: Long-term memory (100,000 patterns), structured data storage, metadata

#### Vector Databases
- **Options**: Chroma, Weaviate, or Pinecone
- **Purpose**: Efficient similarity search and vector storage
- **Key Features**:
  - High-dimensional vector storage
  - Similarity search algorithms
  - Metadata filtering
  - Scalable indexing
- **Usage**: Episodic memory (10,000 scenarios), embedding storage, semantic search

#### SQLite
- **Version**: 3.40+
- **Purpose**: Lightweight embedded database
- **Key Features**: Zero-configuration, serverless, self-contained
- **Usage**: Development, testing, small-scale deployments, configuration storage

### Infrastructure and Deployment

#### Docker
- **Version**: 24.0+
- **Purpose**: Containerization platform
- **Key Features**: Lightweight containers, multi-stage builds, layer caching
- **Usage**: Application packaging, development environment, deployment consistency

#### Kubernetes
- **Version**: 1.28+
- **Purpose**: Container orchestration
- **Key Features**: Auto-scaling, service discovery, rolling updates, health checks
- **Usage**: Production deployment, scaling, service management, high availability

#### NGINX
- **Version**: 1.24+
- **Purpose**: Web server and reverse proxy
- **Key Features**: Load balancing, SSL termination, caching, rate limiting
- **Usage**: API gateway, load balancing, SSL termination, static file serving

### Monitoring and Observability

#### Prometheus
- **Version**: 2.47+
- **Purpose**: Metrics collection and monitoring
- **Key Features**: Time-series database, PromQL query language, alerting
- **Usage**: System metrics, performance monitoring, alerting rules

#### Grafana
- **Version**: 10.0+
- **Purpose**: Metrics visualization and dashboards
- **Key Features**: Rich visualizations, alerting, data source integration
- **Usage**: Performance dashboards, system monitoring, alert visualization

#### Jaeger
- **Version**: 1.50+
- **Purpose**: Distributed tracing
- **Key Features**: Request tracing, performance analysis, dependency mapping
- **Usage**: Request flow analysis, performance bottleneck identification

## Development Setup

### Environment Management

#### Conda Environment
```yaml
name: uber-elite-database-mcp
channels:
  - conda-forge
  - pytorch
  - nvidia
dependencies:
  - python=3.11
  - pytorch
  - tensorflow
  - jax
  - numpy
  - pandas
  - redis-py
  - psycopg2
  - fastapi
  - uvicorn
  - pydantic
  - pytest
  - black
  - mypy
  - pip
  - pip:
    - pydantic-ai
    - candle-core
    - polars
    - chromadb
```

#### Development Tools
- **Code Formatting**: Black, isort
- **Type Checking**: mypy, pyright
- **Linting**: ruff, flake8
- **Testing**: pytest, hypothesis, pytest-asyncio
- **Documentation**: Sphinx, mkdocs

### Project Structure
```
uber_elite_database_mcp/
├── src/
│   ├── core/
│   │   ├── intelligence/          # Core AI engine
│   │   ├── memory/               # Memory architecture
│   │   ├── frameworks/           # AI framework adapters
│   │   └── data/                # Data processing
│   ├── mcp/
│   │   ├── protocol/            # MCP protocol implementation
│   │   ├── gateway/             # API gateway
│   │   └── tools/               # MCP tools and resources
│   ├── infrastructure/
│   │   ├── database/            # Database connections
│   │   ├── cache/               # Caching layer
│   │   ├── monitoring/          # Observability
│   │   └── security/            # Security components
│   └── utils/
│       ├── config/              # Configuration management
│       ├── logging/             # Logging utilities
│       └── validation/          # Data validation
├── tests/
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   ├── performance/             # Performance tests
│   └── fixtures/                # Test fixtures
├── docs/
│   ├── api/                     # API documentation
│   ├── architecture/            # Architecture docs
│   └── deployment/              # Deployment guides
├── scripts/
│   ├── setup/                   # Setup scripts
│   ├── deployment/              # Deployment scripts
│   └── maintenance/             # Maintenance scripts
├── config/
│   ├── development.yaml         # Development configuration
│   ├── production.yaml          # Production configuration
│   └── testing.yaml             # Testing configuration
└── docker/
    ├── Dockerfile               # Main application container
    ├── Dockerfile.dev           # Development container
    └── docker-compose.yml       # Multi-service setup
```

### Configuration Management

#### Environment Variables
```bash
# Database Configuration
DATABASE_URL=postgresql://user:pass@localhost:5432/uber_elite_db
REDIS_URL=redis://localhost:6379/0
VECTOR_DB_URL=http://localhost:8000

# AI Framework Configuration
PYTORCH_DEVICE=cuda
JAX_PLATFORM_NAME=gpu
TF_FORCE_GPU_ALLOW_GROWTH=true

# Performance Configuration
MAX_WORKERS=8
BATCH_SIZE=32
INFERENCE_TIMEOUT=10

# Security Configuration
SECRET_KEY=your-secret-key
API_KEY_EXPIRY=3600
SSL_CERT_PATH=/path/to/cert.pem

# Monitoring Configuration
PROMETHEUS_PORT=9090
JAEGER_ENDPOINT=http://localhost:14268
LOG_LEVEL=INFO
```

#### Configuration Schema
```python
from pydantic import BaseSettings, Field
from typing import Optional

class DatabaseConfig(BaseSettings):
    url: str = Field(..., env="DATABASE_URL")
    pool_size: int = Field(10, env="DB_POOL_SIZE")
    max_overflow: int = Field(20, env="DB_MAX_OVERFLOW")
    pool_timeout: int = Field(30, env="DB_POOL_TIMEOUT")

class AIConfig(BaseSettings):
    pytorch_device: str = Field("cpu", env="PYTORCH_DEVICE")
    jax_platform: str = Field("cpu", env="JAX_PLATFORM_NAME")
    tf_memory_growth: bool = Field(True, env="TF_FORCE_GPU_ALLOW_GROWTH")
    inference_timeout: int = Field(10, env="INFERENCE_TIMEOUT")
    batch_size: int = Field(32, env="BATCH_SIZE")

class PerformanceConfig(BaseSettings):
    max_workers: int = Field(8, env="MAX_WORKERS")
    request_timeout: int = Field(30, env="REQUEST_TIMEOUT")
    max_concurrent_requests: int = Field(1000, env="MAX_CONCURRENT_REQUESTS")
    memory_limit_gb: int = Field(8, env="MEMORY_LIMIT_GB")
```

### Logging Configuration

#### Structured Logging
```python
import structlog
from structlog.stdlib import LoggerFactory

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
```

## Technical Constraints

### Performance Requirements

#### Latency Constraints
- **Inference Time**: < 10ms for standard predictions
- **API Response**: < 100ms for most operations
- **Memory Access**: < 1ms for cache operations
- **Database Queries**: < 50ms for simple queries

#### Throughput Requirements
- **Concurrent Requests**: > 1,000 requests per second
- **Batch Processing**: > 10,000 records per minute
- **Memory Operations**: > 100,000 operations per second
- **Data Ingestion**: > 1GB per minute

#### Resource Constraints
- **Memory Usage**: < 2GB base footprint
- **CPU Utilization**: < 80% under normal load
- **GPU Memory**: < 8GB for AI operations
- **Disk I/O**: < 100MB/s sustained

### Scalability Constraints

#### Horizontal Scaling
- **Stateless Design**: All components must be stateless
- **Load Balancing**: Support for multiple instance deployment
- **Data Partitioning**: Efficient data distribution strategies
- **Cache Distribution**: Distributed caching with consistency

#### Vertical Scaling
- **Memory Scaling**: Linear memory scaling with data size
- **CPU Scaling**: Efficient multi-core utilization
- **GPU Scaling**: Support for multiple GPU configurations
- **Storage Scaling**: Efficient storage utilization and growth

### Security Constraints

#### Data Protection
- **Encryption**: AES-256 for data at rest, TLS 1.3 for data in transit
- **Access Control**: Role-based access with fine-grained permissions
- **Audit Logging**: Comprehensive audit trail for all operations
- **Data Anonymization**: Automatic PII detection and protection

#### Network Security
- **API Security**: OAuth 2.0, API key management, rate limiting
- **Network Isolation**: VPC, firewall rules, network segmentation
- **Certificate Management**: Automated certificate rotation and validation
- **Vulnerability Management**: Regular security scanning and updates

### Compliance Requirements

#### Data Privacy
- **GDPR Compliance**: Right to be forgotten, data portability, consent management
- **CCPA Compliance**: Data transparency, opt-out mechanisms, data deletion
- **HIPAA Compliance**: Healthcare data protection (if applicable)
- **SOC 2 Compliance**: Security, availability, processing integrity

#### Industry Standards
- **ISO 27001**: Information security management
- **PCI DSS**: Payment card data security (if applicable)
- **FedRAMP**: Federal risk and authorization management (if applicable)
- **SOX Compliance**: Financial reporting controls (if applicable)

## Dependencies

### Core Dependencies

#### AI/ML Libraries
```toml
[tool.poetry.dependencies]
python = "^3.11"
pydantic-ai = "^0.0.14"
torch = "^2.1.0"
jax = "^0.4.20"
tensorflow = "^2.15.0"
candle-core = "^0.3.0"
numpy = "^1.24.0"
pandas = "^2.1.0"
polars = "^0.20.0"
scikit-learn = "^1.3.0"
```

#### Web Framework
```toml
fastapi = "^0.104.0"
uvicorn = "^0.24.0"
pydantic = "^2.5.0"
starlette = "^0.27.0"
httpx = "^0.25.0"
websockets = "^12.0"
```

#### Database and Storage
```toml
psycopg2-binary = "^2.9.0"
redis = "^5.0.0"
sqlalchemy = "^2.0.0"
alembic = "^1.12.0"
chromadb = "^0.4.0"
pyarrow = "^14.0.0"
```

#### Infrastructure
```toml
prometheus-client = "^0.19.0"
opentelemetry-api = "^1.21.0"
opentelemetry-sdk = "^1.21.0"
structlog = "^23.2.0"
click = "^8.1.0"
pyyaml = "^6.0.0"
```

### Development Dependencies

#### Testing
```toml
[tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"
pytest-asyncio = "^0.21.0"
pytest-cov = "^4.1.0"
hypothesis = "^6.88.0"
factory-boy = "^3.3.0"
responses = "^0.24.0"
```

#### Code Quality
```toml
black = "^23.9.0"
isort = "^5.12.0"
mypy = "^1.6.0"
ruff = "^0.1.0"
pre-commit = "^3.5.0"
bandit = "^1.7.0"
```

#### Documentation
```toml
sphinx = "^7.2.0"
sphinx-rtd-theme = "^1.3.0"
mkdocs = "^1.5.0"
mkdocs-material = "^9.4.0"
```

### Optional Dependencies

#### GPU Acceleration
```toml
# CUDA support
torch-cuda = { version = "^2.1.0", optional = true }
tensorflow-gpu = { version = "^2.15.0", optional = true }
jax-cuda = { version = "^0.4.20", optional = true }

# ROCm support (AMD GPUs)
torch-rocm = { version = "^2.1.0", optional = true }

# Metal support (Apple Silicon)
tensorflow-metal = { version = "^1.0.0", optional = true }
```

#### Cloud Integrations
```toml
# AWS
boto3 = { version = "^1.29.0", optional = true }
aioboto3 = { version = "^12.0.0", optional = true }

# Google Cloud
google-cloud-storage = { version = "^2.10.0", optional = true }
google-cloud-bigquery = { version = "^3.12.0", optional = true }

# Azure
azure-storage-blob = { version = "^12.19.0", optional = true }
azure-cosmos = { version = "^4.5.0", optional = true }
```

## Integration Points

### MCP Protocol Integration

#### Protocol Implementation
- **Transport**: WebSocket and HTTP support
- **Message Format**: JSON-RPC 2.0 compliance
- **Tool Registration**: Dynamic tool discovery and registration
- **Resource Management**: Efficient resource allocation and cleanup

#### Tool Definitions
```python
from pydantic import BaseModel
from typing import List, Dict, Any

class MCPTool(BaseModel):
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]

class MCPResource(BaseModel):
    uri: str
    name: str
    description: str
    mime_type: str
```

### External API Integration

#### REST API Clients
- **HTTP Client**: httpx with connection pooling
- **Authentication**: OAuth 2.0, API keys, JWT tokens
- **Rate Limiting**: Intelligent rate limiting and backoff
- **Error Handling**: Comprehensive error handling and retry logic

#### GraphQL Integration
- **Client**: gql with async support
- **Schema Introspection**: Automatic schema discovery
- **Query Optimization**: Efficient query batching and caching
- **Subscription Support**: Real-time data updates

### Database Integration

#### SQL Databases
- **ORM**: SQLAlchemy with async support
- **Migration**: Alembic for schema management
- **Connection Pooling**: Optimized connection pools
- **Query Optimization**: Efficient query generation and caching

#### NoSQL Databases
- **Document Stores**: MongoDB, CouchDB integration
- **Key-Value Stores**: Redis, DynamoDB integration
- **Graph Databases**: Neo4j, ArangoDB integration
- **Time Series**: InfluxDB, TimescaleDB integration

### Message Queue Integration

#### Async Messaging
- **Redis Pub/Sub**: Real-time messaging and notifications
- **RabbitMQ**: Reliable message queuing and routing
- **Apache Kafka**: High-throughput event streaming
- **AWS SQS/SNS**: Cloud-native messaging services

#### Event-Driven Architecture
- **Event Sourcing**: Complete event history and replay
- **CQRS**: Command Query Responsibility Segregation
- **Saga Pattern**: Distributed transaction management
- **Event Store**: Persistent event storage and querying

## Deployment Considerations

### Container Strategy

#### Multi-Stage Builds
```dockerfile
# Build stage
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim as production
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY src/ ./src/
EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Resource Limits
```yaml
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "1000m"
    nvidia.com/gpu: "1"
```

### Kubernetes Deployment

#### Deployment Configuration
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: uber-elite-database-mcp
spec:
  replicas: 3
  selector:
    matchLabels:
      app: uber-elite-database-mcp
  template:
    metadata:
      labels:
        app: uber-elite-database-mcp
    spec:
      containers:
      - name: app
        image: uber-elite-database-mcp:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-secret
              key: url
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

#### Service Configuration
```yaml
apiVersion: v1
kind: Service
metadata:
  name: uber-elite-database-mcp-service
spec:
  selector:
    app: uber-elite-database-mcp
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Monitoring and Observability

#### Health Checks
```python
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
    )

@app.get("/ready")
async def readiness_check():
    """Readiness check with dependency validation."""
    # Check database connectivity
    # Check Redis connectivity
    # Check AI framework availability
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "ready", "dependencies": "all_healthy"}
    )
```

#### Metrics Exposure
```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Define metrics
request_count = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
request_duration = Histogram('http_request_duration_seconds', 'HTTP request duration')
active_connections = Gauge('active_connections', 'Number of active connections')
ai_inference_time = Histogram('ai_inference_duration_seconds', 'AI inference duration')

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type="text/plain")
```