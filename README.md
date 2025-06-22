# ABSA Sentiment Analysis Pipeline with SERVQUAL Intelligence

A production-ready hybrid sentiment analysis system combining traditional ABSA (Aspect-Based Sentiment Analysis) with LLM-powered SERVQUAL business intelligence for e-commerce app reviews.

## Features

### Core Capabilities
- **Dual Processing Modes**: Traditional ABSA for technical analysis + LLM SERVQUAL for business intelligence
- **Multi-Platform Support**: Amazon, eBay, Etsy, Temu, Shein-specific analysis
- **LLM Integration**: Mistral 7B via Ollama for direct SERVQUAL dimension classification
- **Real-time Dashboard**: Interactive Streamlit interface with competitive analysis
- **Batch Processing**: Robust pipeline with checkpoint recovery and progress tracking

### Performance Metrics
- **Superior Accuracy**: 71% reliability detection (60%+ improvement over keyword baseline)
- **Production Speed**: 5.5 seconds per review processing time
- **System Reliability**: 100% success rate across diverse review types
- **Multi-platform Intelligence**: Platform-specific context awareness

### SERVQUAL Dimensions
- **Reliability**: Product/service dependability analysis
- **Assurance**: Trust, security, and customer support evaluation  
- **Tangibles**: Interface and user experience assessment
- **Empathy**: Personal care and policy analysis
- **Responsiveness**: Speed and communication evaluation

## Architecture

```
Google Play Reviews → Data Ingestion → Dual Processing → Storage → Dashboard
                                     ↙            ↘
                            ABSA Engine    SERVQUAL LLM
                          (Technical)      (Business Intel)
                                ↓              ↓
                           PostgreSQL    PostgreSQL
                             ↓              ↓
                        Technical      Business Intelligence
                        Dashboard         Dashboard
```

## Installation

### Prerequisites
- Python 3.8+
- PostgreSQL 12+
- Redis 6+
- Ollama runtime
- 8GB+ RAM (recommended)

### Setup Steps

1. **Clone Repository**
```bash
git clone https://github.com/yourusername/absa-sentiment-pipeline.git
cd absa-sentiment-pipeline
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Install Ollama and Models**
```bash
# Install Ollama (follow official instructions for your OS)
curl -fsSL https://ollama.ai/install.sh | sh

# Download Mistral model (4GB)
ollama pull mistral:7b
```

4. **Setup Infrastructure**
```bash
# Start services
docker-compose up -d

# Initialize database
python scripts/setup_database.py
```

5. **Configure Environment**
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Usage

### Starting the Application

```bash
# Start the main dashboard
streamlit run dashboard_app.py

# Or use the main orchestrator
python main.py
```

### Processing Reviews

#### Batch Processing
```bash
# Process all pending reviews
python main.py --mode batch --analysis all

# Process specific app
python main.py --mode batch --app-id com.amazon.mShop.android.shopping

# ABSA only
python main.py --mode batch --analysis absa

# SERVQUAL LLM only  
python main.py --mode batch --analysis servqual_llm
```

#### Interactive Dashboard
- Navigate to `http://localhost:8501`
- Select processing mode from sidebar
- Monitor real-time progress
- View competitive analysis results

### API Usage

```python
from src.absa.engine import ABSAEngine

# Initialize engine
engine = ABSAEngine()

# ABSA Analysis
result = engine.analyze_review(
    review_id="123",
    app_id="com.example.app", 
    review_text="Great app but crashes sometimes",
    mode="deep"
)

# SERVQUAL LLM Analysis
servqual_result = engine.analyze_review_servqual_llm(
    review_id="123",
    app_id="com.example.app",
    review_text="Great app but crashes sometimes", 
    rating=4
)
```

## Configuration

### Environment Variables
```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/absa_db
REDIS_URL=redis://localhost:6379

# LLM Settings
OLLAMA_URL=http://localhost:11434
LLM_MODEL=mistral:7b
LLM_TIMEOUT=20

# Processing
BATCH_SIZE=50
CHECKPOINT_INTERVAL=15  # minutes
MAX_PROCESSING_TIME=10800  # 3 hours
```

### Model Configuration
Edit `config/servqual_llm_config.yml`:
```yaml
llm_settings:
  model_name: "mistral:7b"
  temperature: 0.1
  max_tokens: 120
  
performance_targets:
  max_processing_time: 6.0
  min_success_rate: 0.99
  target_throughput: 0.18
```

## Database Schema

### Key Tables
- `reviews`: Raw review data with processing flags
- `deep_absa`: Traditional ABSA results  
- `servqual_scores`: LLM-enhanced SERVQUAL analysis
- `processing_checkpoints`: Progress tracking and recovery

### Processing Flags
```sql
-- Track processing status per review
ALTER TABLE reviews ADD COLUMN absa_processed BOOLEAN DEFAULT FALSE;
ALTER TABLE reviews ADD COLUMN servqual_processed BOOLEAN DEFAULT FALSE;
```

## Development

### Project Structure
```
├── src/
│   ├── absa/              # ABSA processing engines
│   ├── data/              # Data layer operations  
│   ├── pipeline/          # Processing pipelines
│   └── utils/             # Shared utilities
├── dashboard/             # Streamlit components
├── notebooks/             # Development and testing
├── sql/                   # Database schemas
├── config/                # Configuration files
└── tests/                 # Test suites
```

### Running Tests
```bash
pytest tests/ -v
```

### Development Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with auto-reload
streamlit run dashboard_app.py --server.runOnSave=true
```

## Performance Optimization

### Batch Processing Tips
- Use batch sizes of 50-100 reviews for optimal performance
- Enable checkpoints for long-running processes
- Monitor memory usage during LLM processing
- Clear model cache between large batches

### Dashboard Performance
- Data refreshes every 5 minutes during processing
- Caching enabled for expensive queries
- Lazy loading for large datasets

## Troubleshooting

### Common Issues

**Ollama Connection Failed**
```bash
# Check Ollama status
ollama list

# Restart Ollama service
ollama serve
```

**Memory Issues**
```bash
# Clear model cache
python -c "from src.absa.models import clear_cache; clear_cache()"
```

**Database Connection**
```bash
# Test database connection
python -c "from src.data.storage import test_connection; test_connection()"
```

### Monitoring
- Check logs in `logs/` directory
- Monitor processing progress in dashboard
- Review checkpoint status in database

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Code Standards
- Follow PEP 8 style guidelines
- Add docstrings for all functions
- Include unit tests for new features
- Update documentation as needed

## Acknowledgments

- Mistral AI for the LLM model
- Ollama for local LLM runtime
- HuggingFace for transformer models
- Streamlit for dashboard framework
