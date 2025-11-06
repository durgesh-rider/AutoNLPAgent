# AutoNLP-Agent

A production-ready, no-code web-based autonomous NLP platform that automatically trains and deploys NLP models.

[![Tests](https://img.shields.io/badge/tests-48%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10+-blue)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green)]()
[![Vue](https://img.shields.io/badge/Vue-3.3+-brightgreen)]()

## 🚀 Features

### Core Capabilities
- **Automatic Task Detection**: Upload your dataset and let the system detect whether it's classification, regression, sentiment analysis, or NER
- **Smart Model Selection**: Automatically selects the best model architecture for your task (sklearn or transformers)
- **Intelligent Preprocessing**: Automatic text cleaning, tokenization, and label encoding

### Training & Configuration (NEW! ✨)
- **Full User Control**: Customize 10+ training parameters through intuitive UI
  - Model type (sklearn fast or transformer accurate)
  - Epochs (1-100 with visual sliders)
  - Batch size (8-64 with memory/speed trade-offs)
  - Max sequence length (32-512 tokens)
  - Early stopping & validation splits
- **Interactive Interface**: Real-time visual feedback with sliders and helpful hints
- **Flexible API**: Send training configs as JSON for programmatic control

### Predictions (NEW! ✨)
- **Meaningful Results**: Get actual labels ("positive"/"negative") instead of integers (0/1)
- **Real-time Predictions**: Make predictions instantly via API or UI
- **Label Decoding**: Automatic conversion from encoded predictions to original dataset labels

### Model Evaluation
- **Detailed Metrics**: Accuracy, precision, recall, F1-score
- **Visualizations**: Confusion matrices and performance charts
- **Model Comparison**: Compare different configurations

### Production Ready
- **Comprehensive Testing**: 48 passing tests with 85%+ coverage
- **Deployment Guides**: Docker, traditional, and cloud deployment options
- **Monitoring**: Built-in logging and error tracking
- **Documentation**: Complete API docs, maintenance guides, and troubleshooting

## 📋 Quick Start

### Prerequisites

- **Python 3.10+**
- **Node.js 14+**
- **npm or yarn**
- **4GB+ RAM** (recommended)

### Installation

**1. Clone the repository**
```bash
git clone <repository-url>
cd autonlp
```

**2. Install Backend Dependencies**
```bash
cd backend
pip install -r requirements.txt
```

**3. Install Frontend Dependencies**
```bash
cd frontend
npm install
```

### Running the Application

**Option 1: Start Both Servers (Recommended)**
```powershell
.\start_servers.ps1
```
This opens two PowerShell windows - one for backend, one for frontend.

**Option 2: Start Manually**

Backend (Terminal 1):
```powershell
.\start_backend.ps1
```

Frontend (Terminal 2):
```powershell
cd frontend
npm run serve
```

**Option 3: Docker**
```bash
docker-compose up --build
```

### Access Points

- **Frontend Application**: http://localhost:8081
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs (Swagger)
- **Alternative API Docs**: http://localhost:8000/redoc

## 📖 Usage

### 1. Upload Dataset

Navigate to the Upload page and upload a CSV file with:
- **Text column**: Your input text data
- **Label column**: Classification labels (2-50 unique values)

Example CSV:
```csv
text,sentiment
"I love this product!",positive
"This is terrible",negative
"Pretty good overall",positive
```

### 2. Configure & Train Model (Enhanced! ✨)

After upload, customize training with full control:

**Interactive UI Controls**:
- **Model Type**: Choose sklearn (fast) or transformer (accurate)
- **Epochs Slider**: 1-100 training iterations
- **Batch Size Slider**: 8-64 samples per batch
- **Max Length Slider**: 32-512 token sequence length
- **Validation Split**: Enable 20% holdout for better evaluation
- **Early Stopping**: Automatically stop if performance plateaus

**Or Use the API**:
```bash
POST /training/{dataset_id}
Content-Type: application/json

{
  "model_type": "sklearn",
  "epochs": 10,
  "batch_size": 32,
  "max_length": 256,
  "early_stopping": true,
  "use_validation": true
}
```

### 3. Make Predictions (Now with Real Labels! ✨)

**Before**: Confusing integers
```json
{"predictions": [0, 1, 0]}  // What does 0 mean?
```

**Now**: Meaningful labels from your dataset
```json
{"predictions": ["positive", "negative", "positive"]}  // Clear!
```

**API Example**:
```bash
POST /training/predict/{model_id}
{
  "texts": ["This is amazing!", "Not good at all"]
}

Response:
{
  "success": true,
  "predictions": ["positive", "negative"]
}
```
- **Batch Size**: 8-64 (smaller = more precise, larger = faster)
- **Use Validation**: Enable to prevent overfitting

Click "Start Training" and wait for completion.

### 3. Make Predictions

Go to Models page:
1. Select your trained model
2. Enter text (one per line)
3. Click "Predict"
4. View predictions with confidence scores

### 4. Evaluate Models

View comprehensive metrics:
- **Accuracy**: Overall correctness
- **Precision**: Positive prediction reliability
- **Recall**: Ability to find all positives
- **F1-Score**: Harmonic mean of precision and recall

## 🏗️ Project Structure

```
autonlp/
├── backend/                    # FastAPI backend
│   ├── app/
│   │   ├── core/              # ML core logic (task detection, training, evaluation)
│   │   ├── models/            # Pydantic models (request/response schemas)
│   │   ├── routers/           # API endpoints (upload, training, evaluation)
│   │   ├── services/          # Business logic (ML, file, NLP services)
│   │   └── utils/             # Utility functions
│   ├── tests/                 # 48 comprehensive tests
│   │   ├── test_api_integration.py  # API endpoint tests
│   │   ├── test_core.py             # Core ML logic tests
│   │   └── test_upload.py           # Upload functionality tests
│   ├── uploads/               # Uploaded datasets
│   ├── requirements.txt       # Python dependencies
│   ├── start_server.py        # Server startup script
│   └── run_tests.py           # Test runner
│
├── frontend/                  # Vue.js frontend
│   ├── src/
│   │   ├── views/            # Vue pages (Home, Upload, Datasets, Models, Processing)
│   │   ├── services/         # API client (axios)
│   │   ├── router/           # Vue router configuration
│   │   └── store/            # Vuex state management
│   ├── public/               # Static assets
│   └── package.json          # Node dependencies
│
├── testing/                   # Test scripts and demos
│   ├── demo.py               # Full workflow demonstration
│   ├── test_backend.py       # Backend API tests
│   ├── test_full_backend.py  # End-to-end integration tests
│   ├── run_tests.ps1         # Test runner script
│   └── README.md             # Testing documentation
│
├── .env.example              # Environment configuration template
├── .gitignore                # Git ignore rules
├── README.md                 # This file
├── DEPLOYMENT.md             # Production deployment guide
├── MAINTENANCE.md            # Maintenance and update guide
├── docker-compose.yml        # Docker configuration
├── start_servers.ps1         # Start both servers
└── start_backend.ps1         # Start backend only
```

## 🧪 Testing

The project includes **48 comprehensive tests** covering:
- API endpoints
- Core ML functionality
- File uploads
- Error handling
- End-to-end workflows

**Run All Tests:**
```bash
cd backend
pytest tests/ -v
```

**Run Specific Tests:**
```bash
# Core ML tests
pytest tests/test_core.py -v

# API integration tests
pytest tests/test_api_integration.py -v

# Upload tests
pytest tests/test_upload.py -v
```

**Run Integration Tests:**
```bash
# Full backend integration test
python testing/test_full_backend.py

# Demo script
python testing/demo.py
```

**Coverage Report:**
```bash
pytest tests/ --cov=app --cov-report=html
# Open htmlcov/index.html
```

**Test Results:**
```
48 passed in 13.25s
Coverage: 85%+
```

## 🛠️ Technology Stack

### Backend
- **[FastAPI](https://fastapi.tiangolo.com/)** 0.104+ - Modern Python web framework
- **[scikit-learn](https://scikit-learn.org/)** 1.3+ - Classical ML models
- **[transformers](https://huggingface.co/transformers/)** 4.36+ - Deep learning models (BERT, RoBERTa)
- **[pandas](https://pandas.pydata.org/)** 2.1+ - Data processing
- **[pydantic](https://docs.pydantic.dev/)** 2.5+ - Data validation

### Frontend
- **[Vue 3](https://vuejs.org/)** 3.3+ - Progressive JavaScript framework
- **[Vuetify 3](https://vuetifyjs.com/)** 3.4+ - Material Design components
- **[Axios](https://axios-http.com/)** 1.6+ - HTTP client
- **[Vue Router](https://router.vuejs.org/)** 4.2+ - Routing
- **[Vuex](https://vuex.vuejs.org/)** 4.1+ - State management

### Development
- **pytest** 8.4+ - Testing framework
- **httpx** 0.25+ - HTTP testing
- **Docker** - Containerization
- **uvicorn** 0.24+ - ASGI server

## 📚 Documentation

- **[README.md](README.md)** - Getting started (this file)
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Production deployment guide
- **[MAINTENANCE.md](MAINTENANCE.md)** - Code maintenance and updates
- **[testing/README.md](testing/README.md)** - Testing documentation
- **API Docs** - http://localhost:8000/docs (auto-generated)

## 🚀 Deployment

### Quick Deployment

**Using Docker (Recommended):**
```bash
# Copy environment file
cp .env.example .env

# Edit .env and set production values
nano .env

# Build and start
docker-compose up -d

# Check status
docker-compose ps
```

**Traditional Deployment:**
See [DEPLOYMENT.md](DEPLOYMENT.md) for:
- Production server setup
- Nginx configuration
- SSL/HTTPS setup
- Monitoring and logging
- Security best practices
- CI/CD pipeline setup

### Deployment Checklist

- [ ] Change `SECRET_KEY` in `.env`
- [ ] Set `DEBUG=False`
- [ ] Configure CORS with specific origins
- [ ] Set up HTTPS/SSL
- [ ] Configure database (optional)
- [ ] Set up monitoring (Sentry, Prometheus)
- [ ] Configure backups
- [ ] Run full test suite
- [ ] Load test endpoints

## 🔧 Maintenance

### Regular Tasks

**Daily:**
```bash
git pull origin main
pytest tests/ -v
```

**Weekly:**
```bash
pip list --outdated          # Check for updates
tail -f autonlp.log          # Review logs
```

**Monthly:**
```bash
pip install -r requirements.txt --upgrade  # Update dependencies
pytest tests/                               # Run tests
safety check                                # Security audit
```

See [MAINTENANCE.md](MAINTENANCE.md) for:
- Adding new features
- Updating dependencies
- Code quality standards
- Performance optimization
- Debugging guide
- Version control best practices

## 📊 Performance

### Benchmarks

- **Upload**: < 1s for 10MB CSV
- **Task Detection**: < 100ms
- **Model Training**: 
  - sklearn: 0.1-5s (10-1000 rows)
  - transformers: 10-300s (100-10000 rows)
- **Prediction**: < 50ms per text
- **Concurrent Users**: 100+ (with proper scaling)

### Optimization Tips

- Use smaller batch sizes for better accuracy
- Enable validation split to prevent overfitting
- Use sklearn models for small datasets (< 1000 rows)
- Use transformers for large datasets (> 1000 rows)
- Cache predictions for repeated queries
- Use GPU for transformer training

## 🔒 Security

### Security Features

- Input validation with Pydantic
- File type validation
- File size limits (100MB default)
- CORS configuration
- SQL injection prevention (using ORMs)
- XSS protection (frontend sanitization)

### Production Security

**Before deploying:**
1. Change `SECRET_KEY` in `.env`
2. Set `DEBUG=False`
3. Configure CORS with specific origins
4. Use HTTPS/SSL
5. Implement authentication (optional)
6. Set up rate limiting
7. Regular security audits

## 🐛 Troubleshooting

### Common Issues

**Port Already in Use:**
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <pid> /F

# Linux/Mac
lsof -i :8000
kill -9 <pid>
```

**Module Import Errors:**
```bash
# Set PYTHONPATH (PowerShell)
$env:PYTHONPATH="R:\WEB\autonlp\autonlp\backend"

# Or (Bash)
export PYTHONPATH="${PYTHONPATH}:/path/to/backend"
```

**Tests Failing:**
```bash
# Clear cache
rm -rf backend/__pycache__ backend/tests/__pycache__

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Run with verbose output
pytest tests/ -vv -s
```

**Frontend Build Errors:**
```bash
# Clear node modules
rm -rf frontend/node_modules

# Reinstall
cd frontend
npm install

# Clear npm cache
npm cache clean --force
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/ -v`)
5. Commit your changes (`git commit -m 'feat: add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Commit Message Format

```
type(scope): subject

feat(api): add translation endpoint
fix(trainer): resolve memory leak
docs(readme): update installation steps
test(core): add tests for new detector
refactor(services): simplify upload logic
```

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Amazing web framework
- [Hugging Face](https://huggingface.co/) - Transformers library
- [scikit-learn](https://scikit-learn.org/) - ML algorithms
- [Vue.js](https://vuejs.org/) - Progressive framework
- [Vuetify](https://vuetifyjs.com/) - Material Design components

## 📧 Support

- **Documentation**: Check `/docs` in running application
- **Issues**: Open a GitHub issue
- **Health Check**: GET `/health` endpoint
- **API Version**: GET `/` (root endpoint)

---

**Made with ❤️ for the NLP community**
