# AutoNLP-Agent Developer Guide

> **Complete technical documentation for developers to understand the application architecture, codebase structure, and workflow implementation.**

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Backend Structure](#backend-structure)
4. [Frontend Structure](#frontend-structure)
5. [API Endpoints](#api-endpoints)
6. [User Workflow](#user-workflow)
7. [File-by-File Documentation](#file-by-file-documentation)
8. [Development Setup](#development-setup)
9. [Testing](#testing)

---

## Project Overview

**AutoNLP-Agent** is a no-code web platform for automating Natural Language Processing tasks. Users can upload datasets, configure training parameters, train models, and evaluate results without writing code.

### Tech Stack

**Backend:**
- FastAPI (Python web framework)
- Scikit-learn (Traditional ML models)
- Transformers (Hugging Face - Deep learning models)
- Pandas (Data processing)
- NLTK (Text preprocessing)

**Frontend:**
- Vue.js 3 (Composition API)
- Vuetify 3 (UI components)
- Vue Router (Navigation)
- Axios (API calls)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Frontend (Vue 3)                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Upload  │  │ Training │  │Evaluation│  │  Models  │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
└───────┼─────────────┼─────────────┼─────────────┼──────────┘
        │             │             │             │
        │    REST API (Axios HTTP Calls)          │
        │             │             │             │
┌───────▼─────────────▼─────────────▼─────────────▼──────────┐
│                      Backend (FastAPI)                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Upload  │  │ Training │  │Evaluation│  │   Tasks  │   │
│  │  Router  │  │  Router  │  │  Router  │  │  Router  │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       │             │             │             │           │
│  ┌────▼─────────────▼─────────────▼─────────────▼──────┐   │
│  │                 Services Layer                       │   │
│  │  • FileService  • MLService  • VizService           │   │
│  └──────────────────────────────────────────────────────┘   │
│       │             │             │                          │
│  ┌────▼─────────────▼─────────────▼──────┐                 │
│  │       Data Layer (JSON Files)         │                 │
│  │  • uploads/      • models/            │                 │
│  └───────────────────────────────────────┘                 │
└──────────────────────────────────────────────────────────────┘
```

---

## Backend Structure

### Directory Layout

```
backend/
├── app/
│   ├── __init__.py                 # Package initializer
│   ├── main.py                     # FastAPI app & CORS setup
│   ├── config.py                   # Configuration settings
│   ├── core/
│   │   ├── __init__.py
│   │   ├── task_detector.py        # Auto-detect NLP task type
│   │   ├── preprocessor.py         # Text preprocessing
│   │   ├── trainer.py              # Model training logic
│   │   └── evaluator.py            # Model evaluation
│   ├── models/
│   │   ├── __init__.py
│   │   └── response.py             # Pydantic response models
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── upload.py               # File upload endpoints
│   │   ├── tasks.py                # Task detection/config
│   │   ├── training.py             # Training & prediction
│   │   └── evaluation.py           # Evaluation & metrics
│   ├── services/
│   │   ├── __init__.py
│   │   ├── file_service.py         # File handling service
│   │   ├── ml_service.py           # ML operations service
│   │   └── viz_service.py          # Visualization service
│   └── utils/
│       ├── __init__.py
│       └── helpers.py              # Utility functions
├── uploads/                        # Uploaded datasets
├── models/                         # Trained models
├── requirements.txt                # Python dependencies
└── start_server.py                 # Server startup script
```

### Key Backend Files

#### **app/main.py**
- **Purpose**: Application entry point
- **Functionality**:
  - Creates FastAPI application instance
  - Configures CORS middleware for frontend access
  - Includes all routers (upload, tasks, training, evaluation)
  - Defines health check and root endpoints
- **Key Code**:
  ```python
  app = FastAPI(title="AutoNLP-Agent", version="1.0.0")
  app.add_middleware(CORSMiddleware, allow_origins=["*"])
  app.include_router(upload_router)
  app.include_router(training_router)
  ```

#### **app/config.py**
- **Purpose**: Centralized configuration management
- **Functionality**:
  - Application settings (name, version, debug mode)
  - File paths (uploads, models directories)
  - Model configurations
  - Environment variable loading
- **Key Settings**:
  - `app_name`: Application name
  - `version`: API version
  - `upload_dir`: Dataset storage location
  - `models_dir`: Model storage location

#### **app/routers/upload.py**
- **Purpose**: Handle dataset upload operations
- **Endpoints**:
  - `POST /upload/` - Upload CSV/TXT/Excel files
  - `GET /upload/datasets` - List all uploaded datasets
  - `GET /upload/datasets/{dataset_id}` - Get specific dataset info
  - `DELETE /upload/datasets/{dataset_id}` - Delete dataset
- **Workflow**:
  1. Receive file upload
  2. Generate unique dataset ID (UUID)
  3. Save file to uploads directory
  4. Parse file (CSV/Excel/TXT)
  5. Auto-detect task type
  6. Return dataset metadata

#### **app/routers/training.py**
- **Purpose**: Handle model training and predictions
- **Endpoints**:
  - `POST /training/{dataset_id}` - Train model with config
  - `GET /training/models` - List all trained models
  - `GET /training/models/{model_id}` - Get model details
  - `POST /training/predict/{model_id}` - Make predictions
  - `DELETE /training/models/{model_id}` - Delete model
  - `GET /training/models/{model_id}/download` - Download model
- **TrainingConfig Parameters**:
  - `model_type`: 'sklearn' or 'transformer'
  - `epochs`: Training iterations (1-100)
  - `batch_size`: Samples per batch (1-128)
  - `learning_rate`: Learning rate (0-1)
  - `max_length`: Max sequence length (32-512)
  - `test_size`: Test split ratio (0-0.5)
- **Workflow**:
  1. Load dataset by ID
  2. Apply user configuration
  3. Preprocess text data
  4. Train model (sklearn or transformer)
  5. Save trained model
  6. Return model ID and metrics

#### **app/routers/evaluation.py**
- **Purpose**: Model evaluation and metrics
- **Endpoints**:
  - `GET /evaluation/metrics/{model_id}` - Get evaluation metrics
  - `POST /evaluation/{model_id}` - Evaluate model
  - `POST /evaluation/visualize/{evaluation_id}` - Generate charts
- **Metrics Returned**:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Confusion matrix (for visualization)

#### **app/services/file_service.py**
- **Purpose**: File management operations
- **Key Functions**:
  - `upload_dataset()`: Save and parse uploaded files
  - `get_dataset()`: Retrieve dataset by ID
  - `get_all_datasets()`: List all datasets
  - `delete_dataset()`: Remove dataset
- **File Handling**:
  - Supports CSV, TXT, Excel formats
  - UUID-based file naming
  - Metadata storage in JSON
  - Automatic task type detection

#### **app/services/ml_service.py**
- **Purpose**: Machine learning operations
- **Key Functions**:
  - `train_model()`: Train sklearn or transformer models
  - `predict()`: Make predictions
  - `evaluate_model()`: Calculate metrics
  - `get_model_info()`: Retrieve model metadata
  - `delete_model()`: Remove trained model
- **Model Types**:
  - **sklearn**: Logistic Regression, SVM, Random Forest
  - **transformer**: BERT, DistilBERT, RoBERTa
- **Training Logic**:
  - Auto-selects model based on data size
  - Applies text preprocessing
  - Handles train/test splitting
  - Saves model with metadata

#### **app/core/task_detector.py**
- **Purpose**: Auto-detect NLP task from dataset
- **Supported Tasks**:
  - Classification (text → category)
  - Sentiment Analysis (text → positive/negative/neutral)
  - NER (Named Entity Recognition)
  - Question Answering
  - Summarization
- **Detection Logic**:
  - Analyzes column names
  - Checks data patterns
  - Examines label distributions

#### **app/core/preprocessor.py**
- **Purpose**: Text preprocessing pipeline
- **Operations**:
  - Lowercase conversion
  - Tokenization
  - Stop word removal
  - Lemmatization/Stemming
  - Special character handling
  - Padding/Truncation

#### **app/core/trainer.py**
- **Purpose**: Model training implementation
- **Features**:
  - Supports both sklearn and transformers
  - Configurable hyperparameters
  - Validation split handling
  - Early stopping support
  - Model checkpointing

#### **app/core/evaluator.py**
- **Purpose**: Model evaluation and metrics
- **Metrics Calculated**:
  - Classification: accuracy, precision, recall, F1
  - Regression: MAE, MSE, R²
  - NER: entity-level F1
- **Visualization Support**:
  - Confusion matrix
  - ROC curves
  - Precision-Recall curves

---

## Frontend Structure

### Directory Layout

```
frontend/
├── public/
│   └── index.html                  # HTML template
├── src/
│   ├── main.js                     # Vue app entry point
│   ├── App.vue                     # Root component
│   ├── router/
│   │   └── index.js                # Vue Router configuration
│   ├── services/
│   │   └── api.js                  # Axios API client
│   ├── store/
│   │   └── index.js                # Vuex store (optional)
│   ├── views/
│   │   ├── Home.vue                # Landing page
│   │   ├── Upload.vue              # File upload page
│   │   ├── Training.vue            # Training config page
│   │   ├── Evaluation.vue          # Evaluation & prediction
│   │   ├── Datasets.vue            # Dataset management
│   │   ├── Models.vue              # Model management
│   │   └── Processing.vue          # Processing tasks
│   └── plugins/
│       └── vuetify.js              # Vuetify configuration
├── babel.config.js                 # Babel configuration
└── package.json                    # NPM dependencies
```

### Key Frontend Files

#### **src/main.js**
- **Purpose**: Vue application initialization
- **Functionality**:
  - Creates Vue app instance
  - Registers Vuetify plugin
  - Registers Vue Router
  - Mounts app to DOM
- **Key Code**:
  ```javascript
  import { createApp } from 'vue'
  import App from './App.vue'
  import router from './router'
  import vuetify from './plugins/vuetify'
  
  createApp(App).use(router).use(vuetify).mount('#app')
  ```

#### **src/App.vue**
- **Purpose**: Root component with layout
- **Features**:
  - App bar with navigation
  - Side navigation drawer
  - Main content area
  - Footer
- **Components**:
  - `v-app-bar`: Top navigation
  - `v-navigation-drawer`: Sidebar menu
  - `v-main`: Content area with router-view

#### **src/router/index.js**
- **Purpose**: Route configuration
- **Routes Defined**:
  - `/` → Home.vue
  - `/upload` → Upload.vue
  - `/training/:datasetId` → Training.vue
  - `/evaluation/:modelId` → Evaluation.vue
  - `/datasets` → Datasets.vue
  - `/models` → Models.vue
- **Route Props**: Passes URL params as component props

#### **src/services/api.js**
- **Purpose**: Axios API client configuration
- **Configuration**:
  - Base URL: `http://localhost:8000`
  - Headers: Content-Type, Accept
  - Interceptors for error handling
- **Usage**:
  ```javascript
  import api from '@/services/api'
  const response = await api.post('/training/123', config)
  ```

#### **src/views/Home.vue**
- **Purpose**: Landing page
- **Features**:
  - Hero section with gradient background
  - Feature cards (Upload, Train, Export)
  - Call-to-action button
- **Design**: Marketing-focused with visual appeal

#### **src/views/Upload.vue**
- **Purpose**: Dataset upload interface
- **Features**:
  - Drag-and-drop zone
  - File type validation (.csv, .txt, .xlsx)
  - Upload progress indicator
  - Success message with dataset info
  - Navigation buttons (Train Model, View Datasets)
- **Data Properties**:
  - `selectedFile`: Currently selected file
  - `isDragging`: Drag state
  - `uploading`: Upload in progress
  - `uploadResult`: Upload response data
- **Methods**:
  - `handleDrop()`: Handle file drop
  - `handleFileSelect()`: Handle file input
  - `uploadFile()`: POST to /upload/
  - `goToTraining()`: Navigate to training page

#### **src/views/Training.vue**
- **Purpose**: Model training configuration
- **Features**:
  - Dataset information display
  - Model selection dropdown
  - Parameter sliders (epochs, batch size, learning rate)
  - Advanced options (warmup, weight decay, FP16)
  - Parameter guidance sidebar
  - Training progress dialog
- **Configuration Options**:
  - Model: BERT, DistilBERT, RoBERTa, etc.
  - Epochs: 1-20 (slider)
  - Batch Size: 4-64 (slider)
  - Learning Rate: 0.00001-0.001 (input)
  - Max Length: 64-512 (slider)
  - Test Split: 10%-40% (slider)
- **Data Properties**:
  - `datasetInfo`: Dataset metadata
  - `config`: Training configuration
  - `training`: Training in progress
  - `trainingDialog`: Show progress dialog
- **Methods**:
  - `loadDatasetInfo()`: GET /upload/datasets/{id}
  - `startTraining()`: POST /training/{id} with config
  - `resetToDefaults()`: Reset parameters

#### **src/views/Evaluation.vue**
- **Purpose**: Model evaluation and prediction
- **Features**:
  - Model information display
  - Metrics cards (Accuracy, Precision, Recall, F1)
  - Interactive prediction input
  - Detailed metrics table
  - Model download button
  - API endpoint copy
  - Model deletion
- **Metrics Display**:
  - Visual metric cards with icons
  - Percentage formatting
  - Color-coded by metric type
- **Prediction Interface**:
  - Text area for input
  - Prediction button
  - Result display with confidence
  - Progress bar for confidence
- **Data Properties**:
  - `modelInfo`: Model metadata
  - `metrics`: Evaluation metrics
  - `inputText`: Prediction input
  - `predictionResult`: Prediction response
- **Methods**:
  - `loadModelData()`: Load model and metrics
  - `makePrediction()`: POST /training/predict/{id}
  - `downloadModel()`: GET /training/models/{id}/download
  - `deleteModel()`: DELETE /training/models/{id}

#### **src/views/Datasets.vue**
- **Purpose**: Dataset management
- **Features**:
  - List of uploaded datasets
  - Dataset cards with metadata
  - Train button → navigate to Training
  - Delete button
  - Empty state message
- **Data Display**:
  - Filename, task type, row count
  - Column preview (first 3 columns)
- **Methods**:
  - `fetchDatasets()`: GET /upload/datasets
  - `trainModel()`: Navigate to /training/{id}
  - `deleteDataset()`: DELETE /upload/datasets/{id}

#### **src/views/Models.vue**
- **Purpose**: Trained models management
- **Features**:
  - List of trained models
  - Model cards with metrics
  - View Results → navigate to Evaluation
  - Predict button
  - Delete button
  - Empty state message
- **Data Display**:
  - Model ID, task type, accuracy
  - F1 score chips
- **Methods**:
  - `fetchModels()`: GET /training/models
  - `goToEvaluation()`: Navigate to /evaluation/{id}
  - `showPredictDialog()`: Show prediction dialog
  - `deleteModel()`: DELETE /training/models/{id}

---

## API Endpoints

### Complete API Reference

#### Upload Service

| Method | Endpoint | Description | Request Body | Response |
|--------|----------|-------------|--------------|----------|
| POST | `/upload/` | Upload dataset | FormData(file) | `{dataset_id, task_type, row_count}` |
| GET | `/upload/datasets` | List datasets | - | `{datasets: [...]}` |
| GET | `/upload/datasets/{id}` | Get dataset info | - | `{dataset_id, task_type, columns, ...}` |
| DELETE | `/upload/datasets/{id}` | Delete dataset | - | `{message}` |

#### Training Service

| Method | Endpoint | Description | Request Body | Response |
|--------|----------|-------------|--------------|----------|
| POST | `/training/{dataset_id}` | Train model | TrainingConfig | `{model_id, success, metrics}` |
| GET | `/training/models` | List models | - | `{models: {...}}` |
| GET | `/training/models/{id}` | Get model info | - | `{model_id, task_type, ...}` |
| POST | `/training/predict/{id}` | Make predictions | `{texts: [...]}` | `{predictions: [...]}` |
| DELETE | `/training/models/{id}` | Delete model | - | `{message}` |
| GET | `/training/models/{id}/download` | Download model | - | ZIP file |

#### Evaluation Service

| Method | Endpoint | Description | Request Body | Response |
|--------|----------|-------------|--------------|----------|
| GET | `/evaluation/metrics/{model_id}` | Get metrics | - | `{metrics: {...}}` |
| POST | `/evaluation/{model_id}` | Evaluate model | - | `{success, metrics}` |
| POST | `/evaluation/visualize/{id}` | Generate charts | - | `{viz_id, charts}` |

---

## User Workflow

### Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Upload Dataset                                       │
│                                                               │
│  User Action: Upload CSV/TXT/Excel file                     │
│  Frontend: Upload.vue → POST /upload/                       │
│  Backend: FileService.upload_dataset()                      │
│  Result: dataset_id, task_type, metadata                    │
│                                                               │
│  UI: Success alert with "Configure & Train" button          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Configure Training                                   │
│                                                               │
│  User Action: Click "Configure & Train Model"               │
│  Frontend: Navigate to /training/{dataset_id}               │
│  Component: Training.vue                                     │
│                                                               │
│  User Inputs:                                                │
│  - Select model (BERT, DistilBERT, RoBERTa, etc.)          │
│  - Set epochs (slider: 1-20)                                │
│  - Set batch size (slider: 4-64)                            │
│  - Set learning rate (0.00001-0.001)                        │
│  - Set max length (64-512)                                  │
│  - Set test split (10%-40%)                                 │
│  - Advanced: warmup, weight decay, FP16                     │
│                                                               │
│  UI: Training config form with guidance                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Start Training                                       │
│                                                               │
│  User Action: Click "Start Training"                        │
│  Frontend: Training.vue → POST /training/{dataset_id}       │
│  Backend: TrainingRouter.train_model()                      │
│                                                               │
│  Backend Process:                                            │
│  1. Load dataset from file_service                          │
│  2. Apply user configuration                                │
│  3. Preprocess text (tokenization, cleaning)                │
│  4. Split train/test sets                                   │
│  5. Initialize model (sklearn or transformer)               │
│  6. Train model with config                                 │
│  7. Evaluate on test set                                    │
│  8. Save model to disk                                      │
│  9. Return model_id and metrics                             │
│                                                               │
│  UI: Progress dialog → Success → Auto-navigate              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 4: View Evaluation Results                              │
│                                                               │
│  Auto-Navigate: /evaluation/{model_id}                      │
│  Component: Evaluation.vue                                   │
│                                                               │
│  Data Loaded:                                                │
│  - GET /training/models/{model_id} → model info             │
│  - GET /evaluation/metrics/{model_id} → metrics             │
│                                                               │
│  UI Displays:                                                │
│  ┌─────────────────────────────────────────────┐            │
│  │ Model Info: ID, Task Type, Status           │            │
│  ├─────────────────────────────────────────────┤            │
│  │ Metrics Cards:                               │            │
│  │  • Accuracy: 92.5%                          │            │
│  │  • Precision: 91.2%                         │            │
│  │  • Recall: 93.8%                            │            │
│  │  • F1-Score: 92.4%                          │            │
│  ├─────────────────────────────────────────────┤            │
│  │ Interactive Prediction:                      │            │
│  │  [Text Input Area]                          │            │
│  │  [Predict Button]                           │            │
│  │  → Result: "Positive" (95% confidence)      │            │
│  ├─────────────────────────────────────────────┤            │
│  │ Actions:                                     │            │
│  │  • Download Model (ZIP)                     │            │
│  │  • Copy API Endpoint                        │            │
│  │  • Delete Model                             │            │
│  └─────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

### Alternative Workflows

**From Datasets Page:**
1. Navigate to /datasets
2. View all uploaded datasets
3. Click "Train Model" on any dataset
4. → Jump to Step 2 (Training Configuration)

**From Models Page:**
1. Navigate to /models
2. View all trained models
3. Click "View Results" on any model
4. → Jump to Step 4 (Evaluation)

---

## Development Setup

### Prerequisites

- Python 3.8+
- Node.js 14+
- npm or yarn

### Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment (optional)
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Start server
python start_server.py
# OR
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Backend runs on**: http://localhost:8000  
**API Docs**: http://localhost:8000/docs

### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run serve
```

**Frontend runs on**: http://localhost:8080

### Environment Variables

Create `.env` file in backend directory:

```env
APP_NAME=AutoNLP-Agent
VERSION=1.0.0
DEBUG=true
UPLOAD_DIR=uploads
MODELS_DIR=models
```

---

## Testing

### Backend Testing

```bash
cd backend
python -m pytest tests/ -v
```

**Test Files:**
- `test_upload.py`: Upload functionality
- `test_task_classification.py`: Classification tasks
- `test_task_sentiment.py`: Sentiment analysis
- `test_task_ner.py`: Named entity recognition
- `test_api_integration.py`: Full API integration

### Frontend Testing

```bash
cd frontend
npm run test:unit
```

### End-to-End Testing

```bash
cd testing
python test_api_flow.py
```

**Comprehensive Test Script:**
- Tests all API endpoints
- Validates complete workflow
- Checks data integrity
- Measures response times

### Manual Testing Checklist

- [ ] Upload CSV file
- [ ] View uploaded dataset
- [ ] Navigate to training
- [ ] Adjust training parameters
- [ ] Start training (1 epoch for speed)
- [ ] View evaluation metrics
- [ ] Make test prediction
- [ ] Download model
- [ ] Delete model
- [ ] Delete dataset

---

## Common Development Tasks

### Adding a New API Endpoint

1. **Define route in appropriate router** (`app/routers/`)
   ```python
   @router.post("/new-endpoint/{id}")
   async def new_endpoint(id: str, data: YourModel):
       # Implementation
       pass
   ```

2. **Create Pydantic model** (`app/models/response.py`)
   ```python
   class YourModel(BaseModel):
       field1: str
       field2: int
   ```

3. **Add service method** (`app/services/`)
   ```python
   def new_service_method(self, data):
       # Business logic
       return result
   ```

4. **Update frontend API call** (`src/services/api.js` or component)
   ```javascript
   const response = await api.post('/new-endpoint/123', data)
   ```

### Adding a New Frontend Page

1. **Create Vue component** (`src/views/NewPage.vue`)
   ```vue
   <template>
     <v-container>
       <!-- UI -->
     </v-container>
   </template>
   
   <script>
   export default {
     name: 'NewPage',
     // Component logic
   }
   </script>
   ```

2. **Add route** (`src/router/index.js`)
   ```javascript
   {
     path: '/new-page',
     name: 'NewPage',
     component: () => import('@/views/NewPage.vue')
   }
   ```

3. **Add navigation** (`src/App.vue`)
   ```vue
   <v-list-item to="/new-page" title="New Page" />
   ```

### Debugging Tips

**Backend:**
- Check logs in terminal
- Use FastAPI docs: http://localhost:8000/docs
- Add `print()` statements
- Use Python debugger: `import pdb; pdb.set_trace()`

**Frontend:**
- Open browser DevTools (F12)
- Check Console for errors
- Use Vue DevTools extension
- Check Network tab for API calls

---

## Performance Optimization

### Backend
- Use async/await for I/O operations
- Implement caching for frequently accessed data
- Batch predictions for better throughput
- Use model quantization for smaller models

### Frontend
- Lazy load components with `() => import()`
- Implement pagination for large lists
- Use virtual scrolling for data tables
- Minimize API calls with caching

---

## Security Considerations

- **File Upload**: Validate file types and sizes
- **CORS**: Configure allowed origins properly
- **API Rate Limiting**: Implement rate limiting
- **Input Validation**: Use Pydantic models
- **Error Handling**: Don't expose internal errors

---

## Deployment

### Docker Deployment

```bash
docker-compose up --build
```

**Services:**
- Backend: http://localhost:8000
- Frontend: http://localhost:3000

### Production Checklist

- [ ] Set DEBUG=false
- [ ] Configure CORS properly
- [ ] Use environment variables
- [ ] Set up logging
- [ ] Enable HTTPS
- [ ] Database for metadata (instead of JSON)
- [ ] Implement authentication
- [ ] Add rate limiting
- [ ] Set up monitoring

---

## Troubleshooting

### Common Issues

**Issue**: Backend won't start  
**Solution**: Check if port 8000 is available, install dependencies

**Issue**: Frontend can't connect to backend  
**Solution**: Verify CORS settings, check backend is running

**Issue**: Model training fails  
**Solution**: Check dataset format, reduce batch size, check memory

**Issue**: File upload fails  
**Solution**: Check file size limits, validate file format

---

## Contributing Guidelines

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and test thoroughly
4. Commit with clear messages: `git commit -m "Add feature X"`
5. Push to branch: `git push origin feature-name`
6. Create Pull Request

---

## License

MIT License - See LICENSE file for details

---

## Support

For issues and questions:
- GitHub Issues: [Repository Issues Page]
- Documentation: This file
- API Docs: http://localhost:8000/docs

---

**Last Updated**: November 5, 2025  
**Version**: 1.0.0  
**Maintainers**: AutoNLP-Agent Development Team
