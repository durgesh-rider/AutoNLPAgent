<template>
  <v-container class="pa-6">
    <v-row>
      <v-col cols="12">
        <v-btn text @click="$router.push('/models')" class="mb-4">
          <v-icon left>mdi-arrow-left</v-icon>
          Back to Models
        </v-btn>
        <h1 class="text-h4 font-weight-bold mb-2">Model Evaluation & Prediction</h1>
        <p class="text-subtitle-1 text-grey-darken-1 mb-6">
          Test your trained model and view performance metrics
        </p>
      </v-col>
    </v-row>

    <!-- Loading State -->
    <v-row v-if="loading">
      <v-col cols="12" class="text-center py-12">
        <v-progress-circular indeterminate color="primary" size="64"></v-progress-circular>
        <p class="text-h6 mt-4">Loading model results...</p>
      </v-col>
    </v-row>

    <!-- Model Information -->
    <v-row v-if="!loading && modelInfo">
      <v-col cols="12">
        <v-card elevation="2" class="mb-4">
          <v-card-title class="bg-blue-lighten-5">
            <v-icon class="mr-2">mdi-information</v-icon>
            Model Information
          </v-card-title>
          <v-card-text>
            <v-row>
              <v-col cols="12" md="3">
                <div class="text-caption text-grey-darken-1">Model ID</div>
                <div class="text-body-1 font-weight-medium">{{ modelId }}</div>
              </v-col>
              <v-col cols="12" md="3">
                <div class="text-caption text-grey-darken-1">Task Type</div>
                <div class="text-body-1 font-weight-medium">
                  <v-chip color="primary" size="small">{{ modelInfo.task_type || 'N/A' }}</v-chip>
                </div>
              </v-col>
              <v-col cols="12" md="3">
                <div class="text-caption text-grey-darken-1">Status</div>
                <div class="text-body-1 font-weight-medium">
                  <v-chip :color="getStatusColor(modelInfo.status)" size="small">
                    {{ modelInfo.status || 'Completed' }}
                  </v-chip>
                </div>
              </v-col>
              <v-col cols="12" md="3">
                <div class="text-caption text-grey-darken-1">Created</div>
                <div class="text-body-1 font-weight-medium">{{ formatDate(modelInfo.created_at) }}</div>
              </v-col>
            </v-row>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>

    <!-- Metrics Cards -->
    <v-row v-if="!loading && metrics">
      <v-col cols="12" md="3">
        <v-card elevation="2" class="text-center pa-4">
          <v-icon size="48" color="success" class="mb-2">mdi-check-circle</v-icon>
          <div class="text-h4 font-weight-bold">{{ (metrics.accuracy * 100).toFixed(2) }}%</div>
          <div class="text-caption text-grey-darken-1">Accuracy</div>
        </v-card>
      </v-col>
      <v-col cols="12" md="3">
        <v-card elevation="2" class="text-center pa-4">
          <v-icon size="48" color="primary" class="mb-2">mdi-target</v-icon>
          <div class="text-h4 font-weight-bold">{{ (metrics.precision * 100).toFixed(2) }}%</div>
          <div class="text-caption text-grey-darken-1">Precision</div>
        </v-card>
      </v-col>
      <v-col cols="12" md="3">
        <v-card elevation="2" class="text-center pa-4">
          <v-icon size="48" color="info" class="mb-2">mdi-magnify</v-icon>
          <div class="text-h4 font-weight-bold">{{ (metrics.recall * 100).toFixed(2) }}%</div>
          <div class="text-caption text-grey-darken-1">Recall</div>
        </v-card>
      </v-col>
      <v-col cols="12" md="3">
        <v-card elevation="2" class="text-center pa-4">
          <v-icon size="48" color="warning" class="mb-2">mdi-chart-line</v-icon>
          <div class="text-h4 font-weight-bold">{{ (metrics.f1 * 100).toFixed(2) }}%</div>
          <div class="text-caption text-grey-darken-1">F1-Score</div>
        </v-card>
      </v-col>
    </v-row>

    <!-- Prediction Section -->
    <v-row v-if="!loading">
      <v-col cols="12" md="8">
        <v-card elevation="2">
          <v-card-title class="bg-green-lighten-5">
            <v-icon class="mr-2">mdi-test-tube</v-icon>
            Test Your Model
          </v-card-title>
          <v-card-text class="pa-6">
            <v-textarea
              v-model="inputText"
              label="Enter text to predict"
              placeholder="Type or paste text here..."
              variant="outlined"
              rows="4"
              class="mb-4"
            ></v-textarea>

            <v-btn
              color="primary"
              block
              size="large"
              :loading="predicting"
              :disabled="!inputText || predicting"
              @click="makePrediction"
            >
              <v-icon left>mdi-play</v-icon>
              Predict
            </v-btn>

            <!-- Prediction Result -->
            <v-expand-transition>
              <v-alert
                v-if="predictionResult"
                :type="predictionResult.success ? 'success' : 'error'"
                class="mt-4"
                prominent
              >
                <v-alert-title>
                  {{ predictionResult.success ? 'Prediction Result' : 'Prediction Failed' }}
                </v-alert-title>
                <div v-if="predictionResult.success" class="mt-3">
                  <div class="text-h6 mb-2">
                    {{ predictionResult.data.label }}
                  </div>
                  <div class="text-body-2">
                    Confidence: {{ (predictionResult.data.confidence * 100).toFixed(2) }}%
                  </div>
                  <v-progress-linear
                    :model-value="predictionResult.data.confidence * 100"
                    color="success"
                    height="8"
                    rounded
                    class="mt-2"
                  ></v-progress-linear>
                </div>
                <div v-else class="mt-2">
                  {{ predictionResult.error }}
                </div>
              </v-alert>
            </v-expand-transition>
          </v-card-text>
        </v-card>

        <!-- Detailed Metrics -->
        <v-card elevation="2" class="mt-4" v-if="metrics">
          <v-card-title class="bg-purple-lighten-5">
            <v-icon class="mr-2">mdi-chart-box</v-icon>
            Detailed Metrics
          </v-card-title>
          <v-card-text>
            <v-simple-table>
              <template v-slot:default>
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Description</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td><strong>Accuracy</strong></td>
                    <td>{{ (metrics.accuracy * 100).toFixed(2) }}%</td>
                    <td>Overall correct predictions</td>
                  </tr>
                  <tr>
                    <td><strong>Precision</strong></td>
                    <td>{{ (metrics.precision * 100).toFixed(2) }}%</td>
                    <td>Positive predictions that were correct</td>
                  </tr>
                  <tr>
                    <td><strong>Recall</strong></td>
                    <td>{{ (metrics.recall * 100).toFixed(2) }}%</td>
                    <td>Actual positives that were found</td>
                  </tr>
                  <tr>
                    <td><strong>F1-Score</strong></td>
                    <td>{{ (metrics.f1 * 100).toFixed(2) }}%</td>
                    <td>Harmonic mean of precision and recall</td>
                  </tr>
                  <tr v-if="metrics.loss">
                    <td><strong>Loss</strong></td>
                    <td>{{ metrics.loss.toFixed(4) }}</td>
                    <td>Training loss value</td>
                  </tr>
                </tbody>
              </template>
            </v-simple-table>
          </v-card-text>
        </v-card>
      </v-col>

      <!-- Actions Sidebar -->
      <v-col cols="12" md="4">
        <v-card elevation="2" class="mb-4">
          <v-card-title class="bg-orange-lighten-5">
            <v-icon class="mr-2">mdi-download</v-icon>
            Export Model
          </v-card-title>
          <v-card-text>
            <p class="text-body-2 mb-4">Download your trained model for deployment</p>
            <v-btn
              color="primary"
              block
              prepend-icon="mdi-download"
              @click="downloadModel"
            >
              Download Model
            </v-btn>
          </v-card-text>
        </v-card>

        <v-card elevation="2" class="mb-4">
          <v-card-title class="bg-cyan-lighten-5">
            <v-icon class="mr-2">mdi-api</v-icon>
            API Integration
          </v-card-title>
          <v-card-text>
            <p class="text-body-2 mb-2">Use this endpoint for predictions:</p>
            <v-code class="text-caption">
              POST /predict/{{ modelId }}
            </v-code>
            <v-btn
              variant="outlined"
              color="primary"
              block
              size="small"
              class="mt-3"
              @click="copyApiEndpoint"
            >
              <v-icon left small>mdi-content-copy</v-icon>
              Copy Endpoint
            </v-btn>
          </v-card-text>
        </v-card>

        <v-card elevation="2">
          <v-card-title class="bg-red-lighten-5">
            <v-icon class="mr-2">mdi-delete</v-icon>
            Danger Zone
          </v-card-title>
          <v-card-text>
            <p class="text-body-2 mb-4">Permanently delete this model</p>
            <v-btn
              color="error"
              variant="outlined"
              block
              @click="deleteModel"
            >
              <v-icon left>mdi-delete</v-icon>
              Delete Model
            </v-btn>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>
  </v-container>
</template>

<script>
import api from '@/services/api'

export default {
  name: 'Evaluation',
  props: {
    modelId: {
      type: String,
      required: true
    }
  },
  data() {
    return {
      loading: true,
      modelInfo: null,
      metrics: null,
      inputText: '',
      predicting: false,
      predictionResult: null
    }
  },
  mounted() {
    this.loadModelData()
  },
  methods: {
    async loadModelData() {
      this.loading = true
      try {
        // Load model information
        const modelResponse = await api.get(`/training/models/${this.modelId}`)
        this.modelInfo = modelResponse.data

        // Load evaluation metrics
        const metricsResponse = await api.get(`/evaluation/metrics/${this.modelId}`)
        this.metrics = metricsResponse.data.metrics
      } catch (error) {
        console.error('Failed to load model data:', error)
        alert('Failed to load model data')
      } finally {
        this.loading = false
      }
    },

    async makePrediction() {
      if (!this.inputText) return

      this.predicting = true
      this.predictionResult = null

      try {
        const response = await api.post(`/training/predict/${this.modelId}`, {
          texts: [this.inputText]
        })

        this.predictionResult = {
          success: true,
          data: {
            label: response.data.predictions[0],
            confidence: response.data.probabilities ? response.data.probabilities[0] : 0.95
          }
        }
      } catch (error) {
        this.predictionResult = {
          success: false,
          error: error.response?.data?.detail || 'Prediction failed'
        }
      } finally {
        this.predicting = false
      }
    },

    async downloadModel() {
      try {
        const response = await api.get(`/training/models/${this.modelId}/download`, {
          responseType: 'blob'
        })
        
        const url = window.URL.createObjectURL(new Blob([response.data]))
        const link = document.createElement('a')
        link.href = url
        link.setAttribute('download', `model_${this.modelId}.zip`)
        document.body.appendChild(link)
        link.click()
        link.remove()
      } catch (error) {
        alert('Failed to download model')
      }
    },

    copyApiEndpoint() {
      const endpoint = `${window.location.origin}/api/predict/${this.modelId}`
      navigator.clipboard.writeText(endpoint)
      alert('API endpoint copied to clipboard!')
    },

    async deleteModel() {
      if (!confirm('Are you sure you want to delete this model? This action cannot be undone.')) {
        return
      }

      try {
        await api.delete(`/training/models/${this.modelId}`)
        this.$router.push('/models')
      } catch (error) {
        alert('Failed to delete model')
      }
    },

    getStatusColor(status) {
      const colors = {
        'completed': 'success',
        'training': 'warning',
        'failed': 'error',
        'pending': 'info'
      }
      return colors[status?.toLowerCase()] || 'grey'
    },

    formatDate(dateString) {
      if (!dateString) return 'N/A'
      return new Date(dateString).toLocaleString()
    }
  }
}
</script>

<style scoped>
v-code {
  background: #f5f5f5;
  padding: 8px 12px;
  border-radius: 4px;
  display: block;
  font-family: monospace;
}
</style>
