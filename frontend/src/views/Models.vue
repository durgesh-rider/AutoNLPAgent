<template>
  <v-container>
    <v-row>
      <v-col cols="12">
        <v-card class="pa-6">
          <v-card-title class="text-h4 mb-4">
            <v-icon color="primary" class="mr-2">mdi-brain</v-icon>
            My Models
          </v-card-title>

          <!-- Loading State -->
          <v-progress-linear v-if="loading" indeterminate color="primary"></v-progress-linear>

          <!-- Error State -->
          <v-alert v-if="error" type="error" dismissible @click:close="error = null" class="mb-4">
            {{ error }}
          </v-alert>

          <!-- Empty State -->
          <v-card-text v-if="!loading && Object.keys(models).length === 0">
            <div class="text-center py-8">
              <v-icon size="100" color="grey">mdi-robot-off</v-icon>
              <p class="text-h6 mt-4">No models trained yet</p>
              <p class="text-body-1 text-grey">Train your first model from a dataset</p>
              <v-btn color="primary" to="/datasets" class="mt-4" prepend-icon="mdi-database">
                View Datasets
              </v-btn>
            </div>
          </v-card-text>

          <!-- Models List -->
          <v-card-text v-else>
            <v-row>
              <v-col v-for="(model, modelId) in models" :key="modelId" cols="12" md="6" lg="4">
                <v-card outlined hover>
                  <v-card-title class="d-flex align-center">
                    <v-icon color="success" class="mr-2">mdi-check-circle</v-icon>
                    <span class="text-truncate">{{ modelId }}</span>
                  </v-card-title>
                  <v-card-text>
                    <p><strong>Dataset ID:</strong> {{ model.dataset_id }}</p>
                    <p><strong>Task Type:</strong> {{ model.task_type }}</p>
                    <p v-if="model.model_info"><strong>Model Type:</strong> {{ model.model_info.model_type }}</p>
                    
                    <!-- Metrics -->
                    <div v-if="model.model_info && model.model_info.metrics" class="mt-3">
                      <p class="font-weight-bold">Performance Metrics:</p>
                      <v-chip-group column>
                        <v-chip v-if="model.model_info.metrics.accuracy" small color="success">
                          Accuracy: {{ (model.model_info.metrics.accuracy * 100).toFixed(2) }}%
                        </v-chip>
                        <v-chip v-if="model.model_info.metrics.f1_score" small color="info">
                          F1: {{ (model.model_info.metrics.f1_score * 100).toFixed(2) }}%
                        </v-chip>
                      </v-chip-group>
                    </div>
                  </v-card-text>
                  <v-card-actions>
                    <v-btn color="primary" @click="goToEvaluation(modelId)" prepend-icon="mdi-chart-line">
                      View Results
                    </v-btn>
                    <v-btn color="secondary" @click="showPredictDialog(modelId)" prepend-icon="mdi-play">
                      Predict
                    </v-btn>
                    <v-spacer></v-spacer>
                    <v-btn icon="mdi-delete" color="error" @click="deleteModel(modelId)"></v-btn>
                  </v-card-actions>
                </v-card>
              </v-col>
            </v-row>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>

    <!-- Prediction Dialog -->
    <v-dialog v-model="predictDialog" max-width="600">
      <v-card>
        <v-card-title>Make Predictions</v-card-title>
        <v-card-text>
          <v-textarea
            v-model="predictionText"
            label="Enter text to predict"
            rows="4"
            outlined
            placeholder="Type your text here..."
          ></v-textarea>

          <div v-if="predictions.length > 0" class="mt-4">
            <p class="font-weight-bold">Predictions:</p>
            <v-chip-group column>
              <v-chip v-for="(pred, idx) in predictions" :key="idx" color="primary">
                {{ pred }}
              </v-chip>
            </v-chip-group>
          </div>
        </v-card-text>
        <v-card-actions>
          <v-spacer></v-spacer>
          <v-btn color="primary" @click="predict" :loading="predicting">
            Predict
          </v-btn>
          <v-btn @click="predictDialog = false">Close</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <!-- Evaluation Dialog -->
    <v-dialog v-model="evalDialog" max-width="600">
      <v-card>
        <v-card-title>Model Evaluation</v-card-title>
        <v-card-text>
          <v-progress-circular v-if="evaluating" indeterminate color="primary"></v-progress-circular>
          
          <div v-else-if="evaluationResult">
            <p v-if="!evaluationResult.success" class="error--text">{{ evaluationResult.message }}</p>
            <div v-else>
              <p class="font-weight-bold">Metrics:</p>
              <v-simple-table>
                <template v-slot:default>
                  <thead>
                    <tr>
                      <th>Metric</th>
                      <th>Value</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="(value, key) in evaluationResult.metrics" :key="key">
                      <td class="text-capitalize">{{ key.replace('_', ' ') }}</td>
                      <td>{{ (value * 100).toFixed(2) }}%</td>
                    </tr>
                  </tbody>
                </template>
              </v-simple-table>
            </div>
          </div>
        </v-card-text>
        <v-card-actions>
          <v-spacer></v-spacer>
          <v-btn @click="evalDialog = false">Close</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </v-container>
</template>

<script>
import api from '@/services/api'

export default {
  name: 'Models',
  data() {
    return {
      models: {},
      loading: false,
      error: null,
      predictDialog: false,
      predicting: false,
      selectedModelId: null,
      predictionText: '',
      predictions: [],
      evalDialog: false,
      evaluating: false,
      evaluationResult: null
    }
  },
  mounted() {
    this.fetchModels()
  },
  methods: {
    async fetchModels() {
      this.loading = true
      this.error = null
      try {
        const response = await api.get('/training/models')
        this.models = response.data.models || {}
      } catch (error) {
        this.error = error.response?.data?.detail || 'Failed to fetch models'
        console.error('Error fetching models:', error)
      } finally {
        this.loading = false
      }
    },

    showPredictDialog(modelId) {
      this.selectedModelId = modelId
      this.predictionText = ''
      this.predictions = []
      this.predictDialog = true
    },

    async predict() {
      if (!this.predictionText.trim()) return

      this.predicting = true
      try {
        const texts = this.predictionText.split('\n').filter(t => t.trim())
        const payload = { texts: texts }
        
        const response = await api.post(`/training/predict/${this.selectedModelId}`, payload)
        
        if (response.data.success) {
          this.predictions = response.data.predictions
        } else {
          this.error = response.data.error || 'Prediction failed'
        }
      } catch (error) {
        console.error('Prediction error:', error)
        this.error = error.response?.data?.detail || error.message || 'Prediction failed'
      } finally {
        this.predicting = false
      }
    },

    async evaluateModel(modelId) {
      // Navigate to evaluation page instead of showing dialog
      this.$router.push({
        name: 'Evaluation',
        params: { modelId: modelId }
      })
    },
    
    goToEvaluation(modelId) {
      this.$router.push({
        name: 'Evaluation',
        params: { modelId: modelId }
      })
    },

    async deleteModel(modelId) {
      if (!confirm('Are you sure you want to delete this model?')) return

      try {
        await api.delete(`/training/models/${modelId}`)
        delete this.models[modelId]
        this.$forceUpdate()
      } catch (error) {
        this.error = error.response?.data?.detail || 'Failed to delete model'
      }
    }
  }
}
</script>
