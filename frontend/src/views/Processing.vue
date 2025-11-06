<template>
  <v-container>
    <v-row>
      <v-col cols="12">
        <v-card class="pa-6">
          <v-card-title class="text-h4 mb-4">
            <v-icon color="primary" class="mr-2">mdi-cog</v-icon>
            Processing Dataset
          </v-card-title>

          <!-- Loading State -->
          <v-progress-linear v-if="loading" indeterminate color="primary"></v-progress-linear>

          <!-- Dataset Info -->
          <v-card-text v-if="dataset">
            <v-row>
              <v-col cols="12" md="6">
                <h3 class="mb-2">Dataset Information</h3>
                <p><strong>Filename:</strong> {{ dataset.filename }}</p>
                <p><strong>Task Type:</strong> {{ dataset.task_type }}</p>
                <p><strong>Rows:</strong> {{ dataset.row_count }}</p>
                <p><strong>Columns:</strong> {{ dataset.columns ? dataset.columns.join(', ') : 'N/A' }}</p>
              </v-col>
              <v-col cols="12" md="6">
                <h3 class="mb-2">Processing Options</h3>
                <v-select
                  v-model="trainingConfig.model_type"
                  :items="modelTypes"
                  label="Model Type"
                  outlined
                  dense
                  hint="Transformer models are more accurate but slower"
                  persistent-hint
                ></v-select>
                
                <v-slider
                  v-model="trainingConfig.epochs"
                  :min="1"
                  :max="20"
                  :step="1"
                  label="Training Epochs"
                  thumb-label="always"
                  class="mt-6"
                  hint="More epochs = better accuracy but longer training"
                  persistent-hint
                ></v-slider>
                
                <v-slider
                  v-model="trainingConfig.batch_size"
                  :min="8"
                  :max="64"
                  :step="8"
                  label="Batch Size"
                  thumb-label="always"
                  class="mt-6"
                  hint="Larger batch = faster training but more memory"
                  persistent-hint
                ></v-slider>

                <v-slider
                  v-model="trainingConfig.max_length"
                  :min="32"
                  :max="512"
                  :step="32"
                  label="Max Sequence Length"
                  thumb-label="always"
                  class="mt-6"
                  hint="Longer = more context but slower"
                  persistent-hint
                ></v-slider>
                
                <v-checkbox
                  v-model="trainingConfig.use_validation"
                  label="Use validation split (20%)"
                  hint="Prevents overfitting"
                  persistent-hint
                  class="mt-2"
                ></v-checkbox>

                <v-checkbox
                  v-model="trainingConfig.early_stopping"
                  label="Enable early stopping"
                  hint="Stop training if performance plateaus"
                  persistent-hint
                ></v-checkbox>
              </v-col>
            </v-row>

            <v-divider class="my-4"></v-divider>

            <!-- Action Buttons -->
            <v-row>
              <v-col cols="12">
                <v-btn
                  color="primary"
                  size="large"
                  @click="startTraining"
                  :loading="training"
                  :disabled="training"
                  prepend-icon="mdi-play"
                >
                  Start Training
                </v-btn>
                <v-btn
                  color="secondary"
                  size="large"
                  @click="$router.push('/datasets')"
                  class="ml-2"
                  prepend-icon="mdi-arrow-left"
                >
                  Back to Datasets
                </v-btn>
              </v-col>
            </v-row>

            <!-- Training Progress -->
            <v-card v-if="training || trainingResult" outlined class="mt-4">
              <v-card-title>
                {{ training ? 'Training in Progress...' : 'Training Complete' }}
              </v-card-title>
              <v-card-text>
                <v-progress-linear v-if="training" indeterminate color="primary"></v-progress-linear>
                
                <div v-if="trainingResult">
                  <v-alert :type="trainingResult.success ? 'success' : 'error'" class="mb-4">
                    {{ trainingResult.message }}
                  </v-alert>

                  <div v-if="trainingResult.success">
                    <p><strong>Model ID:</strong> {{ trainingResult.model_id }}</p>
                    <p><strong>Model Type:</strong> {{ trainingResult.model_type }}</p>
                    <p><strong>Training Time:</strong> {{ trainingResult.training_time?.toFixed(2) }}s</p>
                    <p><strong>Status:</strong> {{ trainingResult.status }}</p>

                    <v-btn color="primary" @click="$router.push('/models')" class="mt-4" prepend-icon="mdi-brain">
                      View Model
                    </v-btn>
                  </div>
                </div>
              </v-card-text>
            </v-card>
          </v-card-text>

          <!-- Error State -->
          <v-alert v-if="error" type="error" class="ma-4">
            {{ error }}
          </v-alert>
        </v-card>
      </v-col>
    </v-row>
  </v-container>
</template>

<script>
import api from '@/services/api'

export default {
  name: 'Processing',
  props: {
    datasetId: {
      type: String,
      required: true
    }
  },
  data() {
    return {
      dataset: null,
      loading: false,
      error: null,
      modelTypes: [
        { title: 'Auto (Recommended)', value: null },
        { title: 'Scikit-learn (Fast)', value: 'sklearn' },
        { title: 'Transformer (Accurate)', value: 'transformer' }
      ],
      trainingConfig: {
        model_type: null,
        epochs: 3,
        batch_size: 16,
        max_length: 128,
        use_validation: true,
        early_stopping: false,
        test_size: 0.2
      },
      training: false,
      trainingResult: null
    }
  },
  mounted() {
    this.fetchDataset()
  },
  methods: {
    async fetchDataset() {
      this.loading = true
      this.error = null
      try {
        const response = await api.get(`/upload/datasets/${this.datasetId}`)
        this.dataset = response.data
      } catch (error) {
        this.error = error.response?.data?.detail || 'Failed to fetch dataset'
        console.error('Error fetching dataset:', error)
      } finally {
        this.loading = false
      }
    },

    async startTraining() {
      this.training = true
      this.trainingResult = null
      this.error = null

      try {
        // Send training config as JSON body
        const response = await api.post(`/training/${this.datasetId}`, this.trainingConfig)
        this.trainingResult = response.data
      } catch (error) {
        this.error = error.response?.data?.detail || 'Training failed'
        this.trainingResult = {
          success: false,
          message: this.error
        }
      } finally {
        this.training = false
      }
    }
  }
}
</script>
