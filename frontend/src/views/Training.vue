<template>
  <v-container class="pa-6">
    <v-row>
      <v-col cols="12">
        <v-btn text @click="$router.back()" class="mb-4">
          <v-icon left>mdi-arrow-left</v-icon>
          Back
        </v-btn>
        <h1 class="text-h4 font-weight-bold mb-2">Model Training Configuration</h1>
        <p class="text-subtitle-1 text-grey-darken-1 mb-6">
          Configure and train your model with custom parameters
        </p>
      </v-col>
    </v-row>

    <!-- Dataset Information -->
    <v-row v-if="datasetInfo">
      <v-col cols="12">
        <v-card elevation="2" class="mb-4">
          <v-card-title class="bg-blue-lighten-5">
            <v-icon class="mr-2">mdi-database</v-icon>
            Dataset Information
          </v-card-title>
          <v-card-text>
            <v-row>
              <v-col cols="12" md="3">
                <div class="text-caption text-grey-darken-1">Dataset ID</div>
                <div class="text-body-1 font-weight-medium">{{ datasetId }}</div>
              </v-col>
              <v-col cols="12" md="3">
                <div class="text-caption text-grey-darken-1">Task Type</div>
                <div class="text-body-1 font-weight-medium">
                  <v-chip color="primary" size="small">{{ datasetInfo.task_type || 'Auto-detected' }}</v-chip>
                </div>
              </v-col>
              <v-col cols="12" md="3">
                <div class="text-caption text-grey-darken-1">Rows</div>
                <div class="text-body-1 font-weight-medium">{{ datasetInfo.row_count || 'N/A' }}</div>
              </v-col>
              <v-col cols="12" md="3">
                <div class="text-caption text-grey-darken-1">Columns</div>
                <div class="text-body-1 font-weight-medium">{{ datasetInfo.column_count || 'N/A' }}</div>
              </v-col>
            </v-row>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>

    <!-- Training Configuration -->
    <v-row>
      <v-col cols="12" md="8">
        <v-card elevation="2">
          <v-card-title class="bg-green-lighten-5">
            <v-icon class="mr-2">mdi-tune</v-icon>
            Training Parameters
          </v-card-title>
          <v-card-text class="pa-6">
            <!-- Model Selection -->
            <v-select
              v-model="config.model_name"
              :items="availableModels"
              label="Model"
              prepend-icon="mdi-brain"
              variant="outlined"
              hint="Select the pre-trained model to use"
              persistent-hint
              class="mb-4"
            ></v-select>

            <!-- Epochs -->
            <div class="mb-4">
              <label class="text-subtitle-2 mb-2 d-block">Epochs: {{ config.epochs }}</label>
              <v-slider
                v-model="config.epochs"
                :min="1"
                :max="20"
                :step="1"
                thumb-label
                color="primary"
              ></v-slider>
              <div class="text-caption text-grey-darken-1">Number of training iterations over the dataset</div>
            </div>

            <!-- Batch Size -->
            <div class="mb-4">
              <label class="text-subtitle-2 mb-2 d-block">Batch Size: {{ config.batch_size }}</label>
              <v-slider
                v-model="config.batch_size"
                :min="4"
                :max="64"
                :step="4"
                thumb-label
                color="primary"
              ></v-slider>
              <div class="text-caption text-grey-darken-1">Number of samples processed together</div>
            </div>

            <!-- Learning Rate -->
            <v-text-field
              v-model.number="config.learning_rate"
              label="Learning Rate"
              type="number"
              step="0.00001"
              prepend-icon="mdi-gauge"
              variant="outlined"
              hint="Controls how quickly the model adapts (e.g., 0.00002)"
              persistent-hint
              class="mb-4"
            ></v-text-field>

            <!-- Max Length -->
            <div class="mb-4">
              <label class="text-subtitle-2 mb-2 d-block">Max Sequence Length: {{ config.max_length }}</label>
              <v-slider
                v-model="config.max_length"
                :min="64"
                :max="512"
                :step="64"
                thumb-label
                color="primary"
              ></v-slider>
              <div class="text-caption text-grey-darken-1">Maximum number of tokens per input</div>
            </div>

            <!-- Test Split -->
            <div class="mb-4">
              <label class="text-subtitle-2 mb-2 d-block">Test Split: {{ (config.test_split * 100).toFixed(0) }}%</label>
              <v-slider
                v-model="config.test_split"
                :min="0.1"
                :max="0.4"
                :step="0.05"
                thumb-label
                color="primary"
              ></v-slider>
              <div class="text-caption text-grey-darken-1">Percentage of data reserved for testing</div>
            </div>

            <!-- Advanced Options Toggle -->
            <v-expansion-panels class="mb-4">
              <v-expansion-panel>
                <v-expansion-panel-title>
                  <v-icon class="mr-2">mdi-cog</v-icon>
                  Advanced Options
                </v-expansion-panel-title>
                <v-expansion-panel-text>
                  <v-text-field
                    v-model.number="config.warmup_steps"
                    label="Warmup Steps"
                    type="number"
                    prepend-icon="mdi-fire"
                    variant="outlined"
                    hint="Number of steps for learning rate warmup"
                    persistent-hint
                    class="mb-4"
                  ></v-text-field>

                  <v-text-field
                    v-model.number="config.weight_decay"
                    label="Weight Decay"
                    type="number"
                    step="0.001"
                    prepend-icon="mdi-weight"
                    variant="outlined"
                    hint="L2 regularization factor (e.g., 0.01)"
                    persistent-hint
                    class="mb-4"
                  ></v-text-field>

                  <v-switch
                    v-model="config.fp16"
                    label="Use Mixed Precision (FP16)"
                    color="primary"
                    hint="Faster training with lower memory usage (requires GPU)"
                    persistent-hint
                  ></v-switch>
                </v-expansion-panel-text>
              </v-expansion-panel>
            </v-expansion-panels>

            <!-- Action Buttons -->
            <v-divider class="my-6"></v-divider>
            
            <v-row>
              <v-col cols="12" md="6">
                <v-btn
                  block
                  variant="outlined"
                  color="grey"
                  @click="resetToDefaults"
                >
                  <v-icon left>mdi-restore</v-icon>
                  Reset to Defaults
                </v-btn>
              </v-col>
              <v-col cols="12" md="6">
                <v-btn
                  block
                  color="primary"
                  size="large"
                  :loading="training"
                  :disabled="training"
                  @click="startTraining"
                >
                  <v-icon left>mdi-play</v-icon>
                  Start Training
                </v-btn>
              </v-col>
            </v-row>
          </v-card-text>
        </v-card>
      </v-col>

      <!-- Training Info Sidebar -->
      <v-col cols="12" md="4">
        <v-card elevation="2" class="mb-4">
          <v-card-title class="bg-orange-lighten-5">
            <v-icon class="mr-2">mdi-information</v-icon>
            Parameter Guide
          </v-card-title>
          <v-card-text>
            <v-list density="compact">
              <v-list-item>
                <template v-slot:prepend>
                  <v-icon color="orange" size="small">mdi-circle-small</v-icon>
                </template>
                <v-list-item-title class="text-body-2">
                  <strong>Epochs:</strong> More epochs = better learning but risk overfitting
                </v-list-item-title>
              </v-list-item>
              <v-list-item>
                <template v-slot:prepend>
                  <v-icon color="orange" size="small">mdi-circle-small</v-icon>
                </template>
                <v-list-item-title class="text-body-2">
                  <strong>Batch Size:</strong> Larger = faster but needs more memory
                </v-list-item-title>
              </v-list-item>
              <v-list-item>
                <template v-slot:prepend>
                  <v-icon color="orange" size="small">mdi-circle-small</v-icon>
                </template>
                <v-list-item-title class="text-body-2">
                  <strong>Learning Rate:</strong> Too high = unstable, too low = slow
                </v-list-item-title>
              </v-list-item>
              <v-list-item>
                <template v-slot:prepend>
                  <v-icon color="orange" size="small">mdi-circle-small</v-icon>
                </template>
                <v-list-item-title class="text-body-2">
                  <strong>Max Length:</strong> Longer sequences need more memory
                </v-list-item-title>
              </v-list-item>
            </v-list>
          </v-card-text>
        </v-card>

        <v-card elevation="2">
          <v-card-title class="bg-purple-lighten-5">
            <v-icon class="mr-2">mdi-chart-line</v-icon>
            Expected Results
          </v-card-title>
          <v-card-text>
            <div class="text-body-2 mb-3">
              After training completes, you'll get:
            </div>
            <v-list density="compact">
              <v-list-item>
                <template v-slot:prepend>
                  <v-icon color="success" size="small">mdi-check</v-icon>
                </template>
                <v-list-item-title class="text-body-2">Model accuracy metrics</v-list-item-title>
              </v-list-item>
              <v-list-item>
                <template v-slot:prepend>
                  <v-icon color="success" size="small">mdi-check</v-icon>
                </template>
                <v-list-item-title class="text-body-2">Training loss curves</v-list-item-title>
              </v-list-item>
              <v-list-item>
                <template v-slot:prepend>
                  <v-icon color="success" size="small">mdi-check</v-icon>
                </template>
                <v-list-item-title class="text-body-2">Evaluation results</v-list-item-title>
              </v-list-item>
              <v-list-item>
                <template v-slot:prepend>
                  <v-icon color="success" size="small">mdi-check</v-icon>
                </template>
                <v-list-item-title class="text-body-2">Downloadable model</v-list-item-title>
              </v-list-item>
            </v-list>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>

    <!-- Training Progress Dialog -->
    <v-dialog v-model="trainingDialog" max-width="600" persistent>
      <v-card>
        <v-card-title class="bg-primary text-white">
          <v-icon class="mr-2" color="white">mdi-cog</v-icon>
          Training in Progress
        </v-card-title>
        <v-card-text class="pa-6">
          <div class="text-center">
            <v-progress-circular
              :size="80"
              :width="8"
              color="primary"
              indeterminate
              class="mb-4"
            ></v-progress-circular>
            <div class="text-h6 mb-2">{{ trainingStatus }}</div>
            <div class="text-body-2 text-grey-darken-1">
              This may take several minutes depending on your dataset size and configuration.
            </div>
          </div>
        </v-card-text>
      </v-card>
    </v-dialog>
  </v-container>
</template>

<script>
import api from '@/services/api'

export default {
  name: 'Training',
  props: {
    datasetId: {
      type: String,
      required: true
    }
  },
  data() {
    return {
      datasetInfo: null,
      training: false,
      trainingDialog: false,
      trainingStatus: 'Initializing training...',
      availableModels: [
        'bert-base-uncased',
        'distilbert-base-uncased',
        'roberta-base',
        'albert-base-v2',
        'xlnet-base-cased'
      ],
      config: {
        model_name: 'distilbert-base-uncased',
        epochs: 3,
        batch_size: 16,
        learning_rate: 0.00002,
        max_length: 128,
        test_split: 0.2,
        warmup_steps: 500,
        weight_decay: 0.01,
        fp16: false
      },
      defaultConfig: null
    }
  },
  mounted() {
    this.defaultConfig = { ...this.config }
    this.loadDatasetInfo()
  },
  methods: {
    async loadDatasetInfo() {
      try {
        const response = await api.get(`/upload/datasets/${this.datasetId}`)
        this.datasetInfo = response.data
      } catch (error) {
        console.error('Failed to load dataset info:', error)
      }
    },
    
    resetToDefaults() {
      this.config = { ...this.defaultConfig }
    },

    async startTraining() {
      this.training = true
      this.trainingDialog = true
      this.trainingStatus = 'Starting training process...'

      try {
        // Map frontend config to backend format
        const backendConfig = {
          model_type: 'transformer', // Use transformer models
          epochs: this.config.epochs,
          batch_size: this.config.batch_size,
          learning_rate: this.config.learning_rate,
          max_length: this.config.max_length,
          test_size: this.config.test_split,
          use_validation: true,
          early_stopping: false
        }
        
        // Call training API with configuration
        const response = await api.post(`/training/${this.datasetId}`, backendConfig)
        
        this.trainingStatus = 'Training completed successfully!'
        
        // Wait a moment to show success message
        await new Promise(resolve => setTimeout(resolve, 1500))
        
        // Navigate to evaluation page with model ID
        this.$router.push({
          name: 'Evaluation',
          params: { modelId: response.data.model_id }
        })
      } catch (error) {
        this.trainingDialog = false
        alert(error.response?.data?.detail || 'Training failed. Please try again.')
      } finally {
        this.training = false
      }
    }
  }
}
</script>

<style scoped>
.v-slider {
  margin-top: 8px;
}
</style>
