<template>
  <v-container>
    <v-row>
      <v-col cols="12">
        <v-card class="pa-6">
          <v-card-title class="text-h4 mb-4">
            <v-icon color="primary" class="mr-2">mdi-database</v-icon>
            My Datasets
          </v-card-title>

          <!-- Loading State -->
          <v-progress-linear v-if="loading" indeterminate color="primary"></v-progress-linear>

          <!-- Error State -->
          <v-alert v-if="error" type="error" dismissible @click:close="error = null" class="mb-4">
            {{ error }}
          </v-alert>

          <!-- Empty State -->
          <v-card-text v-if="!loading && datasets.length === 0">
            <div class="text-center py-8">
              <v-icon size="100" color="grey">mdi-database-off</v-icon>
              <p class="text-h6 mt-4">No datasets uploaded yet</p>
              <p class="text-body-1 text-grey">Upload your first dataset to get started</p>
              <v-btn color="primary" to="/upload" class="mt-4" prepend-icon="mdi-upload">
                Upload Dataset
              </v-btn>
            </div>
          </v-card-text>

          <!-- Datasets List -->
          <v-card-text v-else>
            <v-row>
              <v-col v-for="dataset in datasets" :key="dataset.dataset_id" cols="12" md="6" lg="4">
                <v-card outlined hover>
                  <v-card-title>
                    <v-icon color="primary" class="mr-2">mdi-file-table</v-icon>
                    {{ dataset.filename }}
                  </v-card-title>
                  <v-card-text>
                    <p><strong>Task Type:</strong> {{ dataset.task_type || 'Unknown' }}</p>
                    <p><strong>Rows:</strong> {{ dataset.row_count }}</p>
                    <p><strong>Columns:</strong> {{ dataset.columns ? dataset.columns.length : 0 }}</p>
                    <v-chip-group v-if="dataset.columns" class="mt-2">
                      <v-chip v-for="col in dataset.columns.slice(0, 3)" :key="col" small>
                        {{ col }}
                      </v-chip>
                      <v-chip v-if="dataset.columns.length > 3" small>
                        +{{ dataset.columns.length - 3 }} more
                      </v-chip>
                    </v-chip-group>
                  </v-card-text>
                  <v-card-actions>
                    <v-btn color="primary" @click="trainModel(dataset.dataset_id)" prepend-icon="mdi-brain">
                      Train Model
                    </v-btn>
                    <v-spacer></v-spacer>
                    <v-btn icon="mdi-delete" color="error" @click="deleteDataset(dataset.dataset_id)"></v-btn>
                  </v-card-actions>
                </v-card>
              </v-col>
            </v-row>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>

    <!-- Training Dialog -->
    <v-dialog v-model="trainingDialog" max-width="500">
      <v-card>
        <v-card-title>Train Model</v-card-title>
        <v-card-text>
          <v-progress-circular v-if="training" indeterminate color="primary"></v-progress-circular>
          <p v-else-if="trainingResult">{{ trainingResult.message }}</p>
          <p v-else>Starting model training...</p>
        </v-card-text>
        <v-card-actions>
          <v-spacer></v-spacer>
          <v-btn v-if="trainingResult && trainingResult.success" color="primary" @click="goToModels">
            View Models
          </v-btn>
          <v-btn @click="trainingDialog = false">Close</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </v-container>
</template>

<script>
import api from '@/services/api'

export default {
  name: 'Datasets',
  data() {
    return {
      datasets: [],
      loading: false,
      error: null,
      trainingDialog: false,
      training: false,
      trainingResult: null
    }
  },
  mounted() {
    this.fetchDatasets()
  },
  methods: {
    async fetchDatasets() {
      this.loading = true
      this.error = null
      try {
        const response = await api.get('/upload/datasets')
        this.datasets = response.data.datasets || []
      } catch (error) {
        this.error = error.response?.data?.detail || 'Failed to fetch datasets'
        console.error('Error fetching datasets:', error)
      } finally {
        this.loading = false
      }
    },

    async trainModel(datasetId) {
      // Navigate to training configuration page
      this.$router.push({
        name: 'Training',
        params: { datasetId: datasetId }
      })
    },

    async deleteDataset(datasetId) {
      if (!confirm('Are you sure you want to delete this dataset?')) return

      try {
        await api.delete(`/upload/datasets/${datasetId}`)
        this.datasets = this.datasets.filter(d => d.dataset_id !== datasetId)
      } catch (error) {
        this.error = error.response?.data?.detail || 'Failed to delete dataset'
      }
    },

    goToModels() {
      this.trainingDialog = false
      this.$router.push('/models')
    }
  }
}
</script>
