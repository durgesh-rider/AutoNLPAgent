<template>
  <v-container class="pa-6">
    <v-row>
      <v-col cols="12">
        <h1 class="text-h4 font-weight-bold mb-6">Upload Dataset</h1>
      </v-col>
    </v-row>

    <v-row>
      <v-col cols="12" md="8">
        <v-card elevation="2">
          <v-card-text>
            <div
              class="drop-zone"
              :class="{ 'drop-zone-active': isDragging, 'drop-zone-has-file': selectedFile }"
              @drop.prevent="handleDrop"
              @dragover.prevent="isDragging = true"
              @dragleave.prevent="isDragging = false"
              @click="triggerFileInput"
            >
              <input
                ref="fileInput"
                type="file"
                style="display: none"
                accept=".csv,.txt,.xlsx,.xls"
                @change="handleFileSelect"
              />
              <v-icon size="64" color="primary">
                {{ selectedFile ? 'mdi-file-check' : 'mdi-cloud-upload' }}
              </v-icon>
              <h3 class="text-h5 mt-4 mb-2">
                {{ selectedFile ? selectedFile.name : 'Drop your file here' }}
              </h3>
              <p class="text-body-2 text-medium-emphasis">
                {{ selectedFile ? `Size: ${(selectedFile.size / 1024).toFixed(2)} KB` : 'or click to browse' }}
              </p>
              <v-btn
                v-if="selectedFile"
                color="error"
                variant="text"
                @click.stop="clearFile"
              >
                <v-icon left>mdi-delete</v-icon>
                Remove
              </v-btn>
            </div>

            <div v-if="selectedFile" class="mt-4">
              <v-btn
                color="primary"
                :loading="uploading"
                :disabled="uploading"
                block
                @click="uploadFile"
              >
                <v-icon left>mdi-upload</v-icon>
                Upload File
              </v-btn>
            </div>
          </v-card-text>
        </v-card>

        <v-expand-transition>
          <v-alert
            v-if="uploadResult"
            :type="uploadResult.success ? 'success' : 'error'"
            :icon="uploadResult.success ? 'mdi-check-circle' : 'mdi-alert-circle'"
            prominent
            class="mt-6"
          >
            <v-alert-title>
              {{ uploadResult.success ? 'Upload Successful!' : 'Upload Failed' }}
            </v-alert-title>
            <div v-if="uploadResult.success" class="mt-4">
              <v-row>
                <v-col cols="12" sm="6">
                  <div class="text-caption text-medium-emphasis">Dataset ID</div>
                  <div class="text-body-1 font-weight-medium">{{ uploadResult.data.dataset_id }}</div>
                </v-col>
                <v-col cols="12" sm="6">
                  <div class="text-caption text-medium-emphasis">Detected Task</div>
                  <div class="text-body-1 font-weight-medium">{{ uploadResult.data.task_type }}</div>
                </v-col>
              </v-row>
              
              <v-divider class="my-4"></v-divider>
              
              <div class="text-body-2 mb-3">What would you like to do next?</div>
              <v-row>
                <v-col cols="12" sm="6">
                  <v-btn
                    color="primary"
                    block
                    variant="elevated"
                    @click="goToTraining"
                  >
                    <v-icon left>mdi-brain</v-icon>
                    Configure & Train Model
                  </v-btn>
                </v-col>
                <v-col cols="12" sm="6">
                  <v-btn
                    color="secondary"
                    block
                    variant="outlined"
                    @click="$router.push('/datasets')"
                  >
                    <v-icon left>mdi-database</v-icon>
                    View Datasets
                  </v-btn>
                </v-col>
              </v-row>
            </div>
            <div v-else class="mt-2">
              {{ uploadResult.error }}
            </div>
          </v-alert>
        </v-expand-transition>
      </v-col>

      <v-col cols="12" md="4">
        <v-card elevation="2">
          <v-card-title class="bg-blue-lighten-4">
            <v-icon class="mr-2">mdi-information</v-icon>
            Supported Formats
          </v-card-title>
          <v-card-text class="pa-0">
            <v-list>
              <v-list-item>
                <v-list-item-title>CSV Files</v-list-item-title>
                <v-list-item-subtitle>.csv</v-list-item-subtitle>
              </v-list-item>
              <v-list-item>
                <v-list-item-title>Text Files</v-list-item-title>
                <v-list-item-subtitle>.txt</v-list-item-subtitle>
              </v-list-item>
              <v-list-item>
                <v-list-item-title>Excel Files</v-list-item-title>
                <v-list-item-subtitle>.xlsx, .xls</v-list-item-subtitle>
              </v-list-item>
            </v-list>
          </v-card-text>
        </v-card>

        <v-card elevation="2" class="mt-4">
          <v-card-title class="bg-orange-lighten-4">
            <v-icon class="mr-2">mdi-lightbulb-outline</v-icon>
            Tips
          </v-card-title>
          <v-card-text class="pa-4">
            <v-list density="compact">
              <v-list-item>
                <template v-slot:prepend>
                  <v-icon color="orange" size="small">mdi-check-circle</v-icon>
                </template>
                <v-list-item-title class="text-body-2">
                  At least 10 samples
                </v-list-item-title>
              </v-list-item>
              <v-list-item>
                <template v-slot:prepend>
                  <v-icon color="orange" size="small">mdi-check-circle</v-icon>
                </template>
                <v-list-item-title class="text-body-2">
                  Include column headers
                </v-list-item-title>
              </v-list-item>
              <v-list-item>
                <template v-slot:prepend>
                  <v-icon color="orange" size="small">mdi-check-circle</v-icon>
                </template>
                <v-list-item-title class="text-body-2">
                  UTF-8 encoding recommended
                </v-list-item-title>
              </v-list-item>
            </v-list>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>
  </v-container>
</template>

<script>
import api from '@/services/api'

export default {
  name: 'Upload',
  data() {
    return {
      selectedFile: null,
      isDragging: false,
      uploading: false,
      uploadResult: null
    }
  },
  methods: {
    triggerFileInput() {
      this.$refs.fileInput.click()
    },
    handleFileSelect(event) {
      const files = event.target.files
      if (files.length > 0) {
        this.selectedFile = files[0]
        this.uploadResult = null
      }
    },
    handleDrop(event) {
      this.isDragging = false
      const files = event.dataTransfer.files
      if (files.length > 0) {
        this.selectedFile = files[0]
        this.uploadResult = null
      }
    },
    clearFile() {
      this.selectedFile = null
      this.uploadResult = null
      if (this.$refs.fileInput) {
        this.$refs.fileInput.value = ''
      }
    },
    async uploadFile() {
      if (!this.selectedFile) return

      this.uploading = true
      this.uploadResult = null

      try {
        const formData = new FormData()
        formData.append('file', this.selectedFile)

        const response = await api.post('/upload', formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        })

        this.uploadResult = {
          success: true,
          data: response.data
        }
      } catch (error) {
        this.uploadResult = {
          success: false,
          error: error.response?.data?.detail || error.message || 'Upload failed'
        }
      } finally {
        this.uploading = false
      }
    },
    
    goToTraining() {
      if (this.uploadResult && this.uploadResult.data && this.uploadResult.data.dataset_id) {
        this.$router.push({
          name: 'Training',
          params: { datasetId: this.uploadResult.data.dataset_id }
        })
      }
    }
  }
}
</script>

<style scoped>
.drop-zone {
  border: 2px dashed #ccc;
  border-radius: 8px;
  padding: 48px;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s;
}

.drop-zone:hover {
  border-color: #1976d2;
  background: #f5f5f5;
}

.drop-zone-active {
  border-color: #1976d2;
  background: #e3f2fd;
}

.drop-zone-has-file {
  border-color: #4caf50;
  background: #f1f8f4;
}
</style>
