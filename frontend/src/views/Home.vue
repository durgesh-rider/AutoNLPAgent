<template>
  <v-container fluid class="pa-0">
    <v-row no-gutters>
      <v-col cols="12">
        <v-card class="hero-section" flat tile>
          <v-card-text class="text-center py-12">
            <h1 class="text-h2 font-weight-bold mb-4">AutoNLP Platform</h1>
            <p class="text-h5 mb-6">Automated Natural Language Processing Made Easy</p>
            <v-btn color="white" size="x-large" to="/upload" variant="elevated">
              Get Started
              <v-icon end>mdi-arrow-right</v-icon>
            </v-btn>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>

    <v-container class="my-12">
      <v-row>
        <v-col cols="12" md="4">
          <v-card elevation="2" class="h-100">
            <v-card-text class="text-center pa-6">
              <v-icon size="64" color="primary">mdi-upload</v-icon>
              <h3 class="text-h5 my-4">Upload Data</h3>
              <p>Simply upload your dataset and let our system handle the rest</p>
            </v-card-text>
          </v-card>
        </v-col>
        <v-col cols="12" md="4">
          <v-card elevation="2" class="h-100">
            <v-card-text class="text-center pa-6">
              <v-icon size="64" color="primary">mdi-auto-fix</v-icon>
              <h3 class="text-h5 my-4">Auto Training</h3>
              <p>Automatic model selection and training with optimal parameters</p>
            </v-card-text>
          </v-card>
        </v-col>
        <v-col cols="12" md="4">
          <v-card elevation="2" class="h-100">
            <v-card-text class="text-center pa-6">
              <v-icon size="64" color="primary">mdi-download</v-icon>
              <h3 class="text-h5 my-4">Export Models</h3>
              <p>Download your trained models and deploy them anywhere</p>
            </v-card-text>
          </v-card>
        </v-col>
      </v-row>
    </v-container>
  </v-container>
</template>

<script>
export default {
  name: 'Home'
}
</script>

<style scoped>
.hero-section {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}
.h-100 {
  height: 100%;
}
</style>
