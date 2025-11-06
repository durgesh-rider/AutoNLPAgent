<template>
  <v-app>
    <!-- Enhanced App Bar -->
    <v-app-bar 
      color="primary" 
      dark 
      elevation="4"
      prominent
    >
      <v-app-bar-nav-icon @click="drawer = !drawer"></v-app-bar-nav-icon>
      
      <v-toolbar-title class="text-h5 font-weight-bold">
        <v-icon icon="mdi-robot" class="mr-2"></v-icon>
        AutoNLP-Agent
      </v-toolbar-title>

      <v-spacer></v-spacer>

      <!-- Quick Action Buttons -->
      <v-btn 
        to="/upload" 
        variant="outlined" 
        prepend-icon="mdi-upload"
        class="mr-2"
      >
        Upload
      </v-btn>

    </v-app-bar>

    <!-- Enhanced Navigation Drawer -->
    <v-navigation-drawer 
      v-model="drawer"
      :rail="rail"
      permanent
      @click="rail = false"
    >
      <v-list-item
        prepend-icon="mdi-robot"
        title="AutoNLP Agent"
        nav
      >
        <template v-slot:append>
          <v-btn
            variant="text"
            icon="mdi-chevron-left"
            @click.stop="rail = !rail"
          ></v-btn>
        </template>
      </v-list-item>

      <v-divider></v-divider>

      <v-list density="compact" nav>
        <v-list-item 
          link 
          to="/" 
          prepend-icon="mdi-view-dashboard" 
          title="Dashboard"
          value="home"
        ></v-list-item>
        
        <v-list-item 
          link 
          to="/upload" 
          prepend-icon="mdi-cloud-upload" 
          title="Upload Dataset"
          value="upload"
        ></v-list-item>
        
        <v-list-item 
          link 
          to="/datasets" 
          prepend-icon="mdi-database-outline" 
          title="My Datasets"
          value="datasets"
        ></v-list-item>

        <v-list-item 
          link 
          to="/models" 
          prepend-icon="mdi-brain" 
          title="My Models"
          value="models"
        ></v-list-item>

        <v-divider class="my-2"></v-divider>

        <v-list-subheader>Quick Actions</v-list-subheader>
        
        <v-list-item 
          prepend-icon="mdi-help-circle-outline" 
          title="Documentation"
          value="docs"
        ></v-list-item>
        
        <v-list-item 
          prepend-icon="mdi-github" 
          title="GitHub"
          value="github"
        ></v-list-item>
      </v-list>
    </v-navigation-drawer>

    <!-- Main Content Area -->
    <v-main class="bg-grey-lighten-4">
      <v-container fluid>
        <router-view />
      </v-container>
    </v-main>

    <!-- Footer -->
    <v-footer app color="grey-darken-3" dark class="pa-3">
      <v-row align="center" justify="center">
        <v-col cols="auto">
          <span class="text-caption">
            © {{ new Date().getFullYear() }} AutoNLP-Agent | Powered by AI
          </span>
        </v-col>
      </v-row>
    </v-footer>
  </v-app>
</template>

<script>
export default {
  name: 'App',
  data: () => ({
    drawer: true,
    rail: false
  })
}
</script>

<style scoped>
.v-app-bar {
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.v-main {
  min-height: calc(100vh - 64px - 64px);
}
</style>