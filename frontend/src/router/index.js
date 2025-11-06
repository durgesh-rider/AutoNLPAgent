import { createRouter, createWebHistory } from 'vue-router'
import Home from '../views/Home.vue'
import Upload from '../views/Upload.vue'
import Datasets from '../views/Datasets.vue'
import Models from '../views/Models.vue'
import Processing from '../views/Processing.vue'
import Training from '../views/Training.vue'
import Evaluation from '../views/Evaluation.vue'

const routes = [
  {
    path: '/',
    name: 'Home',
    component: Home
  },
  {
    path: '/upload',
    name: 'Upload',
    component: Upload
  },
  {
    path: '/datasets',
    name: 'Datasets',
    component: Datasets
  },
  {
    path: '/models',
    name: 'Models',
    component: Models
  },
  {
    path: '/training/:datasetId',
    name: 'Training',
    component: Training,
    props: true
  },
  {
    path: '/evaluation/:modelId',
    name: 'Evaluation',
    component: Evaluation,
    props: true
  },
  {
    path: '/processing/:datasetId',
    name: 'Processing',
    component: Processing,
    props: true
  },
  {
    path: '/results/:modelId',
    name: 'Results',
    component: Models, // Redirect to models page
    props: true
  }
]

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes
})

export default router