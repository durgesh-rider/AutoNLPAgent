import { createStore } from 'vuex'

export default createStore({
  state: {
    datasets: [],
    models: [],
    currentDataset: null,
    currentModel: null,
    loading: false,
    error: null
  },
  mutations: {
    SET_DATASETS(state, datasets) {
      state.datasets = datasets
    },
    ADD_DATASET(state, dataset) {
      state.datasets.push(dataset)
    },
    SET_MODELS(state, models) {
      state.models = models
    },
    ADD_MODEL(state, model) {
      state.models.push(model)
    },
    SET_CURRENT_DATASET(state, dataset) {
      state.currentDataset = dataset
    },
    SET_CURRENT_MODEL(state, model) {
      state.currentModel = model
    },
    SET_LOADING(state, loading) {
      state.loading = loading
    },
    SET_ERROR(state, error) {
      state.error = error
    }
  },
  actions: {
    async fetchDatasets({ commit }) {
      commit('SET_LOADING', true)
      try {
        // API call would go here
        // const response = await api.get('/datasets')
        // commit('SET_DATASETS', response.data.datasets)
        commit('SET_ERROR', null)
      } catch (error) {
        commit('SET_ERROR', error.message)
      } finally {
        commit('SET_LOADING', false)
      }
    },

    async fetchModels({ commit }) {
      commit('SET_LOADING', true)
      try {
        // API call would go here
        // const response = await api.get('/models')
        // commit('SET_MODELS', response.data.models)
        commit('SET_ERROR', null)
      } catch (error) {
        commit('SET_ERROR', error.message)
      } finally {
        commit('SET_LOADING', false)
      }
    }
  },
  getters: {
    getDatasetById: (state) => (id) => {
      return state.datasets.find(dataset => dataset.id === id)
    },
    getModelById: (state) => (id) => {
      return state.models.find(model => model.id === id)
    }
  }
})