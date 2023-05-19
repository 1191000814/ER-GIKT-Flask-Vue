import Vue from 'vue'
import App from './App.vue'
import router from './router'
import store from './store'
import ElementUI from 'element-ui'
import 'element-ui/lib/theme-chalk/index.css'
import locale from "element-ui/src/mixins/locale"
import * as echarts from 'echarts'
import './assets/css/global.css'

Vue.prototype.$echarts = echarts
Vue.config.productionTip = false
Vue.use(ElementUI, { locale, size: 'small' });

const app = new Vue({
  router,
  store,
  render: h => h(App)
}).$mount('#app')