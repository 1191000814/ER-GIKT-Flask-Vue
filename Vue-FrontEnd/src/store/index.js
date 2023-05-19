import Vue from 'vue'
import Vuex from 'vuex'
import user from "@/store/user";

Vue.use(Vuex)

export default new Vuex.Store({
  // 需要的store模块
  modules: {
    user
  }
})