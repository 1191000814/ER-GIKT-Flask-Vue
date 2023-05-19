const user = {
  // 要管理的状态
  state:{
    user_id: null
  },

  // 定义待发送的事件，第一个参数会注入当前的state
  mutations:{
    set_user(state, user_id){
      state.user_id = user_id
    }
  },
  actions:{
    saveUserInfo({ commit },data){
      commit('USER_INFO', data)
    }
  }
};
export default user