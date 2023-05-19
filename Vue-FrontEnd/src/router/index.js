import Vue from 'vue';
import VueRouter from 'vue-router';
import userData from '../views/data/UserData.vue';
import login from "@/views/Login.vue";
import Home from "@/views/data/Home.vue";
import layout from "@/layout/layout";
import QuestionData from "@/views/data/QuestionData.vue";
import SkillData from "@/views/data/SkillData.vue";
import History from "@/views/kt/History.vue";
import Predict from "@/views/kt/Predict.vue";
import Recommend from "@/views/kt/Recommend.vue";
import SkillGraph from "@/views/kt/SkillGraph.vue";

Vue.use(VueRouter)

// 定义各个页面的路由
const routes = [
  {
    path: '/',
    name: 'layout',
    component: layout,
    redirect: "login", // 主页重定向到登录页面
    children: [
      {
        path: 'home',
        name: 'home',
        component: Home
      },
      // 数据分析
      {
        path: 'history',
        name: 'history',
        component: History
      },
      {
        path: 'predict',
        name: 'predict',
        component: Predict
      },
      {
        path: 'recommend',
        name: 'recommend',
        component: Recommend
      },
      {
        path: 'skillGraph',
        name: 'skillGraph',
        component: SkillGraph
      },
      // 数据管理
      {
        path: 'userData',
        name: 'userData',
        component: userData
      },
      {
        path: 'questionData',
        name: 'questionData',
        component: QuestionData
      },
      {
        path: 'skillData',
        name: 'skillData',
        component: SkillData
      }
    ]
  },
  {
    path: '/login',
    name: 'login',
    component: login
  },
]

const router = new VueRouter({
  mode: 'history',
  base: process.env.BASE_URL,
  routes
})

// 全局前置守卫
router.beforeEach((to, from, next) => {
  const isLoggedIn = sessionStorage.getItem('user') != null
  if (to.path !== '/login' && !isLoggedIn) {
    next('/login')
  } else {
    next()
  }
})

export default router
