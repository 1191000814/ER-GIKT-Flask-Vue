<!-- 页面的头部 -->
<template>
  <div style="height: 80px; line-height: 80px; border-bottom: 1px solid #150101; display: flex; font-size: x-large;
       background-color:rgba(51,51,51,1)">
    <div><img src="@/assets/hit_logo.png" style="height: 60px; margin: 10px 0 0 15px;filter: brightness(100)" alt="hit-logo"/></div>
    <div style="width: 600px; padding-left: 30px; font-weight: bold; color: #ffffff;">
      知识追踪模型分析与习题推荐系统
    </div>
    <div style="flex: 1; background-color: unset"></div>
    <div style="font-weight: bold; color: white">你好，<span style="color: yellow;">{{user.username}}</span>&nbsp;&nbsp;</div>
    <div style="width: 100px; background-color: unset">
      <el-dropdown @command="handleCommand">
        <el-button size="80" icon="el-icon-user" circle></el-button>
        <el-dropdown-menu slot="default">
          <el-dropdown-item>
            <router-link to="home" class="myLink">返回主页</router-link>
          </el-dropdown-item>
          <el-dropdown-item command="out">退出系统</el-dropdown-item>
        </el-dropdown-menu>
      </el-dropdown>
    </div>
  </div>
</template>

<script>

let userStr = sessionStorage.getItem("user")
export default {
  name: "Header",
  data(){
    return{
      // console.log(userStr)
      user: JSON.parse(userStr)
    }
  },
  methods: {
    handleCommand(command) {
      if (command === "out")
        this.$router.push('/login')
    }
  },
  created() {
    // console.log('进入主页')
    let userStr = sessionStorage.getItem("user")
    // console.log(userStr)
    this.user = JSON.parse(userStr)
  }
}
</script>

<style scoped></style>