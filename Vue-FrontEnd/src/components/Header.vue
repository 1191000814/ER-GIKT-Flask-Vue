<!-- 页面的头部 -->
<template>
  <div style="height: 80px; line-height: 80px; border-bottom: 1px solid #150101; display: flex; font-size: x-large;
       background-color:rgba(51,51,51,1)">
    <div><img src="@/assets/hit_logo.png" style="height: 60px; margin: 10px 0 0 15px;filter: brightness(100)" alt="hit-logo"/></div>
    <div style="width: 600px; margin-left: 10px; font-weight: bold; color: white;">
      深度知识追踪模型分析与习题推荐系统
    </div>
    <div style="flex: 1"></div>
    <div style="font-weight: bold; color: white">你好，<span style="color: yellow;">{{user.username}}</span>&nbsp;&nbsp;</div>
    <div style="width: 100px; padding-top: 0">
      <el-dropdown @command="handleCommand" style="margin-top: 0;">
        <el-button size="120" icon="el-icon-user" circle></el-button>
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