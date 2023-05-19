<!-- 页面的侧边栏（菜单栏） -->
<template>
  <div>
    <el-menu style="width: 220px; min-height: 633px; height: 100%" default-active="0" router background-color="#545c64"
      text-color="#fff" active-text-color="#ffd04b">
      <el-submenu index="1">
        <template slot="title">
          <i class="el-icon-s-data"></i>
          <span>学习数据管理</span>
        </template>
        <el-menu-item-group>
          <el-menu-item index="home"><i class="el-icon-s-home"></i>主页信息</el-menu-item>
          <el-menu-item index="userData"><i class="el-icon-user"></i>学生信息</el-menu-item>
          <el-menu-item index="questionData"><i class="el-icon-question"></i>习题统计</el-menu-item>
          <el-menu-item index="skillData"><i class="el-icon-s-opportunity"></i>知识内容</el-menu-item>
        </el-menu-item-group>
      </el-submenu>
      <el-submenu index="2">
        <template slot="title">
          <i class="el-icon-s-marketing"></i>
          <span>学习状态分析</span>
        </template>
        <el-menu-item-group>
          <!-- index对应的是各个组件的路由-->
          <el-menu-item index="history"><i class="el-icon-time"></i>答题记录</el-menu-item>
          <el-menu-item index="predict"><i class="el-icon-edit"></i>模拟练习</el-menu-item>
          <el-menu-item index="recommend"><i class="el-icon-collection"></i>习题推荐</el-menu-item>
          <el-menu-item index="skillGraph"><i class="el-icon-pie-chart"></i>知识图谱</el-menu-item>
        </el-menu-item-group>
      </el-submenu>
    </el-menu>
  </div>
</template>

<script>

export default {
  name: "Aside",
  data() {
    return {
      // path: this.$router.path
    }
  }
}
</script>

<style scoped></style>