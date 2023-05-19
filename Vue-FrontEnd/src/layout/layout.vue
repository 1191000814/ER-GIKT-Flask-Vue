<!-- 整个页面的布局 -->
<template>
  <div class="bg" style="background-color: rgba(254,248,239,1)">
    <!--    头部-->
    <Header />
    <!--    主体-->
    <div style="display: flex">
      <!--      侧边栏-->
      <Aside />
      <!--      内容区域，由路由决定-->
      <router-view style="flex: 1" />
    </div>
  </div>
</template>

<script>

import Header from "@/components/Header";
import Aside from "@/components/Aside";
export default {
  name: "layout",
  components: {
    Header,
    Aside
  }
}
</script>

<style scoped></style>