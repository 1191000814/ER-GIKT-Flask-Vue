<template>
  <div style="margin-left: 20px; margin-top:20px; margin-right: 20px; font-size: 40px">
    <el-descriptions class="margin-top" title="ER-GIKT-Flask-Vue项目基本概述" column="1" size="large" border>
      <el-descriptions-item>
        <template slot="label">
          <i class="el-icon-star-off"></i>
          项目名称
        </template>
        <el-link type="primary" href="https://github.com/1191000814/ER-GIKT-Flask-Vue">基于深度知识追踪模型的习题推荐系统的设计与实现</el-link>
      </el-descriptions-item>
      <el-descriptions-item>
        <template slot="label">
          <i class="el-icon-goods"></i>
          项目来源
        </template>
        哈尔滨工业大学（HIT）-计算学部-软件学院-智慧教育系统（陈源龙老师）的子系统
      </el-descriptions-item>
      <el-descriptions-item>
        <template slot="label">
          <i class="el-icon-s-opportunity"></i>
          项目内容
        </template>
        算法模型：基于Pytorch实现的图扩散卷积网络（GDC）+长短期记忆神经网络（LSTM）组合的深度追踪模型，并使用PEBG模型预训练问题嵌入向量<br/>
        软件系统：基于Flask+Vue前后端分离技术实现的用户交互式系统，主要用于追踪学习状态、预测答题正确率、推荐习题和分析知识图谱等
      </el-descriptions-item><br/>
      <el-descriptions-item>
        <template slot="label">
          <i class="el-icon-setting"></i>
          技术框架
        </template>
        <el-link type="primary" href="https://pytorch.org/">Pytorch &nbsp;</el-link>
        <el-link type="primary" href="https://flask.palletsprojects.com/">Flask &nbsp;</el-link>
        <el-link type="primary" href="https://flask-sqlalchemy.palletsprojects.com/">Flask-Sqlalchemy &nbsp;</el-link>
        <el-link type="primary" href="https://vuejs.org/">Vue &nbsp;</el-link>
        <el-link type="primary" href="https://cli.vuejs.org/">Vue-Cli &nbsp;</el-link>
        <el-link type="primary" href="https://element.eleme.io/">Element &nbsp;</el-link>
        <el-link type="primary" href="https://echarts.apache.org/">Echarts &nbsp;</el-link>
      </el-descriptions-item>
      <el-descriptions-item>
        <template slot="label">
          <i class="el-icon-office-building"></i>
          创作地址
        </template>
        哈尔滨工业大学一校区正心10楼
      </el-descriptions-item>
      <el-descriptions-item>
        <template slot="label">
          <i class="el-icon-user"></i>
          联系作者
        </template>
        <el-link type="primary" href="mailto:z.q.xiao@qq.com">z.q.xiao@qq.com</el-link>
      </el-descriptions-item>
    </el-descriptions>

  </div>
</template>

<script>
export default {
  name: "home"
}
</script>

<style scoped>

</style>