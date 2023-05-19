<!--
当前学习状态
第一个界面的表格：输入的是问题序列，返回的是问题的回答正确的预测概率
将两者组成统计直方图
-->
<template>
  <div>
    <div style="margin: 20px 0 20px 20px;">
      <el-select placeholder="请选择推荐数量" v-model="num">
        <el-option
            v-for="item in numOption"
            :key="item.value"
            :label="item.label"
            :value="item.value">
        </el-option>
      </el-select>
      <el-button type="primary" icon="el-icon-refresh-right" style="margin-left: 5px" @click="load">换一批</el-button>
    </div>
    <!--    用来放table1-->
    <el-row style="height: 20%">
      <!-- 第一个表格 -->
      <el-col :span="12">
        <!-- id="table1"在后面会被get到 -->
        <div id="table1" class="myChart"></div>
      </el-col>
      <!-- 第二个表格 -->
      <el-col :span="12">
        <div id="table2" class="myChart"></div>
      </el-col>
    </el-row>

  </div>
</template>

<script>
import request from "@/api/request";
export default {
  name: "recommend",
  data() {
    return {
      // 第一个表格（折线图）所需数据
      chartInstance1: null,
      option1: {},
      qList: [1, 2, 3, 4, 5],
      cList: [0.1, 0.4, 0.5, 0.8, 0.7],
      numOption:[
        {
          label: '5条',
          value: 5
        },
        {
          label: '10条',
          value: 10
        },
        {
          label: '20条',
          value: 20
        },
      ],
      num: 10,
      // 第二个表格（圆环图）所需数据
      chartInstance2: null,
      option2: {},
      skillData: [
        {
          value: 335,
          name: 'A'
        },
        {
          value: 234,
          name: 'B'
        },
        {
          value: 1548,
          name: 'C'
        }
      ]
    }
  },
  created() {
    let userStr = sessionStorage.getItem("user")
    this.user = JSON.parse(userStr)
  },

  methods: {
    load() {
      request.get("kt/recommend", {
        params: {
          num: this.num,
          userId: this.user.id
        }
      }).then(res => {
        // 返回结果当成图显示，横坐标为问题id，纵坐标为预测的准确度
        console.log(res)
        if(res.msg){
          this.$message(res.msg)
          return
        }
        this.qList = res.data.qList
        this.cList = res.data.cList
        this.skillData = res.data.skillData
        this.updateChart1()
        this.updateChart2()
      })
    },
    // 第一个表格：推荐的习题以及推荐度（预测的正确率）
    initChart1() {
      this.chartInstance1 = this.$echarts.init(document.getElementById('table1'))
    },

    updateChart1() {
      this.option1 = {
        title: {
          text: '推荐习题列表',
          textStyle: {
            color: "black"
          },
        },
        tooltip: { // 提示框组件
          trigger: 'axis', // axis
          triggerOn: 'click', // 触发形式
          formatter: '{b}:{c}' // 提示信息
        },
        toolbox: { // 通用配置工具栏
          feature: {
            saveAsImage: {}, // 导出图片
            dataView: {}, // 数据视图
            restore: {}, // 重置
            dataZoom: {}, // 区域缩放
            magicType: {
              type: ['bar', 'line'] // 类型切换
            }
          }
        },
        legend: { // 图例
          data: ['难度系数']
        },
        grid: {
          left: '3%',
          right: '4%',
          bottom: '3%',
          containLabel: true
        },
        xAxis: {
          type: 'category', // 类型：系列数据（直方图）
          data: this.qList, // x轴数据
        },
        yAxis: {
          type: 'value' // 类型：值
        },
        series: [
          {
            name: '难度系数',
            type: 'line',
            step: 'middle',
            label:{ show:true },
            data: this.cList
          }
        ]
      };
      this.chartInstance1.setOption(this.option1)
    },
    // 第二个表格: 圆环图，知识的分布情况（各个知识占的百分比）
    initChart2(){
      this.chartInstance2 = this.$echarts.init(document.getElementById('table2'))
    },
    updateChart2(){
      this.option2 = {
        title: {
          text: '知识点分布',
          left: 'center',
          top: 'center'
        },
        tooltip: { // 提示框组件
          trigger: 'item', // axis
          triggerOn: 'click', // 触发形式
          formatter: '{b}:{c}' // 提示信息
        },
        toolbox: { // 通用配置工具栏
          feature: {
            saveAsImage: {}, // 导出图片
            dataView: {}, // 数据视图
            restore: {}, // 重置
            dataZoom: {}, // 区域缩放
            magicType: {
              type: ['bar', 'line']
            }
          }
        },
        series: [
          {
            type: 'pie', // 饼图/圆环图
            data: this.skillData,
            radius: ['40%', '70%']
          }
        ]
      };
      this.chartInstance2.setOption(this.option2)
    }
  },

  mounted() {
    this.initChart1()
    this.initChart2()
    this.updateChart1()
    this.updateChart2()
  }
}
</script>

<style scoped></style>