<!-- 习题模拟练习（包含两个图表）-->
<template>
  <div>
    <!-- 顶排按钮和搜索框 -->
    <div style="margin: 20px 0 20px 20px;">
      <el-input v-model="qListInput" placeholder="请输入题目ID[1-15793]（英文逗号分隔，例: 11,423,89）" style="width: 30%;"
                clearable></el-input>
      <el-button type="primary" icon="el-icon-position" style="margin-left: 5px" @click="load">开始模拟</el-button>
    </div>
    <!-- 一点空白行 -->
    <el-row>
      <!-- 第一个表格 -->
      <el-col :span="12">
        <!-- id="table3"在后面会被get到 -->
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
  name: "predict",

  data() {
    return {
      qListInput: null, // 输入的用逗号分隔的问题列表（字符串）
      // 第一个图表（折线图）有关数据
      chartInstance1: null,
      option1: {},
      qList: [1, 2, 3, 4, 5], // 问题列表
      cList: [0.4, 0.5, 0.7, 0.3, 0.5], // 预测正确率列表
      // 第二个图表（雷达图）有关数据
      chartInstance2: null,
      option2: null,
      skillIndicator: [
        {name: 'Sales', max: 1},
        {name: 'Administration', max: 1},
        {name: 'Information Technology', max: 1},
        {name: 'Customer Support', max: 1},
        {name: 'Development', max: 1},
        {name: 'Marketing', max: 1}
      ],
      skillMastery: [0.2, 0.4, 0.3, 0.8, 0.6, 0.7], // 每个知识点的掌握程度
    }
  },
  methods: {
    // 加载输入框中的数据，并将其发往后端
    load() {
      request.get("kt/predict", {
        params:{
          qList: this.qListInput
        }
      }).then(res => {
        console.log(res)
        if(res.msg){ // 报错信息
          this.$message(res.msg)
          return
        }
        this.qList = res.data.qList
        this.cList = res.data.cList
        this.skillIndicator = res.data.skillIndicator
        this.skillMastery = res.data.skillMastery
        this.updateChart1()
        this.updateChart2()
      })
    },
    // 初始化第一个表格
    initChart1() {
      // 传入一个dom，并初始化容器为一个表格
      this.chartInstance1 = this.$echarts.init(document.getElementById('table1'))
    },
    // 初始化第二个表格
    initChart2() {
      // 与上同理
      this.chartInstance2 = this.$echarts.init(document.getElementById('table2'))
    },
    // 更新第一个表格
    updateChart1() {
      this.option1 = {
        dataZoom: [
          {
            type: 'inside', // slider 缩放方式
            xAxisIndex: 0 // 对x轴进行缩放
          }
        ],
        title: {
          text: '习题模拟',
          textStyle: {
            color: 'black'
          },
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
        legend: { // 图例
          data: ['预测答题准确率']
        },
        xAxis: {
          type: 'category',
          data: this.qList
        },
        yAxis: {
          type: 'value'
        },
        series: [
          {
            name: '预测答题准确率',
            type: 'line',
            markLine: { // 标记线
              data: [
                {
                  type: 'average', name: 'averageValue'
                }
              ]
            },
            label: {
              show: true
            },
            data: this.cList,
            // smooth: true, // 平滑
            lineStyle: { // 线条样式
              color: 'blue',
              type: 'solid' // dashed dotted solid
            },
          },
        ]
      }
      this.chartInstance1.setOption(this.option1)
    },
    // 更新第二个表格(雷达图)
    updateChart2() {
      // 表格的option，决定怎么显示这个图表，下面的属性都是echarts图表本该有的属性
      this.option2 = {
        title: {
          text: '知识掌握',
          textStyle: {
            color: "black"
          },
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
        legend: { // 图例
          data: ['知识点掌握程度']
        },
        radar: {
          indicator: this.skillIndicator // 雷达图各个指标名称和最高得分
        },
        series: [
          {
            name: '知识点掌握程度',
            type: 'radar',
            label: {
              show: true
            },
            data: [{
              value: [0.42, 0.3, 0.2, 0.35, 0.5, 0.18],
              name: '知识点掌握程度'
            }]
          }
        ]
      }
      // 将上面option加载到图表chart2中
      this.chartInstance2.setOption(this.option2)
    },
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