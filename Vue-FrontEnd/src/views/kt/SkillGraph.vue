<!-- 知识点掌握图 -->
<template>
  <!-- 顶排按钮和搜索框 -->
  <div style="margin: 20px 0 20px 20px; width: 600px">
    <el-input v-model="sListInput" placeholder="请输入知识点ID[1-123]（英文逗号分隔，例: 25,109,84）" style="width: 30%;" clearable>
    </el-input>
    <el-button type="primary" icon="el-icon-search" style="margin-left: 5px" @click="load">搜索知识图谱</el-button>
    <div id="table" class="myChart"></div>
  </div>
</template>

<script>
import request from "@/api/request";

export default {
  name: "skillGraph",
  data() {
    return {
      sListInput : null,
      // 图标数据
      chartInstance: null,
      option: {},
      data: [
        {'id': 1, name: 'skill1'},
        {'id': 2, name: 'skill2'},
        {'id': 3, name: 'skill3'}
      ], // 顶点
      links: [
        {source: 0, target: 1}
      ], // 边
    }
  },

  mounted() {
    this.initChart()
    this.updateChart()
  },

  methods: {
    load() {
      console.log(this.nodes)
      request.get("/kt/skillGraph", {
        params:{
          sList: this.sListInput
        }
      }).then(res => {
        console.log(res)
        this.data = res.data.data
        this.links = res.data.links
        this.updateChart()
      })
    },

    initChart() {
      this.chartInstance = this.$echarts.init(document.getElementById('table'))
    },

    updateChart() {
      this.option = {
        title: {
          text: '知识图谱' // 图的标题
        },
        tooltip: {}, // 提示框的配置
        // 工具箱
        toolbox: { // 显示工具箱
          show: true,
          feature: {
            mark: {
              show: true
            },  // 还原
            restore: {
              show: true
            }, // 保存为图片
            saveAsImage: {
              show: true
            },
            dataView: {}, // 数据视图
            dataZoom: {}, // 区域缩放
          }
        },
        series: [
          {
            type: 'graph', // 类型:关系图
            // layout: 'none', // 图的布局，类型为无向图，但是必须需要坐标表示
            layout: 'force', // 力传导图，不需要坐标
            symbolSize: 100, // 调整节点的大小
            roam: true, // 是否开启鼠标缩放和平移漫游。默认不开启。如果只想要开启缩放或者平移,可以设置成 'scale' 或者 'move'。设置成 true 为都开启
            edgeSymbol: ['circle', 'arrow'],
            edgeSymbolSize: [2, 5],
            force: {
              repulsion: 2500,
              edgeLength: [10, 500]
            },
            draggable: true,
            lineStyle: {
              width: 2,
              color: 'source',
              curveness: 0.3
            },
            emphasis: {
              focus: 'adjacency',
              lineStyle: {
                width: 5
              }
            },
            label: {
              normal: {
                show: true,
                textStyle: {}
              }
            },
            // 下面是后端应该传来的数据
            data: this.data, // 顶点
            links: this.links, // 边
          }]
      };
      this.chartInstance.setOption(this.option)
    }
  }
}
</script>

<style scoped></style>
