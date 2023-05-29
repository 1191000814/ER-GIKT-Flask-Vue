<template>
  <div style="padding: 10px">
    <div style="margin: 10px 0">
      <el-button type="primary" icon="el-icon-plus" style="margin-left: 5px" @click="handleAdd">新增</el-button>
      <el-input v-model="search" placeholder="请输入习题ID" style="width: 20%; margin-left: 5px"
                clearable></el-input>
      <el-button type="primary" icon="el-icon-search" style="margin-left: 5px" @click="load()">搜索</el-button>
      <el-tag style="margin-left: 10px">共查询到{{ totalNum }}条记录</el-tag>
    </div>
    <el-table :data="tableData" border style="width: 100%" size="small">
      <el-table-column fixed prop="id" label="回答ID" sortable></el-table-column>
      <el-table-column prop="username" label="学习者用户名"></el-table-column>
      <el-table-column prop="q_id" label="习题ID"></el-table-column>
      <el-table-column prop="skills" label="习题相关知识点"></el-table-column>
      <el-table-column prop="correct" label="回答结果">
        <template v-slot="scope">
          <el-tag v-if="scope.row.correct === 1" type="success">正确</el-tag>
          <el-tag v-if="scope.row.correct === 0" type="danger">错误</el-tag>
        </template>
      </el-table-column>
      <el-table-column label="操作">
        <template v-slot="scope">
          <el-button :disabled="true" type="warning">编辑</el-button>
          <el-popconfirm
              confirm-button-text='好的'
              cancel-button-text='不用了'
              icon="el-icon-info"
              icon-color="red"
              title="这是一段内容确定删除吗？"
              @confirm="handleDelete(scope.row)"
          >
            <el-button :disabled="true" slot="reference" type="danger" style="margin-left: 5px">删除</el-button>
          </el-popconfirm>
        </template>
      </el-table-column>
    </el-table>

    <div style="margin: 10px 0">
      <el-pagination
          @size-change="handleSizeChange"
          @current-change="handleIndexChange"
          :current-page="pageIndex"
          :page-sizes="[5, 10, 20]"
          :page-size="pageSize"
          layout="sizes, prev, pager, next, jumper, total"
          :total="totalNum">
      </el-pagination>
    </div>
  </div>
</template>

<script>

import request from "@/api/request";

export default {
  name: 'UserData',
  components: {},

  data() {
    return {
      user: null, // 登录者信息
      formData: {
        id: null,
        username: null,
        q_id: 12,
        skills: null,
        correct: null,
      }, // 增加或修改的表单数据
      dialogVisible: false,
      tableData: [], // 全部表格数据
      search: '', // 搜索输入框
      totalNum: 0,
      pageIndex: 1,
      pageSize: 10
    }
  },

  // 创建该页面时，先加载登录者信息，再加载页面数据
  created() {
    let userStr = sessionStorage.getItem("user")
    this.user = JSON.parse(userStr)
    this.load()
  },

  methods: {
    // 加载全部数据
    load() {
      request.get("/kt/history", {
        params: {
          pageIndex: this.pageIndex,
          pageSize: this.pageSize,
          search: this.search,
          userId: this.user.id
        }
      }).then(res => {
        if(res.msg){
          this.$message(res.msg)
          return
        }
        this.tableData = res.data
        this.totalNum = res.num
      })
    },
    handleAdd(){
    },
    handleUpdate(row){
    },
    handleDelete(){
    },

    handleSizeChange(pageSize) {
      this.pageSize = pageSize
      this.load()
    },

    handleIndexChange(pageIndex) {
      this.pageIndex = pageIndex
      this.load()
    }
  }
}
</script>