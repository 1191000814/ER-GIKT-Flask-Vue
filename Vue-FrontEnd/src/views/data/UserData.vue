<template>
  <div style="padding: 10px">
    <div style="margin: 10px 0">
      <el-button type="primary" icon="el-icon-plus" style="margin-left: 5px" @click="handleAdd" :disabled="user.role">新增</el-button>
      <el-input v-model="search" placeholder="请输入用户名" style="width: 20%; margin-left: 5px"
                clearable></el-input>
      <el-button type="primary" icon="el-icon-search" style="margin-left: 5px" @click="load()">搜索</el-button>
      <el-tag style="margin-left: 10px">共查询到{{ totalNum }}条记录</el-tag>
    </div>
    <el-table :data="tableData" border style="width: 100%;" size="small">
      <el-table-column fixed prop="id" label="ID" sortable></el-table-column>
      <el-table-column prop="username" label="用户名"></el-table-column>
      <el-table-column prop="age" label="年龄"></el-table-column>
      <el-table-column prop="sex" label="性别">
        <template v-slot="scope">
          <span v-if="scope.row.sex === 0">女</span>
          <span v-if="scope.row.sex === 1">男</span>
        </template>
      </el-table-column>
      <el-table-column prop="role" label="权限">
        <template v-slot="scope">
          <span v-if="scope.row.role === 0">管理员</span>
          <span v-if="scope.row.role === 1">普通用户</span>
        </template>
      </el-table-column>
      <el-table-column label="操作">
        <template v-slot="scope">
          <el-button :disabled="user.role" type="warning" @click="handleUpdate(scope.row)">
            <i class="el-icon-edit"></i>编辑
          </el-button>
          <el-popconfirm
              confirm-button-text='好的'
              cancel-button-text='不用了'
              icon="el-icon-info"
              icon-color="red"
              title="这是一段内容确定删除吗？"
              @confirm="handleDelete(scope.row)"
          >
            <el-button :disabled="user.role" slot="reference" type="danger" style="margin-left: 5px">
              <i class="el-icon-delete"></i>删除
            </el-button>
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

      <!-- 增加一行的输入框-->
      <el-dialog title="请输入数据" :visible.sync="dialogVisible" width="30%">
        <el-form ref="form" :model="formData" label-width="80px">
          <el-form-item label="用户姓名">
            <el-input v-model="formData.username"></el-input>
          </el-form-item>
          <el-form-item label="用户密码">
            <el-input v-model="formData.password"></el-input>
          </el-form-item>
          <el-form-item label="性别">
            <el-select v-model="formData.sex" placeholder="请选择性别">
              <el-option label="男" value="1"></el-option>
              <el-option label="女" value="0"></el-option>
            </el-select>
          </el-form-item>
          <el-form-item label="年龄">
            <el-input-number v-model="formData.age" :min="6" :max="60" placeholder="请选择年龄">
            </el-input-number>
          </el-form-item>
          <el-form-item label="权限">
            <el-select v-model="formData.role" placeholder="请选择权限">
              <el-option label="管理员" value="0"></el-option>
              <el-option label="普通用户" value="1"></el-option>
            </el-select>
          </el-form-item>
        </el-form>
        <span slot="footer" class="dialog-footer">
        <el-button @click="dialogVisible = false">取 消</el-button>
        <el-button type="primary" @click="handleSave">确 定</el-button>
        </span>
      </el-dialog>
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
        age: 12,
        sex: null,
        role: null,
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
      request.get("/user", {
        params: {
          pageIndex: this.pageIndex,
          pageSize: this.pageSize,
          search: this.search
        }
      }).then(res => {
        this.tableData = res.data
        this.totalNum = res.num
      })
    },

    handleAdd() {
      this.dialogVisible = true;
      this.formData = {};
    },

    handleSave() {
      if (this.formData.id) { // 参数带有id的，是更新（update）操作
        console.log('打印')
        console.log(this.formData)
        request.post("/user/update", this.formData).then(res => {
          console.log(res)
          if (res.code === 0)
            this.$message({
              type: "success", message: "更新成功"
            })
          else
            this.$message({
              type: "error", message: "更新失败"
            })
        })
        this.dialogVisible = false
      } else { // 参数不带id，是添加（add）操作
        request.post("/user/add", this.formData).then(res => {
          if (res.code === 0)
            this.$message({
              type: "success", message: "添加成功"
            })
          else
            this.$message({
              type: "error", message: "添加失败"
            })
        });
        this.dialogVisible = false;
      }
      this.load()
    },

    handleUpdate(row) {
      this.formData = JSON.parse(JSON.stringify(row))
      console.log('更新时的表单')
      console.log(this.formData)
      this.dialogVisible = true
    },

    handleDelete(row) {
      const row_id = row.id
      request.post("/user/delete", {
        user_id: row_id
      }).then(res => {
        console.log(res)
        if (res.code === 0)
          this.$message({
            type: "success", message: "删除成功"
          })
        else
          this.$message({
            type: "error", message: res.msg
          })
      })
      this.load()
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
