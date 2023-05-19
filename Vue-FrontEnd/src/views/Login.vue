<!-- 登录与注册页面 -->
<template>
  <div style="width: 100%; height: 100vh; background-color: burlywood; overflow: hidden">
    <div style="width: 400px; margin: 200px auto">
      <!--      登录页面-->
      <div style="font-size: xx-large; text-align: center">欢迎登录</div>
      <el-form ref="form" :model="formData">
        <el-form-item label="用户名">
          <el-input placeholder="请输入用户名" prefix-icon="el-icon-user-solid" type="username"
                    v-model="formData.username"></el-input>
        </el-form-item>
        <el-form-item label="密码" prop="password">
          <el-input :placeholder="msg" prefix-icon="el-icon-lock" type="password" v-model="formData.password" clearable
                    show-password></el-input>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="login">登录</el-button>
          <el-button @click="beginRegister">注册</el-button>
        </el-form-item>
      </el-form>
      <!--      注册框-->
      <el-dialog title="欢迎注册" :visible.sync="dialogVisible" width="30%">
        <el-form ref="form" :model="formData" label-width="80px">
          <el-form-item label="用户姓名">
            <el-input placeholder="请输入自己的用户名" v-model="formData.username"></el-input>
          </el-form-item>
          <el-form-item label="用户密码">
            <el-input placeholder="默认密码为123456" v-model="formData.password"></el-input>
          </el-form-item>
          <el-form-item label="性别">
            <el-select v-model="formData.sex" placeholder="请选择性别">
              <el-option label="男" value="1"></el-option>
              <el-option label="女" value="0"></el-option>
            </el-select>
          </el-form-item>
          <el-form-item label="年龄">
            <el-input-number v-model="formData.age" :min=6 :max=60 placeholder="请选择年龄">
            </el-input-number>
          </el-form-item>
          <el-form-item label="权限">
            <el-select v-model="formData.role" placeholder="请选择角色">
              <el-option label="普通用户" value="1"></el-option>
            </el-select>
          </el-form-item>
        </el-form>
        <span slot="footer" class="dialog-footer">
          <el-button @click="dialogVisible = false">取 消</el-button>
          <el-button type="primary" @click="register">确 定</el-button>
        </span>
      </el-dialog>
    </div>
  </div>
</template>

<script>
import request from "@/api/request";

export default {
  name: "login",
  data() {
    return {
      formData: {
        username: null,
        password: null,
        age: 18,
        sex: null,
        role: null
      },
      dialogVisible: false, // 是否显示注册输入框
      msg: "请输入密码",
    }
  },

  methods: {
    login() {
      console.log(this.formData)
      request.post('/user/login', this.formData).then(res => {
        console.log(res)
        if (res.code === 0) { // code=='0'表示登录成功
          this.$message({
            type: "success", message: "登录成功"
          })
          sessionStorage.setItem("user", JSON.stringify(res.data))
          // 在此会话中保存用户信息(setItem只能存储字符串)
          // console.log(sessionStorage.getItem("user"))
          this.$router.push("/home")
        } else {
          this.msg = "用户名或密码错误"
          this.formData.password = "" // 清空密码
        }
      })
    },

    beginRegister() {
      this.dialogVisible = true // 显示注册填写框
    },

    register() {
      request.post("/user/register", this.formData).then(res => {
        if (res.code === 0) {
          this.$message({
            type: "success", message: "注册成功"
          })
          this.dialogVisible = false;
          this.$router.push('/home')
        } else
          this.$message({
            type: "error", message: "注册失败"
          })
      });
    },
  },

  created(){
    // 每次在登录页面时，清除保存的用户信息
    sessionStorage.clear()
  }
}
</script>

<style scoped>
</style>