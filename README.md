# LLMChatBackend



---



基于Python Flask框架和LangChain开发的智能聊天应用，支持文本、文件和图片理解，具备记忆功能和多种工具调用能力。



## 项目概述



这是一个基于Python Flask框架和LangChain开发的智能聊天应用后端，支持多模态交互（文本、文件、图片），具备智能记忆功能和多种工具调用能力。



### 核心功能特性

- **多模态对话**: 支持文本、文件、图片混合交互

- **智能记忆**: 基于Redis的对话记忆，支持会话持久化

- **向量检索**: 集成ChromaDB实现文档内容的智能检索

- **工具调用**: 支持联网搜索、网页爬取、URL访问等工具

- **语音合成**: 文本转语音功能（开发中）

- **会话管理**: 支持会话历史的清理和恢复



## 技术栈



### 后端框架

- **Flask**: Web框架

- **Python**: 3.11版本

- **Redis**: 会话存储

- **ChromaDB**: 向量数据库



### AI/ML技术

- **LangChain**: AI应用开发框架

- **OpenAI兼容API**: 支持多种大模型

- **DashScope**: 阿里云模型服务

- **Embedding模型**: 文本向量化



## 功能特性



### 🗣️ 多模态交互

- **纯文本对话**：直接与LLM进行智能对话

- **文件处理**：支持上传文档文件（PDF、Word、TXT等），自动提取文本并生成摘要

- **图片理解**：支持图片上传，提取文字信息并生成详细描述

- **智能记忆**：基于Redis的对话记忆，保持会话连贯性



### 🔧 智能工具集成

- **向量数据库查询**：基于上传文档的智能检索

- **联网搜索**：实时获取最新信息

- **URL访问**：直接获取网页内容

- **动态工具分配**：根据上下文智能选择可用工具



### ⚡ 技术特色

- **会话记忆**：Redis存储，3600秒自动过期

- **可配置系统提示**：支持用户自定义角色设定

- **模块化设计**：易于扩展和维护

- **文本转语音**（开发中）：基于DashScope的TTS服务



### 文件处理流程

1. **文本提取**：从PDF、Word等文件中提取纯文本

2. **文档理解**：使用LLM生成长文档摘要

3. **文本分片**：将文档分割为适合向量化的片段

4. **向量存储**：将文本片段和摘要存入向量数据库



### 图片处理流程

1. **文字提取**：使用OCR技术提取图片中的文字

2. **视觉描述**：使用多模态LLM生成图片详细描述

3. **上下文整合**：将提取的文字和描述添加到对话上下文




### 文本转语音 (TTS)

- **服务商**：DashScope阿里云
- **计划功能**：将指定文本转换为语音输出



## 项目结构



```

LLM_Chat_Backend/

├── app.py                 # 应用入口
├── routes.py             # API路由定义
├── Config.py.example     # 配置模板
├── models/               # 数据模型
│   ├── llm_factory.py    # LLM工厂类
│   ├── vector_db_manager.py  # 向量数据库管理
│   └── prompts.py        # 提示词模板
├── services/             # 业务服务
│   ├── chat_service.py   # 聊天核心服务
│   └── audio_service.py  # 音频服务
├── utils/                # 工具类
│   ├── session_storage.py    # Redis会话管理
│   ├── file_util.py      # 文件处理工具
│   ├── web_utils.py      # 网络工具
│   └── audio_utils.py    # 音频处理
└── static/               # 静态资源

```



## 快速开始



### 本地开发



1. **环境准备**

   ```bash
   
   # 创建虚拟环境
   
   python -m venv .venv
   
   .\.venv\Scripts\activate
   
   
   
   # 安装依赖
   
   pip install -r requirements.txt
   
   ```



2. **配置环境变量**

   ```bash
   
   # 复制配置文件
   
   cp .env.example .env
   
   
   
   # 编辑.env文件，填入API密钥
   
   LLM_API_KEY=your-llm-key
   
   VISION_MODEL_API_KEY=your-vision-key
   
   DASHSCOPE_API_KEY=your-dashscope-key
   
   TAVILY_API_KEY=your-search-key
   
   ```



3. **启动服务**

   ```bash
   
   # 启动Redis（需要先安装Redis）
   
   redis-server
   
   
   
   # 启动应用
   
   python app.py
   
   ```



   应用将在 http://localhost:5000 启动




## API接口文档



### 基础URL

- 本地: `http://localhost:5000/api/v1`

- 生产环境: 根据实际部署地址



### 核心接口



#### 1. 文本对话

```http

POST /api/v1/chat

Content-Type: application/json



{

  "message": "用户消息",

  "session_id": "会话ID",

  "system_prompt": "可选的系统提示词"

}

```



#### 2. 图片对话

```http

POST /api/v1/chat_with_image

Content-Type: multipart/form-data



- image: 图片文件

- message: 文本消息

- session_id: 会话ID

```



#### 3. 文件对话

```http

POST /api/v1/chat_with_file

Content-Type: multipart/form-data



- file: 文档文件

- message: 文本消息

- session_id: 会话ID

```



#### 4. 清理会话历史

```http

POST /api/v1/clear_current_chat_history

Content-Type: application/json



{

  "session_id": "要清理的会话ID"

}

```



#### 5. 健康检查

```http

GET /api/v1/health

```



## 配置说明



### 环境变量配置



| 变量名 | 说明 | 示例 |
|--------|------|------|
| LLM_API_KEY | 主LLM模型的API密钥 | sk-xxx |
| VISION_MODEL_API_KEY | 视觉模型的API密钥 | sk-xxx |
| DASHSCOPE_API_KEY | DashScope服务的API密钥 | sk-xxx |
| TAVILY_API_KEY | 搜索服务的API密钥 | tvly-xxx |
| REDIS_HOST | Redis主机地址 | localhost |
| REDIS_PORT | Redis端口 | 6379 |



### 应用配置



在 `Config.py` 中可以配置：

- LLM模型名称和API地址

- 嵌入模型配置

- 文件上传限制

- 支持文件类型

- 向量数据库路径



## 开发规范



### 代码风格

- 使用Python 3.11 SDK

- 遵循PEP 8规范

- 函数和变量使用snake_case命名

- 类名使用PascalCase



### 目录结构约定

- `models/`: 数据模型和LLM工厂

- `services/`: 业务逻辑服务

- `utils/`: 通用工具类

- `static/`: 静态资源文件

- `uploads/`: 用户上传文件



### 错误处理

- 使用try-catch处理异常

- 记录详细的错误日志



## 部署指南



### 生产环境部署



1. **环境准备**

   - 安装Python 3.11+

   - 安装Redis服务

   - 配置域名和SSL证书



2. **配置优化**

   - 修改 `Config.py` 中的生产环境配置

   - 设置环境变量

   - 配置日志级别



3. **性能优化**

   - 使用Gunicorn作为WSGI服务器

   - 配置Nginx反向代理

   - 设置适当的worker数量


## 故障排查



### 常见问题



1. **Redis连接失败**

   - 检查Redis服务是否启动

   - 确认REDIS_HOST和REDIS_PORT配置



2. **API调用失败**

   - 检查API密钥是否正确

   - 确认网络连接

   - 查看日志获取详细错误



3. **文件上传失败**

   - 检查文件大小限制

   - 确认文件类型是否支持

   - 检查磁盘空间

   
## 扩展开发



### 添加新工具

1. 在 `utils/` 目录下创建新工具类

2. 在 `chat_service.py` 中注册新工具

3. 更新系统提示词



### 添加新模型

1. 在 `llm_factory.py` 中添加新模型配置

2. 更新 `Config.py` 中的模型参数

3. 测试新模型集成



### Agent能力扩展

1. **多Agent协作系统**

   - 实现专门的任务型Agent（如数据分析Agent、代码生成Agent等）

   - 开发Agent间通信机制，支持复杂任务分解和协作

   - 添加Agent调度器，根据任务类型自动选择合适的Agent



2. **增强的工具调用能力**

   - 实现工具的动态注册和发现机制

   - 添加工具执行结果的缓存机制，避免重复调用

   - 开发工具执行监控和错误处理机制

   - 支持异步工具调用，提升响应速度



3. **Agent记忆和学习能力**

   - 扩展长期记忆系统，支持跨会话知识积累

   - 实现基于用户反馈的Agent自我优化机制

   - 添加对话策略学习能力，根据历史对话优化响应



4. **个性化Agent定制**

   - 开发Agent角色和性格定制功能

   - 实现基于用户偏好的响应风格调整

   - 添加用户画像系统，支持个性化服务推荐



5. **Agent安全和权限控制**

   - 实现细粒度的工具调用权限控制

   - 添加恶意操作检测和防护机制

   - 开发用户数据隐私保护功能



## 许可证



MIT License - 详见 LICENSE 文件
