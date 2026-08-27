import streamlit as st
import re
import matplotlib.pyplot as plt
import numpy as np
import datetime
from datetime import timezone, timedelta
import db_core
import rag_core
import plot_core  # 导入新解耦的专属绘图核心模块

# ==========================================
# 1. 页面与全局设置
# ==========================================
st.set_page_config(page_title="信号与系统 AI 助教", page_icon="📡", layout="wide")

# 🎨 注入自定义 CSS，完美复刻 Gemini 居中留白布局
st.markdown("""
    <style>
    /* 限制主聊天区域的最大宽度并使其居中 */
    .block-container {
        max-width: 900px !important; 
        padding-top: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 文本解析组件
# ==========================================
def render_markdown_with_latex(text):
    """自动将大模型的 LaTeX 符号转换为 Streamlit 支持的格式"""
    if not isinstance(text, str):
        return
    # 替换独立公式块
    text = text.replace('\\[', '$$').replace('\\]', '$$')
    # 替换行内公式
    text = text.replace('\\(', '$').replace('\\)', '$')
    st.markdown(text)

# ==========================================
# 3. 登录认证模块 (带密码)
# ==========================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

st.sidebar.title("🔐 用户登录")
if not st.session_state.logged_in:
    username_input = st.sidebar.text_input("👤 用户名")
    password_input = st.sidebar.text_input("🔑 密码", type="password") # 密码输入框，自动隐藏字符
    
    if st.sidebar.button("登录"):
        if username_input.strip() and password_input.strip():
            st.session_state.logged_in = True
            st.session_state.username = username_input.strip()
            st.rerun()
        else:
            st.sidebar.error("用户名和密码不能为空！")
    st.stop() # 阻断页面继续加载，直到登录成功
else:
    st.sidebar.success(f"欢迎回来：{st.session_state.username}")
    if st.sidebar.button("退出登录"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

# ==========================================
# 4. 主界面与专家选择
# ==========================================
st.sidebar.markdown("---")
expert_modes = [
    "🔍 深度答疑专家 (讲解/解惑)", 
    "📝 测验与解析专家 (出题/批改)", 
    "📊 仿真绘图专家 (波形/频谱)"
]
selected_expert = st.sidebar.radio("请选择你的专属 AI 助教", expert_modes)

PDF_FILE_PATH = "signal_and_systems.pdf" 

# ==========================================
# 5. 历史记录模块 (排在符号面板上方)
# ==========================================
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 历史对话记录")

# 从云端数据库拉取当前登录用户的全部历史记录
with st.spinner("🔄 同步历史记忆..."):
    full_history = db_core.load_user_history(st.session_state.username)

# 获取当前选中专家下的所有对话标题
available_sessions = list(full_history.get(selected_expert, {}).keys())
if not available_sessions:
    available_sessions = ["默认对话"]

# 侧边栏：新建对话按钮
if st.sidebar.button("➕ 新建对话"):
    new_session = datetime.datetime.now().strftime("对话_%m%d_%H%M")
    st.session_state.current_session = new_session
    st.rerun()

# 确保 session_state 里有 current_session
if "current_session" not in st.session_state or st.session_state.current_session not in available_sessions:
    if "current_session" in st.session_state and st.session_state.current_session.startswith("对话_"):
        available_sessions.insert(0, st.session_state.current_session)
    else:
        st.session_state.current_session = available_sessions[-1] if available_sessions else "默认对话"

# 侧边栏：下拉菜单选择历史对话
current_session = st.sidebar.selectbox(
    "选择或切换聊天记录",
    options=available_sessions,
    index=available_sessions.index(st.session_state.current_session) if st.session_state.current_session in available_sessions else 0
)
st.session_state.current_session = current_session

# 获取当前选中的具体聊天消息
messages = full_history.get(selected_expert, {}).get(current_session, [])

# ==========================================
# 6. 专属符号模块 (折叠面板)
# ==========================================
st.sidebar.markdown("---")

def render_symbol_sidebar():
    """渲染侧边栏的专属符号复制面板（折叠形式）"""
    with st.sidebar.expander("🧮 专属符号面板 (点击展开)"):
        st.caption("👉 鼠标悬浮在符号上，点击 📋 即可复制")

        st.markdown("**1. 常用希腊字母**")
        c1, c2, c3, c4 = st.columns(4)
        c1.code("ω", language="text")
        c2.code("Ω", language="text")
        c3.code("π", language="text")
        c4.code("τ", language="text")
        
        st.markdown("**2. 核心奇异信号**")
        c1, c2, c3 = st.columns(3)
        c1.code("δ(t)", language="text")
        c2.code("u(t)", language="text")
        c3.code("Sa(t)", language="text")
        
        st.markdown("**3. 变换算子**")
        c1, c2, c3 = st.columns(3)
        c1.code("ℱ", language="text")
        c2.code("ℒ", language="text")
        c3.code("𝒵", language="text")
        
        st.markdown("**4. 数学运算**")
        c1, c2, c3, c4 = st.columns(4)
        c1.code("*", language="text")
        c2.code("∞", language="text")
        c3.code("∫", language="text")
        c4.code("∑", language="text")

# 调用折叠面板
render_symbol_sidebar()

# ==========================================
# 7. 对话渲染与交互
# ==========================================
st.title(selected_expert.split()[1]) # 在主界面显示助教名字

# 渲染历史对话
for i, msg in enumerate(messages):
    with st.chat_message(msg["role"]):
        # 调取数据库中存储的时间戳并进行展示
        if "time" in msg and msg["time"]:
            st.caption(f"🕒 {msg['time']}")
            
        render_markdown_with_latex(msg["content"])
        
        # 🌟 新增功能 1：为历史记录里的 AI 回复添加一键复制面板
        if msg["role"] == "assistant":
            with st.expander("📋 点击展开以一键复制全文"):
                st.code(msg["content"], language="markdown")
        
        # 如果是历史记录里的画图专家回复，调用模块渲染出带微调台的图像
        if "📊" in selected_expert and msg["role"] == "assistant":
            plot_core.render_interactive_plot(msg["content"], msg_index=f"history_{i}")

# ==========================================
# 8. AI 调用与数据库写入
# ==========================================
if prompt := st.chat_input("输入你的问题，或展开左侧面板复制符号..."):
    
    # 🌟 新增功能 2：在发给大模型之前，先把刚才的历史记录整理成上下文 (海马体模块)
    history_for_chain = []
    for msg in messages:
        role = "assistant" if msg["role"] == "assistant" else "user"
        history_for_chain.append((role, msg["content"]))
    
    # 智能重命名逻辑：如果是新对话，发第一句话时自动生成标题并同步数据库名字
    was_renamed = False
    if st.session_state.current_session.startswith("对话_"):
        old_session_name = st.session_state.current_session
        new_session_name = prompt[:12] + ("..." if len(prompt) > 12 else "")
        
        st.session_state.current_session = new_session_name
        was_renamed = True
        
        db_core.rename_session_in_db(
            st.session_state.username, 
            selected_expert, 
            old_session_name, 
            new_session_name
        )

    # 捕获精确的北京提问时间
    bj_tz = timezone(timedelta(hours=8))
    current_time = datetime.datetime.now(bj_tz).strftime("%Y年%m月%d日 %H:%M:%S")

    # 1. 展示用户提问
    with st.chat_message("user"):
        st.caption(f"🕒 {current_time}")
        render_markdown_with_latex(prompt)
        
    # 2. 调用大模型生成回复
    with st.chat_message("assistant"):
        with st.spinner("🧠 AI 正在检索教材并思考中..."):
            try:
                api_key = st.secrets["API_KEY"]
                
                # 初始化 RAG 系统
                qa_chain = rag_core.init_rag_system(
                    api_key=api_key, 
                    expert_mode=selected_expert, 
                    pdf_name=PDF_FILE_PATH
                )
                
                # 🌟 核心修改：通过字典将问题和历史记录一起传给大模型
                response = qa_chain.invoke({
                    "query": prompt,
                    "chat_history": history_for_chain
                })
                
                # 兼容处理返回的各种数据格式
                if isinstance(response, dict) and "result" in response:
                    ai_reply = response["result"]
                elif isinstance(response, dict) and "text" in response:
                    ai_reply = response["text"]
                elif hasattr(response, "content"):
                    ai_reply = response.content
                else:
                    ai_reply = str(response)
                
                # 展现 AI 回复文本与时间戳
                st.caption(f"🕒 {current_time}")
                render_markdown_with_latex(ai_reply)
                
                # 🌟 新增功能 1：为新生成的 AI 回复添加一键复制面板
                with st.expander("📋 点击展开以一键复制全文"):
                    st.code(ai_reply, language="markdown")
                
                # 3. 若当前是绘图专家，调用外包模块输出可微调图像
                if "📊" in selected_expert:
                    plot_core.render_interactive_plot(ai_reply, msg_index=f"new_{datetime.datetime.now().timestamp()}")
                    
                # 4. 将本次完整交互连同精准时间戳存入云数据库
                db_core.log_interaction(
                    username=st.session_state.username,
                    expert_mode=selected_expert,
                    session_id=st.session_state.current_session,
                    query=prompt,
                    response=ai_reply
                )
                
            except KeyError:
                st.error("🔑 发生错误：未能从系统配置(Secrets)中找到 API_KEY，请检查配置！")
            except Exception as e:
                st.error(f"❌ 系统发生异常: {e}")
                
        # 对话生成结束。如果触发了智能改名，强制刷新页面使得侧边栏实时生效
        if was_renamed:
            st.rerun()
