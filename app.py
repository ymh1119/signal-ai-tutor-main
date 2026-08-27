import streamlit as st
import re
import matplotlib.pyplot as plt
import numpy as np
import datetime
from datetime import timezone, timedelta
import db_core
import rag_core
import plot_core
import os

# ==========================================
# 1. 页面与全局设置
# ==========================================
st.set_page_config(page_title="信号与系统 AI 助教", page_icon="📡", layout="wide")

# 🎨 注入自定义 CSS
st.markdown("""
    <style>
    .block-container {
        max-width: 900px !important; 
        padding-top: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    .page-ref {
        background-color: #e8f0fe;
        border-radius: 12px;
        padding: 8px 16px;
        font-size: 0.9rem;
        color: #1a73e8;
        display: inline-block;
        margin: 4px 0;
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
    text = text.replace('\\[', '$$').replace('\\]', '$$')
    text = text.replace('\\(', '$').replace('\\)', '$')
    st.markdown(text)

# ==========================================
# 3. 登录认证模块
# ==========================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

st.sidebar.title("🔐 用户登录")
if not st.session_state.logged_in:
    username_input = st.sidebar.text_input("👤 用户名")
    password_input = st.sidebar.text_input("🔑 密码", type="password")
    
    if st.sidebar.button("登录"):
        if username_input.strip() and password_input.strip():
            st.session_state.logged_in = True
            st.session_state.username = username_input.strip()
            st.rerun()
        else:
            st.sidebar.error("用户名和密码不能为空！")
    st.stop()
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

PDF_FILE_PATH = "data/signal_and_systems.pdf"

# ==========================================
# 5. 初始化RAG引擎（新增：带页码引用）
# ==========================================
@st.cache_resource
def get_rag_engine(api_key, pdf_path):
    """获取RAG引擎实例"""
    try:
        from rag_engine import get_rag_engine as init_engine
        return init_engine(api_key, pdf_path)
    except Exception as e:
        st.warning(f"⚠️ RAG引擎初始化失败: {e}")
        return None

# 检查PDF是否存在
if not os.path.exists(PDF_FILE_PATH):
    st.sidebar.warning(f"⚠️ 教材PDF不存在: {PDF_FILE_PATH}\n请将教材放在 data/ 目录下")

# ==========================================
# 6. 历史记录模块
# ==========================================
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 历史对话记录")

with st.spinner("🔄 同步历史记忆..."):
    full_history = db_core.load_user_history(st.session_state.username)

available_sessions = list(full_history.get(selected_expert, {}).keys())
if not available_sessions:
    available_sessions = ["默认对话"]

if st.sidebar.button("➕ 新建对话"):
    new_session = datetime.datetime.now().strftime("对话_%m%d_%H%M")
    st.session_state.current_session = new_session
    st.rerun()

if "current_session" not in st.session_state or st.session_state.current_session not in available_sessions:
    if "current_session" in st.session_state and st.session_state.current_session.startswith("对话_"):
        available_sessions.insert(0, st.session_state.current_session)
    else:
        st.session_state.current_session = available_sessions[-1] if available_sessions else "默认对话"

current_session = st.sidebar.selectbox(
    "选择或切换聊天记录",
    options=available_sessions,
    index=available_sessions.index(st.session_state.current_session) if st.session_state.current_session in available_sessions else 0
)
st.session_state.current_session = current_session

messages = full_history.get(selected_expert, {}).get(current_session, [])

# ==========================================
# 7. 专属符号模块
# ==========================================
st.sidebar.markdown("---")

def render_symbol_sidebar():
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

render_symbol_sidebar()

# ==========================================
# 8. 对话渲染与交互
# ==========================================
st.title(selected_expert.split()[1])

for i, msg in enumerate(messages):
    with st.chat_message(msg["role"]):
        if "time" in msg and msg["time"]:
            st.caption(f"🕒 {msg['time']}")
        render_markdown_with_latex(msg["content"])
        if msg["role"] == "assistant":
            with st.expander("📋 点击展开以一键复制全文"):
                st.code(msg["content"], language="markdown")
        if "📊" in selected_expert and msg["role"] == "assistant":
            plot_core.render_interactive_plot(msg["content"], msg_index=f"history_{i}")

# ==========================================
# 9. AI 调用与数据库写入（新增：页码引用）
# ==========================================
if prompt := st.chat_input("输入你的问题，或展开左侧面板复制符号..."):
    
    # 构建历史对话
    history_for_chain = []
    for msg in messages:
        role = "assistant" if msg["role"] == "assistant" else "user"
        history_for_chain.append((role, msg["content"]))
    
    # 智能重命名
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

    bj_tz = timezone(timedelta(hours=8))
    current_time = datetime.datetime.now(bj_tz).strftime("%Y年%m月%d日 %H:%M:%S")

    # 展示用户提问
    with st.chat_message("user"):
        st.caption(f"🕒 {current_time}")
        render_markdown_with_latex(prompt)
        
    # 调用AI生成回复
    with st.chat_message("assistant"):
        with st.spinner("🧠 AI 正在检索教材并思考中..."):
            try:
                api_key = st.secrets["API_KEY"]
                
                # 尝试使用带页码引用的RAG引擎
                rag_engine = get_rag_engine(api_key, PDF_FILE_PATH)
                
                if rag_engine is not None:
                    # 使用新引擎（带页码引用）
                    result = rag_engine.answer_with_references(
                        query=prompt,
                        chat_history=history_for_chain
                    )
                    ai_reply = result["answer"]
                    pages = result.get("pages", [])
                else:
                    # 降级到原RAG系统
                    qa_chain = rag_core.init_rag_system(
                        api_key=api_key, 
                        expert_mode=selected_expert, 
                        pdf_name=PDF_FILE_PATH
                    )
                    response = qa_chain.invoke({
                        "query": prompt,
                        "chat_history": history_for_chain
                    })
                    if isinstance(response, dict) and "result" in response:
                        ai_reply = response["result"]
                    elif isinstance(response, dict) and "text" in response:
                        ai_reply = response["text"]
                    elif hasattr(response, "content"):
                        ai_reply = response.content
                    else:
                        ai_reply = str(response)
                    pages = []
                
                # 展示AI回复
                st.caption(f"🕒 {current_time}")
                render_markdown_with_latex(ai_reply)
                
                # 🌟 新增：显示页码引用
                if pages:
                    st.markdown(f'<div class="page-ref">📚 知识点位于教材第 {", ".join(map(str, pages[:5]))} 页</div>', unsafe_allow_html=True)
                else:
                    st.caption("💡 提示：如需查看教材页码引用，请确保PDF文件在 data/ 目录下")
                
                with st.expander("📋 点击展开以一键复制全文"):
                    st.code(ai_reply, language="markdown")
                
                if "📊" in selected_expert:
                    plot_core.render_interactive_plot(ai_reply, msg_index=f"new_{datetime.datetime.now().timestamp()}")
                    
                # 存入数据库
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
                
        if was_renamed:
            st.rerun()
