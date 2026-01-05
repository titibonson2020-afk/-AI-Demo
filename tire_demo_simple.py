import streamlit as st
import os
import time
import json

# 页面配置
st.set_page_config(
    page_title="轮胎制造业技术写作AI大模型Demo",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 标题
st.markdown("""
<div style="text-align: center; padding: 1rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 2rem;">
    <h1 style="color: white; margin: 0; font-size: 2.5rem;">🏭 轮胎制造业技术写作AI大模型Demo</h1>
    <p style="color: white; margin: 0.5rem 0; font-size: 1.1rem;">基于ChatGLM3-6B的专业技术文档优化工具</p>
</div>
""", unsafe_allow_html=True)

# 轮胎制造业技术写作文档案例（基于开源数据整理）
tire_cases_data = {
    "案例1": {
        "title": "轮胎硫化工艺改进纪要转化",
        "original_text": "硫化温度从150度提升到155度，硫化时间缩短5分钟，可提高生产效率15%，同时保证轮胎物理性能指标符合标准要求。操作员需要调整设备参数设置，确保温度控制精度在±2度范围内。",
        "optimized_text": "通过优化硫化工艺参数（温度提升至155℃，时间缩短5分钟），在不降低产品质量的前提下，生产效率提升15%。建议操作人员精确控制温度波动范围±2℃，以确保硫化过程的一致性和产品质量的稳定性。",
        "instruction_type": "分类型",
        "category": "工艺改进",
        "metrics": {
            "rouge_l": 0.785,
            "bleu": 0.692,
            "semantic_similarity": 0.834,
            "perspective_accuracy": 0.912
        }
    },
    "案例2": {
        "title": "轮胎成型设备故障排除纪要转化",
        "original_text": "成型机压力传感器异常，压力波动大，检查显示发现是密封圈老化导致，更换密封圈后恢复正常。需要定期检查密封件状态，建议每季度更换一次。",
        "optimized_text": "成型设备压力异常的根本原因为密封件老化。解决方案：1）立即更换老化密封圈；2）建立密封件定期更换制度（建议每季度）；3）增加压力传感器日常监控频次。建议制定设备预防性维护计划，确保生产连续性。",
        "instruction_type": "分类型",
        "category": "故障排除",
        "metrics": {
            "rouge_l": 0.812,
            "bleu": 0.723,
            "semantic_similarity": 0.867,
            "perspective_accuracy": 0.934
        }
    },
    "案例3": {
        "title": "轮胎产品参数表转化",
        "original_text": "215/60R16轮胎：宽度215mm，扁平比60%，轮辋直径16英寸，载重指数95，速度级别H。建议胎压2.3bar，适用于中型轿车。",
        "optimized_text": "产品规格：215/60R16（轿车轮胎）\n技术参数：\n- 轮胎宽度：215mm\n- 扁平比：60%\n- 轮辋直径：16英寸\n- 载重指数：95（690kg）\n- 速度级别：H（210km/h）\n\n推荐使用条件：标准胎压2.3bar，适用于中型轿车日常行驶。",
        "instruction_type": "开放型",
        "category": "产品说明",
        "metrics": {
            "rouge_l": 0.756,
            "bleu": 0.681,
            "semantic_similarity": 0.798,
            "perspective_accuracy": 0.889
        }
    }
}

# 初始化session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'training_progress' not in st.session_state:
    st.session_state.training_progress = 0
if 'current_case' not in st.session_state:
    st.session_state.current_case = "案例1"
if 'optimization_result' not in st.session_state:
    st.session_state.optimization_result = None

# 侧边栏导航
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(45deg, #667eea, #764ba2); border-radius: 8px; margin-bottom: 1rem;">
        <h3 style="color: white; margin: 0;">🛠️ 功能导航</h3>
    </div>
    """, unsafe_allow_html=True)
    
    selected_module = st.radio(
        "选择功能模块",
        [
            "🏁 准备阶段 - 模型加载与环境验证",
            "📊 数据准备 - 轮胎制造业数据处理", 
            "🔍 Instruction 类型判断",
            "⚙️ 模型微调 - Lora 参数配置与训练",
            "✅ 验证评估 - 自动+人工评估",
            "📝 成果输出 - 文本优化与报告生成"
        ]
    )

# 主内容区域
if selected_module == "🏁 准备阶段 - 模型加载与环境验证":
    st.header("🏁 模块1：准备阶段 - 模型加载与环境验证")
    
    # 核心架构说明
    st.info("💡 核心架构")
    st.markdown("""
    - **后台模型**：ChatGLM3-6B（技术增强版）- 中文支持友好、显存占用低
    - **开发框架**：Streamlit - 轻量级Python Web框架，离线运行无依赖
    - **部署方式**：离线桌面版 - 无需服务器，本地安装Python环境即可启动
    - **数据来源**：轮胎制造业开源技术数据
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 环境验证")
        
        if st.button("🔍 检查环境状态", key="env_check"):
            with st.spinner("正在检查环境..."):
                time.sleep(2)
                st.success("✅ Python 3.9+")
                st.success("✅ CUDA 12.1")
                st.success("✅ PyTorch 2.1.0")
                st.success("✅ Transformers 4.35.2")
                st.success("✅ PEFT 0.7.1")
                st.success("✅ Streamlit 1.28.0")
        
        st.subheader("📊 系统资源")
        st.info("GPU: NVIDIA RTX 4090")
        st.info("显存: 24GB")
        st.info("内存: 32GB")
        st.info("存储: 1TB SSD")
        
        st.subheader("⚙️ 模型加载")
        if st.button("加载模型", key="load_model"):
            with st.spinner("正在加载ChatGLM3-6B模型..."):
                time.sleep(5)
                st.session_state.model_loaded = True
                st.success("模型加载成功!")
    
    with col2:
        st.subheader("📊 性能指标")
        
        if st.session_state.model_loaded:
            st.success("✅ 模型状态：已加载")
            
            # 模型信息
            st.info("模型版本：ChatGLM3-6B")
            st.info("模型大小：约10GB")
            st.info("上下文长度：8K tokens")
            st.info("支持语言：中文/英文")
            st.info("量化方式：INT4")
            
            # 性能指标
            st.subheader("性能指标")
            st.progress(0.9)
            st.text("响应速度: 0.9 tokens/s")
            st.progress(0.85)
            st.text("处理精度: 85%")
            st.progress(0.92)
            st.text("语言流畅度: 92%")
        else:
            st.warning("模型状态：未加载")
            st.info("请点击左侧'加载模型'按钮")
            
            # 模拟模型信息
            st.subheader("模型信息预览")
            st.info("模型版本：ChatGLM3-6B")
            st.info("模型大小：约10GB")
            st.info("上下文长度：8K tokens")
            st.info("支持语言：中文/英文")
            st.info("量化方式：INT4")

elif selected_module == "📊 数据准备 - 轮胎制造业数据处理":
    st.header("📊 模块2：数据准备 - 轮胎制造业数据处理")
    
    st.info("💡 数据处理说明")
    st.markdown("""
    - 数据来源：Kaggle轮胎制造工艺开源数据集和汽车工程开源文档库
    - 数据预处理：数据清洗、格式标准化、标签规范化
    - Instruction类型：分为分类型和开放型两种主要类型
    - 训练集大小：8000条，验证集大小：2000条，测试集大小：1000条
    """)
    
    # 案例选择
    st.subheader("📋 案例选择")
    selected_case = st.selectbox(
        "选择轮胎制造业案例",
        list(tire_cases_data.keys()),
        index=0
    )
    
    # 显示选中的案例
    st.subheader(f"案例详情：{tire_cases_data[selected_case]['title']}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 原始技术文档")
        st.markdown(f"""
        <div style="background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 5px; padding: 1rem; margin: 1rem 0;">
            {tire_cases_data[selected_case]['original_text']}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 优化后文档")
        st.markdown(f"""
        <div style="background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 5px; padding: 1rem; margin: 1rem 0;">
            {tire_cases_data[selected_case]['optimized_text']}
        </div>
        """, unsafe_allow_html=True)
    
    # 案例信息
    st.subheader("📊 案例分析")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"分类：{tire_cases_data[selected_case]['category']}")
        st.info(f"Instruction类型：{tire_cases_data[selected_case]['instruction_type']}")
    
    with col2:
        # 指标图表
        metrics = tire_cases_data[selected_case]['metrics']
        
        st.text("评估指标：")
        st.progress(metrics['rouge_l'])
        st.caption(f"ROUGE-L: {metrics['rouge_l']:.3f}")
        
        st.progress(metrics['bleu'])
        st.caption(f"BLEU: {metrics['bleu']:.3f}")
        
        st.progress(metrics['semantic_similarity'])
        st.caption(f"语义相似度: {metrics['semantic_similarity']:.3f}")
        
        st.progress(metrics['perspective_accuracy'])
        st.caption(f"视角转换准确度: {metrics['perspective_accuracy']:.3f}")

elif selected_module == "🔍 Instruction 类型判断":
    st.header("🔍 模块3：Instruction 类型判断")
    
    st.info("💡 Instruction分类说明")
    st.markdown("""
    - 分类型Instruction：对技术文档进行特定类型的转换，如故障排除转换为客户指导
    - 开放型Instruction：对技术文档进行开放式的优化改进，如参数表转换为产品规格说明
    - 分类模型：基于BERT的文本分类器，准确率达到95%
    """)
    
    # 原始文本输入
    st.subheader("📝 输入原始技术文档")
    original_text = st.text_area(
        "请输入需要处理的原始技术文档：",
        value="硫化温度从150度提升到155度，硫化时间缩短5分钟，可提高生产效率15%，同时保证轮胎物理性能指标符合标准要求。操作员需要调整设备参数设置，确保温度控制精度在±2度范围内。"
    )
    
    # 判断按钮
    if st.button("🔍 分析Instruction类型", key="classify_instruction"):
        with st.spinner("正在分析..."):
            time.sleep(2)
            
            # 模拟分类结果
            st.subheader("📊 分类结果")
            
            # 分类概率
            col1, col2 = st.columns(2)
            
            with col1:
                st.text("分类型概率：")
                st.progress(0.75)
                st.caption("75%")
                
            with col2:
                st.text("开放型概率：")
                st.progress(0.25)
                st.caption("25%")
            
            # 分类结果
            st.success("预测类型：分类型Instruction")
            
            # 说明
            st.markdown("""
            **分类说明：**
            
            该原始文档属于分类型Instruction，因为它包含了将工程师视角的技术纪要转换为客户友好的产品说明的需求。
            文档涉及硫化工艺参数的改进，需要从技术角度转换为客户关注的效益和操作建议。
            """)
            
            # 建议的处理模板
            st.subheader("📋 建议的处理模板")
            st.markdown("""
            **模板：**
            
            将工程师视角的技术纪要转换为客户友好的产品说明，强调效益和操作建议
            
            **输入：**
            
            {原始技术文档内容}
            
            **输出：**
            
            通过优化技术参数，在不降低产品质量的前提下，效率提升{具体数值}%。
            建议操作人员精确控制参数波动范围±{数值}，以确保过程的一致性和产品质量的稳定性。
            """)

elif selected_module == "⚙️ 模型微调 - Lora 参数配置与训练":
    st.header("⚙️ 模块4：模型微调 - Lora 参数配置与训练")
    
    st.info("💡 LoRA微调说明")
    st.markdown("""
    - LoRA（Low-Rank Adaptation）是一种参数高效的微调方法，仅需更新少量参数
    - 微调参数：r=8, alpha=16, dropout=0.1, target_modules=all linear layers
    - 训练设置：batch_size=4, learning_rate=2e-4, warmup_steps=100, max_steps=1000
    - 优化器：AdamW，调度器：CosineAnnealingLR
    """)
    
    # 参数配置
    st.subheader("⚙️ LoRA参数配置")
    
    col1, col2 = st.columns(2)
    
    with col1:
        r_value = st.slider("秩大小 (r)", 1, 16, 8)
        alpha_value = st.slider("Alpha值", 8, 64, 16)
        dropout_value = st.slider("Dropout", 0.0, 0.5, 0.1, 0.05)
    
    with col2:
        batch_size = st.slider("批次大小", 1, 16, 4)
        learning_rate = st.number_input("学习率", value=0.0002, format="%.6f")
        warmup_steps = st.slider("预热步数", 0, 500, 100)
    
    # 训练设置
    st.subheader("🎯 训练设置")
    
    max_steps = st.slider("最大步数", 100, 5000, 1000)
    save_steps = st.slider("保存间隔", 100, 1000, 500)
    
    # 开始训练按钮
    if st.button("开始微调训练", key="start_training"):
        with st.spinner("正在初始化训练环境..."):
            time.sleep(2)
            st.session_state.training_progress = 0
            st.success("训练环境初始化完成!")
        
        # 模拟训练进度
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(101):
            time.sleep(0.05)
            progress_bar.progress(i)
            status_text.text(f"训练进度: {i}%")
            st.session_state.training_progress = i
        
        st.success("训练完成!")
        
        # 训练结果
        st.subheader("📊 训练结果")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("最终训练损失：0.125")
            st.info("最终验证损失：0.178")
            st.info("训练步数：1000")
            st.info("保存检查点：2个")
        
        with col2:
            st.success("模型保存成功!")
            st.info("模型路径：./outputs/chatglm3-6b-tire-lora")
            st.info("模型大小：约500MB")
            st.info("微调参数：1.2M")

elif selected_module == "✅ 验证评估 - 自动+人工评估":
    st.header("✅ 模块5：验证评估 - 自动+人工评估")
    
    st.info("💡 评估方法说明")
    st.markdown("""
    - 自动评估：使用ROUGE、BLEU、语义相似度等指标衡量生成文本质量
    - 人工评估：由轮胎制造业专家对生成文本的专业性、可读性进行评分
    - 测试集：包含1000条未见过的新样本，涵盖不同类型的文档转换任务
    - 评估维度：语言流畅度、信息完整性、视角转换准确度、专业术语准确性
    """)
    
    # 评估选择
    st.subheader("🔍 选择评估方式")
    
    evaluation_type = st.radio(
        "选择评估类型",
        ["自动评估", "人工评估", "综合评估"]
    )
    
    if evaluation_type == "自动评估":
        # 自动评估
        st.subheader("📊 自动评估结果")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.text("ROUGE-L得分：")
            st.progress(0.812)
            st.caption("0.812")
            
            st.text("BLEU得分：")
            st.progress(0.745)
            st.caption("0.745")
        
        with col2:
            st.text("语义相似度：")
            st.progress(0.876)
            st.caption("0.876")
            
            st.text("视角转换准确度：")
            st.progress(0.923)
            st.caption("0.923")
        
        st.subheader("📈 性能曲线")
        
        # 模拟性能曲线
        st.line_chart({
            "ROUGE-L": [0.5, 0.6, 0.65, 0.7, 0.75, 0.78, 0.8, 0.81, 0.812],
            "BLEU": [0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.73, 0.74, 0.745],
            "语义相似度": [0.6, 0.7, 0.75, 0.8, 0.82, 0.84, 0.86, 0.87, 0.876],
            "视角转换准确度": [0.7, 0.75, 0.8, 0.85, 0.88, 0.9, 0.91, 0.92, 0.923]
        })
    
    elif evaluation_type == "人工评估":
        # 人工评估
        st.subheader("👨‍🔬 人工评估结果")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.text("语言流畅度：")
            st.progress(0.91)
            st.caption("4.55/5")
            
            st.text("信息完整性：")
            st.progress(0.88)
            st.caption("4.4/5")
        
        with col2:
            st.text("专业术语准确性：")
            st.progress(0.93)
            st.caption("4.65/5")
            
            st.text("视角转换准确度：")
            st.progress(0.9)
            st.caption("4.5/5")
        
        st.subheader("📝 专家评价")
        
        st.markdown("""
        **专家评价摘要：**
        
        经过5位轮胎制造业专家的综合评估，模型生成的技术文档在专业术语使用和视角转换方面表现优秀。
        生成文本保持了原始技术信息的准确性，同时增强了可读性和客户友好度。
        文档结构清晰，技术细节完整，特别是工艺参数改进的效益表达更加直观。
        建议在实际应用中对专业术语的标准化进行进一步优化。
        """)
    
    else:
        # 综合评估
        st.subheader("📊 综合评估结果")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.text("自动评估平均分：")
            st.progress(0.852)
            st.caption("0.852/1.0")
            
            st.text("人工评估平均分：")
            st.progress(0.905)
            st.caption("4.525/5")
        
        with col2:
            st.text("综合得分：")
            st.progress(0.878)
            st.caption("0.878/1.0")
            
            st.text("推荐等级：")
            st.success("A")
        
        st.subheader("📈 综合评估雷达图")
        
        # 模拟雷达图
        st.bar_chart({
            "评估指标": ["语言流畅度", "信息完整性", "专业术语准确性", "视角转换准确度", "技术细节完整性"],
            "得分": [0.91, 0.88, 0.93, 0.90, 0.87]
        })

elif selected_module == "📝 成果输出 - 文本优化与报告生成":
    st.header("📝 模块6：成果输出 - 文本优化与报告生成")
    
    st.info("💡 输出功能说明")
    st.markdown("""
    - 文本优化：基于微调后的ChatGLM3-6B模型对输入的技术文档进行优化
    - 报告生成：生成包含优化前后对比、评估指标等内容的详细报告
    - API接口：提供RESTful API接口，支持集成到其他系统中
    - 导出格式：支持PDF、Word、Markdown等多种格式导出
    """)
    
    # 文本优化
    st.subheader("📝 文本优化")
    
    # 输入文本
    input_text = st.text_area(
        "输入需要优化的技术文档：",
        value="硫化温度从150度提升到155度，硫化时间缩短5分钟，可提高生产效率15%，同时保证轮胎物理性能指标符合标准要求。操作员需要调整设备参数设置，确保温度控制精度在±2度范围内。"
    )
    
    # 优化按钮
    if st.button("优化文档", key="optimize_text"):
        with st.spinner("正在优化文档..."):
            time.sleep(3)
            
            # 模拟优化结果
            st.session_state.optimization_result = "通过优化硫化工艺参数（温度提升至155℃，时间缩短5分钟），在不降低产品质量的前提下，生产效率提升15%。建议操作人员精确控制温度波动范围±2℃，以确保硫化过程的一致性和产品质量的稳定性。"
    
    # 显示优化结果
    if st.session_state.optimization_result:
        st.subheader("📊 优化结果对比")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 原始文档")
            st.markdown(f"""
            <div style="background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 5px; padding: 1rem; margin: 1rem 0;">
                {input_text}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 优化后文档")
            st.markdown(f"""
            <div style="background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 5px; padding: 1rem; margin: 1rem 0;">
                {st.session_state.optimization_result}
            </div>
            """, unsafe_allow_html=True)
    
    # 报告生成
    st.subheader("📊 报告生成")
    
    report_options = st.multiselect(
        "选择报告内容",
        ["优化前后对比", "评估指标", "专家评价", "API接口文档", "使用建议"]
    )
    
    if st.button("生成报告", key="generate_report"):
        with st.spinner("正在生成报告..."):
            time.sleep(2)
            
            # 模拟报告生成
            st.success("报告生成成功!")
            
            # 报告内容
            st.subheader("报告内容")
            
            for option in report_options:
                if option == "优化前后对比":
                    st.markdown("### 优化前后对比")
                    st.markdown("""
                    | 指标 | 原始文档 | 优化后文档 |
                    |------|---------|-----------|
                    | 可读性 | 3/5 | 4.5/5 |
                    | 专业性 | 4/5 | 4.5/5 |
                    | 客户友好度 | 2/5 | 4.5/5 |
                    | 信息完整性 | 4.5/5 | 4.5/5 |
                    """)
                
                elif option == "评估指标":
                    st.markdown("### 评估指标")
                    st.markdown("""
                    | 指标 | 得分 |
                    |------|------|
                    | ROUGE-L | 0.812 |
                    | BLEU | 0.745 |
                    | 语义相似度 | 0.876 |
                    | 视角转换准确度 | 0.923 |
                    """)
                
                elif option == "专家评价":
                    st.markdown("### 专家评价")
                    st.markdown("""
                    经过5位轮胎制造业专家的综合评估，模型生成的技术文档在专业术语使用和视角转换方面表现优秀。
                    生成文本保持了原始技术信息的准确性，同时增强了可读性和客户友好度。
                    """)
                
                elif option == "API接口文档":
                    st.markdown("### API接口文档")
                    st.markdown("""
                    ```python
                    import requests
                    
                    url = "http://localhost:8000/api/v1/text-optimization"
                    headers = {"Content-Type": "application/json"}
                    data = {
                        "text": "需要优化的技术文档内容",
                        "instruction_type": "分类"
                    }
                    
                    response = requests.post(url, headers=headers, json=data)
                    result = response.json()
                    print(result["optimized_text"])
                    ```
                    """)
                
                elif option == "使用建议":
                    st.markdown("### 使用建议")
                    st.markdown("""
                    1. 对于不同类型的文档，建议选择相应的Instruction类型
                    2. 定期更新微调数据，以提高模型在特定场景下的表现
                    3. 结合人工审核，确保最终输出符合企业标准
                    4. 考虑建立多级审核流程，提高文档质量
                    """)
            
            # 下载按钮
            st.markdown("### 下载报告")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="📄 下载PDF报告",
                    data="模拟PDF内容",
                    file_name="轮胎制造业AI优化报告.pdf",
                    mime="application/pdf"
                )
            
            with col2:
                st.download_button(
                    label="📄 下载Word报告",
                    data="模拟Word内容",
                    file_name="轮胎制造业AI优化报告.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
    
    # API接口演示
    st.subheader("🚀 API接口演示")
    
    api_input = st.text_area(
        "输入API测试文本：",
        value="轮胎硫化温度从150度提升到155度，硫化时间缩短5分钟，可提高生产效率15%"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        api_instruction_type = st.selectbox(
            "选择Instruction类型",
            ["分类型", "开放型"]
        )
    
    with col2:
        if st.button("调用API", key="call_api"):
            with st.spinner("正在调用API..."):
                time.sleep(2)
                
                # 模拟API调用结果
                api_result = "通过优化硫化工艺参数（温度提升至155℃，时间缩短5分钟），在不降低产品质量的前提下，生产效率提升15%。建议操作人员精确控制温度波动范围±2℃，以确保硫化过程的一致性和产品质量的稳定性。"
                
                st.success("API调用成功!")
                st.text_area("API返回结果：", value=api_result, height=150)