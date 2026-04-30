from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app.services.aspect_service import analyze_aspects, extract_main_aspect
from app.services.mode_router import detect_mode
from app.services.pain_service import mine_pain_points
from app.services.predict_service import get_model_status, get_system, predict_batch, predict_single, warmup_inference


@st.cache_resource
def load_system():
    system = get_system()
    best_model = warmup_inference(system)
    return system, best_model


st.set_page_config(page_title="电商评论文本情感分析系统", page_icon="📊", layout="wide")
st.title("电商评论文本情感分析系统")
st.caption("支持单条评论预测、批量评论分析、细粒度分析与用户痛点挖掘")

if "results" not in st.session_state:
    st.session_state["results"] = None
if "comments" not in st.session_state:
    st.session_state["comments"] = []
if "mode" not in st.session_state:
    st.session_state["mode"] = None

system, default_model = load_system()
model_status = get_model_status(system)

input_type = st.radio("请选择输入方式", ["手动输入", "上传文件"], horizontal=True)
comments: list[str] = []

if input_type == "手动输入":
    text = st.text_area("请输入评论文本", height=150)
    if text.strip():
        comments = [text.strip()]

if input_type == "上传文件":
    uploaded_file = st.file_uploader("上传 CSV 或 TXT 文件", type=["csv", "txt"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            st.write("文件预览：")
            st.dataframe(df.head(), use_container_width=True)
            if "content" in df.columns:
                comments = df["content"].dropna().astype(str).tolist()
            else:
                st.error("CSV 文件中未找到 content 列")
        elif uploaded_file.name.endswith(".txt"):
            text = uploaded_file.read().decode("utf-8")
            comments = [line.strip() for line in text.splitlines() if line.strip()]

comment_count = len(comments)
mode = detect_mode(comment_count)
st.session_state["comments"] = comments
st.session_state["mode"] = mode

if mode == "single":
    st.info("当前为单条评论模式")
elif mode == "small_batch":
    st.info(f"当前为基础批量模式，共 {comment_count} 条评论")
elif mode == "full_batch":
    st.info(f"当前为完整批量模式，共 {comment_count} 条评论")

with st.sidebar:
    st.header("参数设置")
    model_options = ["BERT", "TextCNN", "SVM"]
    model_labels = [f"{name}（未加载）" if not model_status.get(name, False) else name for name in model_options]
    ready_options = [name for name in model_options if model_status.get(name, False)]
    default_choice = default_model if default_model in ready_options else (ready_options[0] if ready_options else "SVM")
    selected_index = model_options.index(default_choice)
    selected_label = st.selectbox("选择模型", model_labels, index=selected_index)
    model_name = model_options[model_labels.index(selected_label)]
    if not model_status.get(model_name, False):
        st.warning(f"{model_name} 当前不可用，请先训练并加载该模型权重。")

    single_task = "情感预测"
    show_aspect = False
    do_aspect = True
    do_fine_grained = True
    do_pain = True

    if mode == "single":
        single_task = st.radio("分析类型", ["情感预测", "情感预测 + 主方面识别"])
    if mode == "small_batch":
        show_aspect = st.checkbox("输出方面分布统计", value=False)
    if mode == "full_batch":
        do_aspect = st.checkbox("输出方面分布图", value=True)
        do_fine_grained = st.checkbox("输出细粒度分析", value=True)
        do_pain = st.checkbox("输出痛点挖掘", value=True)
        st.caption("样本<10时痛点挖掘将自动跳过")

    run_btn = st.button("开始分析", use_container_width=True)

if run_btn:
    try:
        if mode == "empty":
            st.warning("请先输入评论或上传文件")
        elif mode == "single":
            text = comments[0]
            sentiment_result = predict_single(system=system, text=text, model_name=model_name)
            result = {"mode": "single", "text": text, "sentiment": sentiment_result}
            if single_task == "情感预测 + 主方面识别":
                result["aspect"] = extract_main_aspect(system, text)
            st.session_state["results"] = result
        elif mode == "small_batch":
            batch_result = predict_batch(system=system, comments=comments, model_name=model_name)
            result = {
                "mode": "small_batch",
                "predictions": batch_result["predictions"],
                "label_counts": batch_result["label_counts"],
                "distribution_fig": batch_result["distribution_fig"],
            }
            if show_aspect:
                result["aspect_distribution"] = analyze_aspects(system, comments, model_name)
            st.session_state["results"] = result
        elif mode == "full_batch":
            batch_result = predict_batch(system=system, comments=comments, model_name=model_name)
            result = {
                "mode": "full_batch",
                "predictions": batch_result["predictions"],
                "label_counts": batch_result["label_counts"],
                "distribution_fig": batch_result["distribution_fig"],
            }
            if do_aspect or do_fine_grained:
                result["aspect_analysis"] = analyze_aspects(system, comments, model_name)
            if do_pain:
                result["pain_analysis"] = mine_pain_points(system, comments, model_name)
            st.session_state["results"] = result
    except Exception as exc:
        st.error(str(exc))

results = st.session_state.get("results")
if results and results["mode"] == "single":
    st.subheader("分析结果")
    st.write("评论文本：", results["text"])
    st.success(f"情感预测结果：{results['sentiment']['predicted_label']}")
    if "aspect" in results:
        st.info(f"主方面识别结果：{results['aspect']}")

if results and results["mode"] in ["small_batch", "full_batch"]:
    tab1, tab2, tab3, tab4 = st.tabs(["预测结果", "情感分布", "方面分析", "痛点挖掘"])
    with tab1:
        st.subheader("评论情感预测结果")
        st.dataframe(pd.DataFrame(results["predictions"]), use_container_width=True)
    with tab2:
        st.subheader("情感分布图")
        distribution_fig = results.get("distribution_fig")
        if distribution_fig and Path(distribution_fig).exists():
            st.image(distribution_fig)
        else:
            st.info("暂无情感分布图")
    with tab3:
        if "aspect_analysis" in results or "aspect_distribution" in results:
            st.subheader("方面分析结果")
            aspect_result = results.get("aspect_distribution") or results.get("aspect_analysis")
            if aspect_result:
                st.dataframe(pd.DataFrame(aspect_result.get("aspect_distribution", [])), use_container_width=True)
        else:
            st.info("当前未开启方面分析")
    with tab4:
        if "pain_analysis" in results:
            st.subheader("用户痛点挖掘结果")
            st.json(results["pain_analysis"])
        else:
            st.info("当前未开启痛点挖掘")
