import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="铁路客运站弹性配置系统", layout="wide")
st.title("🚄 基于旅客发送量的铁路客运站安检通道与候车容量弹性配置系统")

# ===== 左侧参数面板 =====
with st.sidebar:
    st.header("⚙️ 参数配置面板")
    with st.expander("📊 客流反推参数", expanded=True):
        k_h = st.slider("高峰小时集中系数 k_h", 0.1, 0.5, 0.25, 0.01)
        peak_h = st.slider("高峰持续小时数", 1, 5, 2, 1)
        open_h = st.slider("车站日开放小时数", 10, 24, 16, 1)
    with st.expander("🔍 安检通道参数", expanded=True):
        svc = st.slider("单机安检服务时间(秒/人)", 5.0, 10.0, 7.0, 0.5)
        fast = st.slider("快速通道分流比例(%)", 0, 50, 20, 5)
        mach = st.number_input("现场既有安检机数(台)", 1, 20, 3, 1)
        snake = st.slider("蛇形围栏效率系数", 0.5, 1.0, 0.85, 0.05)
    with st.expander("🪑 候车区与闸机参数", expanded=True):
        area = st.number_input("候车区有效面积(㎡)", 1000, 100000, 3000, 100)
        p_area = st.slider("人均候车面积标准(㎡/人)", 1.5, 4.5, 2.0, 0.1)
        gate = st.number_input("闸机通道数(个)", 1, 30, 8, 1)
        gate_spd = st.slider("单闸机通行速度(人/秒，实际约600-900人/小时)", 0.1, 0.4, 0.2, 0.05)
    
    st.markdown("---")
    st.subheader("🔮 预测与配置基准")
    pred_days = st.slider("预测未来天数", 1, 14, 7, 1)
    trend_window = st.slider("趋势拟合窗口(天)", 14, 120, 60, 7,
                             help="越小越看重近期变化，预测越贴合当前水平")
    base_select = st.radio("弹性配置基准", ["最新实际日", "预测明日", "预测峰值日"])

# ===== 读取数据 =====
st.markdown("---")
try:
    df = pd.read_excel("data.xlsx")
    df.columns = [c.strip() for c in df.columns]
    date_col, send_col = "date", "send_vol"
    df[date_col] = pd.to_datetime(df[date_col])
    df[send_col] = pd.to_numeric(df[send_col], errors='coerce')
    df.dropna(subset=[date_col, send_col], inplace=True)
    df = df.sort_values(date_col).reset_index(drop=True)

    weekday_map = dict(enumerate(['周一','周二','周三','周四','周五','周六','周日']))
    df["星期"] = df[date_col].dt.dayofweek.map(weekday_map)
    weekday_order = list(weekday_map.values())
    df["星期"] = pd.Categorical(df["星期"], categories=weekday_order, ordered=True)
    df["7日均值"] = df[send_col].rolling(7).mean()

    st.success(f"✅ 已加载 {len(df)} 条数据（{df[date_col].min():%Y-%m-%d} 至 {df[date_col].max():%Y-%m-%d}）")

    # ===== 预测函数：近期加权趋势 + 星期季节性 =====
    def predict_flow(data, days=pred_days, window=trend_window):
        recent = data.tail(window).reset_index(drop=True)
        x = np.arange(len(recent))
        y = recent[send_col].values
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        overall_mean = data[send_col].mean()
        wd_factors = {}
        for i in range(7):
            wd_data = data[data[date_col].dt.dayofweek == i][send_col]
            wd_mean = wd_data.mean() if len(wd_data) > 0 else overall_mean
            wd_factors[i] = wd_mean / overall_mean if overall_mean > 0 else 1.0
        last_date = data[date_col].max()
        pred_dates = [last_date + pd.Timedelta(days=i+1) for i in range(days)]
        pred_values = []
        for d in pred_dates:
            base_idx = len(recent) - 1
            trend_val = p(base_idx + len(pred_values))
            wd = d.dayofweek
            season_factor = wd_factors.get(wd, 1.0)
            pred_val = trend_val * season_factor
            pred_values.append(max(0, int(pred_val)))
        return pd.DataFrame({date_col: pred_dates, "预测发送量": pred_values})

    pred_df = predict_flow(df, days=pred_days, window=trend_window)

    # ===== 顶部 KPI 卡 =====
    latest = df[send_col].iloc[-1]
    avg = df[send_col].mean()
    peak = df[send_col].max()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📅 最新日发送量", f"{int(latest):,} 人")
    c2.metric("📈 全期日均", f"{int(avg):,} 人")
    c3.metric("🚀 历史峰值", f"{int(peak):,}", f"{(df[send_col].idxmax()+1)}条")
    c4.metric("🔮 预测明日", f"{int(pred_df['预测发送量'].iloc[0]):,} 人")
    c5.metric("📅 预测峰值", f"{int(pred_df['预测发送量'].max()):,} 人",
               f"第{(pred_df['预测发送量'].idxmax()+1)}天")

    # ===== 图表区 =====
    g1, g2 = st.columns(2)
    with g1:
        st.subheader("📈 日发送量趋势与预测")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df[date_col], y=df[send_col], name="历史日发送量",
                      line=dict(color="#3b82f6", width=2)))
        fig1.add_trace(go.Scatter(x=df[date_col], y=df["7日均值"], name="7日均值",
                      line=dict(color="#22d3ee", width=2, dash="dot")))
        fig1.add_trace(go.Scatter(x=pred_df[date_col], y=pred_df["预测发送量"], name="未来预测",
                      line=dict(color="#f59e0b", width=2.5, dash="dash")))
        fig1.update_layout(height=360, margin=dict(l=0,r=0,t=20,b=0), legend=dict(orientation="h"))
        st.plotly_chart(fig1, use_container_width=True)
    with g2:
        st.subheader("📊 星期分组均值")
        wday_mean = df.groupby("星期")[send_col].mean().reindex(weekday_order)
        fig2 = px.bar(x=weekday_order, y=wday_mean.values, labels={"x":"星期","y":"日均发送量"})
        fig2.update_traces(marker_color="#22d3ee", marker_line_color="#3b82f6", marker_line_width=1.5)
        fig2.update_layout(xaxis_type="category", height=360, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig2, use_container_width=True)

    # ===== 预测明细表 =====
    st.markdown("---")
    st.subheader("📋 未来逐日客流预测明细")
    pred_df_disp = pred_df.copy()
    pred_df_disp["日期"] = pred_df_disp[date_col].dt.strftime("%Y-%m-%d")
    pred_df_disp["星期"] = pred_df_disp[date_col].dt.dayofweek.map(weekday_map)
    pred_df_disp = pred_df_disp[["日期", "星期", "预测发送量"]]
    pred_df_disp.index = pred_df_disp.index + 1
    pred_df_disp.index.name = "序号"
    st.dataframe(pred_df_disp, use_container_width=True)

    # ===== 弹性配置计算结果 =====
    st.markdown("---")
    st.subheader("🧮 弹性配置计算结果")
    if base_select == "最新实际日":
        base_flow = latest; base_label = "最新实际日"
    elif base_select == "预测明日":
        base_flow = pred_df["预测发送量"].iloc[0]; base_label = "预测明日"
    else:
        base_flow = pred_df["预测发送量"].max(); base_label = "预测峰值日"
    
    peak_hour_flow = base_flow * k_h / peak_h
    eff_svc = svc * (1 - fast/100) / snake
    need_mach = peak_hour_flow * eff_svc / 3600
    wait_cap = area / p_area
    gate_cap_peak = gate * gate_spd * 3600       # 高峰1小时闸机通行能力
    gate_cap = gate * gate_spd * 3600 * open_h   # 全日能力（供说明用）

    st.info(f"🎯 当前配置基准：**{base_label}**（发送量 {int(base_flow):,} 人）", icon="📌")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("⏱️ 高峰小时客流", f"{int(peak_hour_flow):,} 人")
    m2.metric("🛂 理论需安检机", f"{need_mach:.1f} 台", f"现有 {mach} 台")
    m3.metric("🪑 候车区容量", f"{int(wait_cap):,} 人", f"人均 {p_area} ㎡")
    m4.metric("🚪 闸机高峰小时能力", f"{int(gate_cap_peak):,} 人/小时", f"{gate} 通道")

    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown("#### 🛂 安检通道")
        ratio = min(mach / need_mach, 1.0) if need_mach > 0 else 1.0
        st.progress(ratio); st.caption(f"配置率 {ratio*100:.1f}%")
        if mach >= need_mach:
            st.success("安检能力充足 ✅")
        else:
            st.warning(f"建议增配 {int(need_mach-mach)+1} 台 ⚠️")
    
    with r2:
        st.markdown("#### 🪑 候车区容量")
        ratio2 = min(peak_hour_flow / wait_cap, 1.0) if wait_cap > 0 else 1.0
        st.progress(ratio2); st.caption(f"高峰小时承载力 {ratio2 * 100:.1f}%")
        if peak_hour_flow <= wait_cap:
            st.success("候车空间充裕 ✅")
        else:
            st.error("候车面积不足，需扩容 ❌")

    with r3:
        st.markdown("#### 🚪 闸机通行")
        ratio3 = min(peak_hour_flow / gate_cap_peak, 1.0) if gate_cap_peak > 0 else 1.0
        st.progress(ratio3); st.caption(f"高峰小时利用率 {ratio3 * 100:.1f}%")
        if peak_hour_flow <= gate_cap_peak:
            st.success("闸机通行顺畅 ✅")
        else:
            st.warning("高峰闸机可能拥堵 ⚠️")

    with st.expander("📋 配置说明明细"):
        st.write(f"配置基准：**{base_label}**（发送量 {int(base_flow):,} 人）")
        st.write(f"趋势拟合窗口：最近 {trend_window} 天；预测 {pred_days} 天")
        st.write("预测方法：近期加权线性趋势外推 + 星期季节性因子")
        st.write(f"高峰小时客流：**{int(peak_hour_flow):,} 人**（{peak_h} 小时内，集中系数 {k_h}）")
        st.write(f"单机等效服务时间：{eff_svc:.2f} 秒/人（已计入快速通道分流 {fast}%、蛇形效率 {snake}）")
        st.write(f"安检：需 {need_mach:.1f} 台 vs 现有 {mach} 台")
        st.write(f"候车：可容纳 {int(wait_cap):,} 人（{area}㎡÷{p_area}㎡/人）vs 高峰小时 {int(peak_hour_flow):,} 人")
        st.write(f"闸机：高峰小时通行 {int(gate_cap_peak):,} 人（{gate}通道×{gate_spd}人/秒×3600）vs 高峰小时 {int(peak_hour_flow):,} 人")
        st.write(f"闸机：全日通行 {int(gate_cap):,} 人（{open_h}小时开放）")

except FileNotFoundError:
    st.error("找不到 data.xlsx 或数据格式错误！")
except Exception as e:
    st.error(f"读取数据时发生错误: {e}")