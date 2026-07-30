import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.ticker as ticker

# ----------------------------------------------------------------------------
# 1. CONFIGURAÇÃO INICIAL
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Meta-Analysis App - Sleep & Cannabis", layout="wide")

if "lang" not in st.session_state:
    st.session_state.lang = "en"  # Padrão: inglês

# ----------------------------------------------------------------------------
# 2. FUNÇÕES COMPARTILHADAS
# ----------------------------------------------------------------------------
def fmt_num(value, decimals=3):
    """Formata números com vírgula decimal no PT e ponto no EN."""
    lang = st.session_state.lang
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        formatted = f"{value:.{decimals}f}"
        if lang == "pt":
            formatted = formatted.replace(".", ",")
        return formatted
    return str(value)

def run_meta_analysis(df):
    """Executa o modelo de efeitos aleatórios de DerSimonian-Laird."""
    if df.empty:
        return None
    d = df["d"].values
    se = df["se"].values
    k = len(d)
    w = 1 / (se**2)
    sum_w = np.sum(w)
    sum_wd = np.sum(w * d)
    sum_wd2 = np.sum(w * (d**2))
    Q = sum_wd2 - (sum_wd**2) / sum_w
    df_het = k - 1
    if df_het > 0:
        tau2 = max(0, (Q - df_het) / (sum_w - (np.sum(w**2) / sum_w)))
    else:
        tau2 = 0
    w_star = 1 / (se**2 + tau2)
    sum_w_star = np.sum(w_star)
    pooled_d = np.sum(w_star * d) / sum_w_star
    se_pooled = np.sqrt(1 / sum_w_star)
    ci_lb = pooled_d - 1.96 * se_pooled
    ci_ub = pooled_d + 1.96 * se_pooled
    z = pooled_d / se_pooled
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))
    i2 = max(0, ((Q - df_het) / Q) * 100) if Q > 0 else 0
    weights_percent = (w_star / sum_w_star) * 100
    return {
        "k": k,
        "pooled_d": pooled_d,
        "se_pooled": se_pooled,
        "ci_lb": ci_lb,
        "ci_ub": ci_ub,
        "p_val": p_val,
        "tau2": tau2,
        "i2": i2,
        "q": Q,
        "df": df_het,
        "weights": weights_percent,
        "w_star": w_star,
    }

def plot_forest(df, results):
    """Forest Plot com textos traduzidos e números formatados."""
    if df.empty or results is None:
        return None
    lang = st.session_state.lang
    fig, ax = plt.subplots(figsize=(10, 6))
    df_sorted = df.copy()
    df_sorted["weight"] = results["weights"]
    df_sorted = df_sorted.sort_values("d", ascending=False).reset_index(drop=True)
    y_pos = np.arange(len(df_sorted))
    d_values = df_sorted["d"].values
    se_values = df_sorted["se"].values
    ci_lower = d_values - 1.96 * se_values
    ci_upper = d_values + 1.96 * se_values
    ax.hlines(y=y_pos, xmin=ci_lower, xmax=ci_upper, color="gray", linewidth=1, alpha=0.7)
    sizes = (df_sorted["weight"] / df_sorted["weight"].max()) * 100 + 20
    ax.scatter(d_values, y_pos, s=sizes, color="#1f77b4", zorder=5, edgecolors="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted["id"].tolist(), fontsize=9)
    ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(x=results["pooled_d"], color="red", linestyle="-", linewidth=1.5, alpha=0.6)
    x_min = min(-0.5, ci_lower.min() - 0.2) if len(ci_lower) > 0 else -0.5
    x_max = max(2.5, ci_upper.max() + 0.2) if len(ci_upper) > 0 else 2.5
    ax.set_xlim(x_min, x_max)

    if lang == "en":
        ax.set_xlabel("Cohen's d", fontsize=10)
        ax.set_title("Forest Plot", fontsize=12)
        label_pooled = "Pooled"
    else:
        ax.set_xlabel("d de Cohen", fontsize=10)
        ax.set_title("Gráfico de Floresta", fontsize=12)
        label_pooled = "Combinado"
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: fmt_num(x, 2)))

    diamond_y = -0.5
    diamond_x = results["pooled_d"]
    ax.plot(
        [results["ci_lb"], diamond_x, results["ci_ub"], diamond_x, results["ci_lb"]],
        [diamond_y, diamond_y - 0.2, diamond_y, diamond_y + 0.2, diamond_y],
        color="red",
        linewidth=2,
    )
    ax.text(
        diamond_x,
        diamond_y - 0.5,
        f"{label_pooled}: {fmt_num(results['pooled_d'], 3)} [{fmt_num(results['ci_lb'], 3)}, {fmt_num(results['ci_ub'], 3)}]",
        ha="center",
        fontsize=9,
        color="red",
        fontweight="bold",
    )
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    return fig

def plot_funnel(df, results):
    """Funnel Plot com textos traduzidos."""
    if df.empty or results is None:
        return None
    lang = st.session_state.lang
    fig, ax = plt.subplots(figsize=(8, 6))
    d = df["d"].values
    se = df["se"].values
    ax.scatter(d, se, color="#1f77b4", zorder=5, edgecolors="black", linewidth=0.5)

    if lang == "en":
        ax.set_xlabel("Cohen's d", fontsize=10)
        ax.set_ylabel("Standard Error", fontsize=10)
        ax.set_title("Funnel Plot", fontsize=12)
    else:
        ax.set_xlabel("d de Cohen", fontsize=10)
        ax.set_ylabel("Erro Padrão", fontsize=10)
        ax.set_title("Gráfico de Funil (Viés de Publicação)", fontsize=12)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: fmt_num(x, 2)))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, p: fmt_num(y, 2)))

    ax.axvline(x=results["pooled_d"], color="red", linestyle="-", linewidth=1.0, alpha=0.5)
    x_limits = np.linspace(results["ci_lb"] - 0.5, results["ci_ub"] + 0.5, 100)
    y_limits = (np.max(se) / (results["ci_ub"] - results["ci_lb"] + 1)) * np.abs(x_limits - results["pooled_d"])
    ax.fill_between(x_limits, 0, y_limits, color="gray", alpha=0.1)
    ax.invert_yaxis()
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    return fig

def plot_sensitivity(df, results):
    """Sensitivity Plot com textos traduzidos."""
    if df.empty or results is None or len(df) < 2:
        return None
    lang = st.session_state.lang
    d_orig = results["pooled_d"]
    studies = df["id"].tolist()
    d_loo = []
    for i in range(len(df)):
        df_loo = df.drop(df.index[i])
        res_loo = run_meta_analysis(df_loo)
        if res_loo is not None:
            d_loo.append(res_loo["pooled_d"])
        else:
            d_loo.append(np.nan)
    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(studies))
    ax.scatter(d_loo, y_pos, color="#1f77b4", zorder=5, edgecolors="black", linewidth=0.5)

    if lang == "en":
        ax.set_xlabel("Cohen's d (leave-one-out)", fontsize=10)
        ax.set_title("Sensitivity Analysis (Leave-One-Out)", fontsize=12)
        ax.axvline(x=d_orig, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="Original pooled d")
    else:
        ax.set_xlabel("d de Cohen (leave-one-out)", fontsize=10)
        ax.set_title("Análise de Sensibilidade (Leave-One-Out)", fontsize=12)
        ax.axvline(x=d_orig, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="d combinado original")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: fmt_num(x, 2)))

    ax.set_yticks(y_pos)
    ax.set_yticklabels(studies, fontsize=8)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig

def render_comparator(results, lang, df_selected):
    """
    Renderiza a tabela comparativa "Sua Dose Atual vs. Estudos".
    """
    # Inicializa os estados dos inputs do comparador
    if "comp_thc_conc" not in st.session_state:
        st.session_state.comp_thc_conc = 32.0
    if "comp_quantity" not in st.session_state:
        st.session_state.comp_quantity = 0.05

    st.markdown("---")
    with st.expander("📊 Sua Dose Atual vs. Estudos" if lang == "pt" else "📊 Your Current Dose vs. Studies"):
        # Explicação inicial (neutra e clínica)
        if lang == "pt":
            st.markdown("""
            Compare a quantidade de THC que você utiliza atualmente com as doses empregadas nos estudos científicos 
            incluídos nesta meta-análise. Preencha os dados do seu produto abaixo.
            """)
        else:
            st.markdown("""
            Compare the amount of THC you currently use with the doses employed in the scientific studies 
            included in this meta-analysis. Fill in your product data below.
            """)

        # Inputs do usuário
        col1, col2 = st.columns(2)
        with col1:
            thc_conc = st.number_input(
                "THC (%)" if lang == "pt" else "THC (%)",
                min_value=0.0, max_value=100.0, value=st.session_state.comp_thc_conc, step=0.5,
                key="comp_thc_conc_input",
                help="Concentração de THC no seu produto (ex: 32%)" if lang == "pt" else "THC concentration in your product (e.g., 32%)"
            )
            st.session_state.comp_thc_conc = thc_conc
        with col2:
            quantity = st.number_input(
                "Quantidade diária (g)" if lang == "pt" else "Daily amount (g)",
                min_value=0.001, max_value=10.0, value=st.session_state.comp_quantity, step=0.001, format="%.3f",
                key="comp_quantity_input",
                help="Quantidade do produto que você usa por dia (ex: 0.05g)" if lang == "pt" else "Amount of product you use per day (e.g., 0.05g)"
            )
            st.session_state.comp_quantity = quantity

        if thc_conc > 0 and quantity > 0:
            # Cálculo do THC ingerido
            thc_mg_per_g = thc_conc * 10  # 32% = 320 mg/g
            user_thc_mg = thc_mg_per_g * quantity

            st.markdown("---")
            st.markdown(f"**🧪 {fmt_num(user_thc_mg, 1)} mg** de THC por dia." if lang == "pt" else f"**🧪 {fmt_num(user_thc_mg, 1)} mg** of THC per day.")

            # Montar dados da tabela
            dados_tabela = []

            # Linha do usuário (sempre a primeira) - com destaque visual
            dados_tabela.append({
                "Fonte" if lang == "pt" else "Source": "🟢 **" + ("Sua Dose Atual" if lang == "pt" else "Your Current Dose") + "**",
                "Dose (mg)" if lang == "pt" else "Dose (mg)": f"{user_thc_mg:.1f}",
                "Comparação" if lang == "pt" else "Comparison": "🔹 " + ("(Referência)" if lang == "pt" else "(Reference)")
            })

            # Estudos de referência (com valores fixos e faixas)
            estudos_ref = [
                {"nome": "Pakdee et al. (2026)", "dose": 2.5, "tipo": "fixa", "desc": "Dose baixa (óleo Tailândia)"},
                {"nome": "Ried et al. (2023)", "dose": 15.0, "tipo": "fixa", "desc": "Dose moderada (óleo Entoura)"},
                {"nome": "Montebello et al. (2022)", "dose_min": 11.0, "dose_max": 19.0, "tipo": "faixa", "desc": "Nabiximols spray 1:1"},
                {"nome": "UK Registries (média)", "dose": 20.0, "tipo": "fixa", "desc": "Datta, Erridge, Vivek"},
                {"nome": "Erridge et al. (2026)", "dose": 60.0, "tipo": "fixa", "desc": "Dose alta (24 meses)"},
            ]

            for estudo in estudos_ref:
                if estudo["tipo"] == "fixa":
                    dose_float = estudo["dose"]
                    if abs(dose_float - user_thc_mg) < 1.0:
                        comparacao = "✅ " + ("Muito Próxima" if lang == "pt" else "Very Close")
                    elif dose_float > user_thc_mg:
                        comparacao = "⬆️ " + ("Maior" if lang == "pt" else "Higher") + " (" + ("sua dose é menor" if lang == "pt" else "your dose is lower") + ")"
                    else:
                        comparacao = "⬇️ " + ("Menor" if lang == "pt" else "Lower") + " (" + ("sua dose é maior" if lang == "pt" else "your dose is higher") + ")"
                    dose_str = f"{dose_float:.1f}"
                else:  # faixa
                    if estudo["dose_min"] <= user_thc_mg <= estudo["dose_max"]:
                        comparacao = "✅ " + ("Dentro da Faixa" if lang == "pt" else "Within Range")
                    elif user_thc_mg < estudo["dose_min"]:
                        comparacao = "⬆️ " + ("Abaixo da Faixa" if lang == "pt" else "Below Range")
                    else:
                        comparacao = "⬇️ " + ("Acima da Faixa" if lang == "pt" else "Above Range")
                    dose_str = f"{estudo['dose_min']:.1f} – {estudo['dose_max']:.1f}"

                dados_tabela.append({
                    "Fonte" if lang == "pt" else "Source": estudo["nome"] + " (" + estudo["desc"] + ")",
                    "Dose (mg)" if lang == "pt" else "Dose (mg)": dose_str,
                    "Comparação" if lang == "pt" else "Comparison": comparacao
                })

            # Exibir tabela
            df_tabela = pd.DataFrame(dados_tabela)
            st.dataframe(df_tabela, use_container_width=True, hide_index=True)

            # Interpretação simples e ética
            if lang == "pt":
                st.caption("💡 Esta comparação mostra onde sua dose atual se posiciona em relação à literatura científica. Não recomendamos alterações na dose sem orientação médica.")
            else:
                st.caption("💡 This comparison shows where your current dose stands relative to the scientific literature. We do not recommend dose changes without medical supervision.")
        else:
            st.info("👈 Insira a concentração e a quantidade diária para ver a comparação." if lang == "pt" else "👈 Enter the concentration and daily amount to see the comparison.")

# ----------------------------------------------------------------------------
# 3. BASE DE DADOS (10 ESTUDOS)
# ----------------------------------------------------------------------------
STUDIES = [
    {"id": "Erridge et al. (2026)", "type": "Coorte", "n": 8945, "d": 0.84, "se": 0.009},
    {"id": "Pakdee, Sribunrieng e Poowanna (2026)", "type": "RCT", "n": 20, "d": 2.84, "se": 0.635},
    {"id": "Datta et al. (2025a)", "type": "Coorte", "n": 517, "d": 0.68, "se": 0.030},
    {"id": "Datta et al. (2025b)", "type": "Coorte", "n": 269, "d": 0.32, "se": 0.020},
    {"id": "Short et al. (2025)", "type": "Coorte", "n": 137, "d": 0.80, "se": 0.068},
    {"id": "Vivek et al. (2024)", "type": "Coorte", "n": 40, "d": 1.35, "se": 0.213},
    {"id": "Cooke et al. (2023)", "type": "Coorte", "n": 163, "d": 0.55, "se": 0.043},
    {"id": "Vaddiparti et al. (2023)", "type": "Coorte", "n": 15, "d": 1.44, "se": 0.372},
    {"id": "Ried et al. (2023)", "type": "RCT", "n": 29, "d": 0.69, "se": 0.128},
    {"id": "Montebello et al. (2022)", "type": "RCT", "n": 128, "d": 0.60, "se": 0.181},
]

# ----------------------------------------------------------------------------
# 4. INTERFACE PRINCIPAL
# ----------------------------------------------------------------------------
def main():
    lang = st.session_state.lang

    # ======================================================================
    # INICIALIZAÇÃO DOS CHECKBOXES (TODOS MARCADOS)
    # ======================================================================
    if "initialized" not in st.session_state:
        for idx in range(len(STUDIES)):
            key = f"include_{idx}"
            st.session_state[key] = True
        st.session_state.initialized = True

    # ======================================================================
    # SIDEBAR
    # ======================================================================
    with st.sidebar:
        st.header("Language / Idioma")
        lang_choice = st.radio(
            "",
            options=["English", "Português"],
            index=0 if lang == "en" else 1,
            label_visibility="collapsed",
        )
        if lang_choice == "Português" and lang != "pt":
            st.session_state.lang = "pt"
            st.rerun()
        elif lang_choice == "English" and lang != "en":
            st.session_state.lang = "en"
            st.rerun()

        st.divider()

        if lang == "en":
            st.subheader("ℹ️ About this App")
            st.info("Select/deselect studies to see the pooled effect change. Uses Random-Effects (DL).\n\nFormatted for: **EN**")
            st.caption("Developed for academic research purposes.")
            st.divider()
            st.subheader("⚙️ Statistical Method")
            st.markdown("**DerSimonian-Laird (DL)** random-effects model.\n\nWeights = within-study + between-studies variance (Tau²).")
        else:
            st.subheader("ℹ️ Sobre este App")
            st.info("Selecione/desselecione estudos para ver o efeito combinado mudar. Usa Efeitos Aleatórios (DL).\n\nFormatado para: **PT-BR**")
            st.caption("Desenvolvido para fins acadêmicos.")
            st.divider()
            st.subheader("⚙️ Método Estatístico")
            st.markdown("Modelo de efeitos aleatórios **DerSimonian-Laird (DL)**.\n\nPesos = variância intra + variância entre estudos (Tau²).")

    # ======================================================================
    # CORPO (INGLÊS)
    # ======================================================================
    if lang == "en":
        st.title("🗂️ Interactive Meta-Analysis: THC/Cannabis & Sleep")
        st.caption("Based on Tait et al. (2026) - Quality of Life Research")

        df_studies = pd.DataFrame(STUDIES)
        st.subheader("📊 Select Studies to Include")
        st.caption("**RCT** = Randomized Controlled Trial (gold standard, participants are randomly assigned); **Cohort** = Observational study (participants are followed without controlled intervention).")

        col1, col2, col3, col4, col5, col6 = st.columns([0.6, 3.5, 1.5, 1, 1.5, 1.5])
        col1.markdown("<div style='text-align: center; font-weight: bold;'>Include</div>", unsafe_allow_html=True)
        col2.markdown("<div style='text-align: left; font-weight: bold;'>Study</div>", unsafe_allow_html=True)
        col3.markdown("<div style='text-align: center; font-weight: bold;'>Type</div>", unsafe_allow_html=True)
        col4.markdown("<div style='text-align: center; font-weight: bold;'>N</div>", unsafe_allow_html=True)
        col5.markdown("<div style='text-align: center; font-weight: bold;'>Cohen's d</div>", unsafe_allow_html=True)
        col6.markdown("<div style='text-align: center; font-weight: bold;'>Std. Error</div>", unsafe_allow_html=True)

        selected_ids = []
        for idx, row in df_studies.iterrows():
            key = f"include_{idx}"
            if key not in st.session_state:
                st.session_state[key] = True
            c1, c2, c3, c4, c5, c6 = st.columns([0.6, 3.5, 1.5, 1, 1.5, 1.5])
            checked = c1.checkbox("", value=st.session_state[key], key=key, label_visibility="collapsed")
            if checked:
                selected_ids.append(idx)
            c2.write(row["id"])
            c3.write(row["type"])
            c4.write(f"{row['n']:,}")
            c5.write(f"{row['d']:.2f}")
            c6.write(f"{row['se']:.3f}")

        df_selected = df_studies.loc[selected_ids].copy()
        st.caption(f"**{len(df_selected)}** out of **{len(df_studies)}** selected.")

        if len(df_selected) >= 2:
            results = run_meta_analysis(df_selected)
        else:
            results = None
            st.warning("Select at least **2 studies**.")

        if results:
            with st.sidebar:
                st.subheader("📊 Current Results")
                st.metric("Q-statistic", f"{fmt_num(results['q'], 2)}")
                st.metric("I²", f"{fmt_num(results['i2'], 1)} %")
                st.metric("Tau²", f"{fmt_num(results['tau2'], 4)}")

            st.divider()
            st.subheader("📈 Meta-Analysis Results")
            st.markdown("""
            **What do these numbers mean?**  
            - **Pooled Effect (d):** The average effect across all selected studies.  
            - **95% CI:** The range where the true effect likely lies.  
            - **P-value:** Tells us if the effect is statistically significant (p < 0.05 means "yes").  
            - **I²:** Measures how much the studies differ from each other (0% = identical, 100% = completely different).
            """)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Pooled Effect (d)", fmt_num(results["pooled_d"], 3))
            c2.metric("95% CI", f"[{fmt_num(results['ci_lb'], 3)}, {fmt_num(results['ci_ub'], 3)}]")
            c3.metric("P-value", "< 0.001" if results["p_val"] < 0.001 else fmt_num(results["p_val"], 3))
            c4.metric("I²", f"{fmt_num(results['i2'], 1)} %")

            st.divider()
            st.subheader("📋 Study Weights")
            st.markdown("""
            **What are study weights?**  
            Studies with smaller error bars (more precise estimates) get more weight in the final average. Larger squares in the Forest Plot correspond to higher weights.
            """)

            df_weights = df_selected.copy()
            df_weights["Weight (%)"] = [fmt_num(w, 2) for w in results["weights"]]
            df_display = df_weights[["id", "type", "n", "d", "se", "Weight (%)"]]
            df_display["n"] = df_display["n"].apply(lambda x: f"{x:,}")
            df_display["d"] = df_display["d"].apply(lambda x: f"{x:.2f}")
            df_display["se"] = df_display["se"].apply(lambda x: f"{x:.3f}")
            st.dataframe(df_display, use_container_width=True)
            st.caption("Larger squares = greater weight.")

            st.divider()
            st.subheader("🌲 Forest Plot")
            fig = plot_forest(df_selected, results)
            if fig:
                st.pyplot(fig)
                st.caption("Right-click to save.")
                st.caption("**What this shows:** Each horizontal line represents the confidence interval of a single study. The square's size indicates its weight in the meta-analysis. The red diamond at the bottom shows the overall pooled effect. If the diamond does not cross the vertical zero line, the effect is statistically significant.")

            st.divider()
            st.subheader("🔍 Funnel Plot (Publication Bias)")
            fig = plot_funnel(df_selected, results)
            if fig:
                st.pyplot(fig)
                st.caption("Asymmetry may indicate publication bias.")
                st.caption("**What this shows:** Each dot is a study. If the plot is symmetric (like an inverted funnel), it suggests no publication bias. If studies are missing on one side, there might be bias (e.g., small studies with negative results not published).")

            st.divider()
            st.subheader("📉 Sensitivity Analysis (Leave-One-Out)")
            fig = plot_sensitivity(df_selected, results)
            if fig:
                st.pyplot(fig)
                st.caption("If effect remains stable, result is robust.")
                st.caption("**What this shows:** Each point shows the pooled effect after removing that study. If all points stay close to the red line, the result is stable and not driven by any single study. Large deviations suggest that study is overly influential.")

            st.divider()
            st.subheader("📖 Interpretation Guide")
            # --- INTERPRETAÇÃO DINÂMICA (INGLÊS) ---
            d = results["pooled_d"]
            i2 = results["i2"]
            p = results["p_val"]
            ci_lb = results["ci_lb"]
            ci_ub = results["ci_ub"]
            effect_desc = "small" if abs(d) < 0.2 else "moderate" if abs(d) < 0.5 else "large"
            het_desc = "low" if i2 < 25 else "moderate" if i2 < 50 else "high"
            
            st.markdown(f"""
            **Cohen's d = {fmt_num(d, 3)}** – This means the treatment improved sleep by **{abs(d):.1f} standard deviations** compared to the control.  
            This is a **{effect_desc}** effect, indicating a **{'clinically relevant' if abs(d) >= 0.5 else 'modest'}** improvement in sleep quality.
            
            **What does this mean in practice?**  
            - **d < 0.2:** Small effect (hardly noticeable).  
            - **d ≈ 0.5:** Moderate effect (clearly noticeable).  
            - **d ≥ 0.8:** Large effect (substantial and clinically meaningful).
            """)
            st.markdown(f"**I² = {fmt_num(i2, 1)}%** – **{het_desc}** heterogeneity.")
            st.markdown(f"**p = {fmt_num(p, 3)}** – {'Significant' if p < 0.05 else 'Not significant'} (p < 0.05).")
            contains_zero = "does" if ci_lb < 0 < ci_ub else "does not"
            st.markdown(f"95% CI [{fmt_num(ci_lb, 3)}, {fmt_num(ci_ub, 3)}] {contains_zero} contain zero.")

            # ===== COMPARADOR (INGLÊS) =====
            render_comparator(results, lang, df_selected)

        else:
            st.info("👈 Select more studies.")

    # ======================================================================
    # CORPO (PORTUGUÊS)
    # ======================================================================
    else:
        st.title("🗂️ Meta-Análise Interativa: THC/Cannabis & Sono")
        st.caption("Baseado em Tait et al. (2026) - Quality of Life Research")

        df_studies = pd.DataFrame(STUDIES)
        st.subheader("📊 Selecione os Estudos para Incluir")
        st.caption("**ECR** = Ensaio Clínico Randomizado (padrão ouro, participantes são sorteados); **Coorte** = Estudo observacional (participantes são acompanhados sem intervenção controlada).")

        col1, col2, col3, col4, col5, col6 = st.columns([0.6, 3.5, 1.5, 1, 1.5, 1.5])
        col1.markdown("<div style='text-align: center; font-weight: bold;'>Incluir</div>", unsafe_allow_html=True)
        col2.markdown("<div style='text-align: left; font-weight: bold;'>Estudo</div>", unsafe_allow_html=True)
        col3.markdown("<div style='text-align: center; font-weight: bold;'>Tipo</div>", unsafe_allow_html=True)
        col4.markdown("<div style='text-align: center; font-weight: bold;'>N</div>", unsafe_allow_html=True)
        col5.markdown("<div style='text-align: center; font-weight: bold;'>d de Cohen</div>", unsafe_allow_html=True)
        col6.markdown("<div style='text-align: center; font-weight: bold;'>Erro Padrão</div>", unsafe_allow_html=True)

        selected_ids = []
        for idx, row in df_studies.iterrows():
            key = f"include_{idx}"
            if key not in st.session_state:
                st.session_state[key] = True
            c1, c2, c3, c4, c5, c6 = st.columns([0.6, 3.5, 1.5, 1, 1.5, 1.5])
            checked = c1.checkbox("", value=st.session_state[key], key=key, label_visibility="collapsed")
            if checked:
                selected_ids.append(idx)

            tipo_traduzido = "Coorte" if row["type"] == "Coorte" else "ECR"
            c2.write(row["id"])
            c3.write(tipo_traduzido)
            c4.write(f"{row['n']:,}".replace(",", "."))
            c5.write(fmt_num(row["d"], 2))
            c6.write(fmt_num(row["se"], 3))

        df_selected = df_studies.loc[selected_ids].copy()
        st.caption(f"**{len(df_selected)}** de **{len(df_studies)}** selecionados.")

        if len(df_selected) >= 2:
            results = run_meta_analysis(df_selected)
        else:
            results = None
            st.warning("Selecione pelo menos **2 estudos**.")

        if results:
            with st.sidebar:
                st.subheader("📊 Resultados Atuais")
                st.metric("Estatística Q", fmt_num(results["q"], 2))
                st.metric("I²", f"{fmt_num(results['i2'], 1)} %")
                st.metric("Tau²", fmt_num(results["tau2"], 4))

            st.divider()
            st.subheader("📈 Resultados da Meta-Análise")
            st.markdown("""
            **O que significam esses números?**  
            - **Efeito Combinado (d):** A média dos efeitos de todos os estudos selecionados.  
            - **IC 95%:** O intervalo onde provavelmente está o efeito verdadeiro.  
            - **Valor-p:** Indica se o efeito é estatisticamente significativo (p < 0,05 significa "sim").  
            - **I²:** Mede o quanto os estudos diferem entre si (0% = idênticos, 100% = completamente diferentes).
            """)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Efeito Combinado (d)", fmt_num(results["pooled_d"], 3))
            c2.metric("IC 95%", f"[{fmt_num(results['ci_lb'], 3)}, {fmt_num(results['ci_ub'], 3)}]")
            c3.metric("Valor-p", "< 0,001" if results["p_val"] < 0.001 else fmt_num(results["p_val"], 3))
            c4.metric("I²", f"{fmt_num(results['i2'], 1)} %")

            st.divider()
            st.subheader("📋 Pesos dos Estudos")
            st.markdown("""
            **O que são os pesos dos estudos?**  
            Estudos com barras de erro menores (estimativas mais precisas) recebem mais peso na média final. Quadrados maiores no Forest Plot correspondem a pesos maiores.
            """)

            df_weights = df_selected.copy()
            df_weights["Peso (%)"] = [fmt_num(w, 2) for w in results["weights"]]
            df_display = df_weights[["id", "type", "n", "d", "se", "Peso (%)"]]
            df_display["n"] = df_display["n"].apply(lambda x: f"{x:,}".replace(",", "."))
            df_display["d"] = df_display["d"].apply(lambda x: fmt_num(x, 2))
            df_display["se"] = df_display["se"].apply(lambda x: fmt_num(x, 3))
            df_display = df_display.rename(columns={
                "id": "Estudo",
                "type": "Tipo",
                "n": "N",
                "d": "d de Cohen",
                "se": "Erro Padrão",
                "Peso (%)": "Peso (%)"
            })
            st.dataframe(df_display, use_container_width=True)
            st.caption("Quadrados maiores = maior peso.")

            st.divider()
            st.subheader("🌲 Forest Plot")
            fig = plot_forest(df_selected, results)
            if fig:
                st.pyplot(fig)
                st.caption("Clique com o direito para salvar.")
                st.caption("**O que mostra:** Cada linha horizontal é o intervalo de confiança de um estudo. O tamanho do quadrado indica seu peso na meta-análise. O losango vermelho na parte inferior mostra o efeito combinado geral. Se o losango não cruzar a linha vertical zero, o efeito é estatisticamente significativo.")

            st.divider()
            st.subheader("🔍 Gráfico de Funil (Viés de Publicação)")
            fig = plot_funnel(df_selected, results)
            if fig:
                st.pyplot(fig)
                st.caption("Assimetria pode indicar viés de publicação.")
                st.caption("**O que mostra:** Cada ponto é um estudo. Se o gráfico for simétrico (como um funil invertido), sugere que não há viés de publicação. Se faltarem estudos de um lado, pode haver viés (ex: estudos pequenos com resultados negativos não publicados).")

            st.divider()
            st.subheader("📉 Análise de Sensibilidade (Leave-One-Out)")
            fig = plot_sensitivity(df_selected, results)
            if fig:
                st.pyplot(fig)
                st.caption("Se o efeito se mantém estável, o resultado é robusto.")
                st.caption("**O que mostra:** Cada ponto mostra o efeito combinado após remover aquele estudo. Se todos os pontos ficarem próximos da linha vermelha, o resultado é estável e não é influenciado por um único estudo. Grandes desvios sugerem que aquele estudo é muito influente.")

            st.divider()
            st.subheader("📖 Guia de Interpretação")
            # --- INTERPRETAÇÃO DINÂMICA (PORTUGUÊS) ---
            d = results["pooled_d"]
            i2 = results["i2"]
            p = results["p_val"]
            ci_lb = results["ci_lb"]
            ci_ub = results["ci_ub"]
            effect_desc = "pequeno" if abs(d) < 0.2 else "moderado" if abs(d) < 0.5 else "grande"
            het_desc = "baixa" if i2 < 25 else "moderada" if i2 < 50 else "alta"
            
            st.markdown(f"""
            **d de Cohen = {fmt_num(d, 3)}** – Isso significa que o tratamento melhorou o sono em **{abs(d):.1f} desvios padrão** em comparação com o controle.  
            Este é um efeito **{effect_desc}**, indicando uma melhora **{'clinicamente relevante' if abs(d) >= 0.5 else 'modesta'}** na qualidade do sono.

            **O que isso significa na prática?**  
            - **d < 0,2:** Efeito pequeno (dificilmente perceptível).  
            - **d ≈ 0,5:** Efeito moderado (claramente perceptível).  
            - **d ≥ 0,8:** Efeito grande (substancial e clinicamente significativo).
            """)
            st.markdown(f"**I² = {fmt_num(i2, 1)}%** – Heterogeneidade **{het_desc}**.")
            st.markdown(f"**p = {fmt_num(p, 3)}** – {'Significativo' if p < 0.05 else 'Não significativo'} (p < 0,05).")
            contains_zero = "sim" if ci_lb < 0 < ci_ub else "não"
            st.markdown(f"IC 95% [{fmt_num(ci_lb, 3)}, {fmt_num(ci_ub, 3)}] {contains_zero} contém zero.")

            # ===== COMPARADOR (PORTUGUÊS) =====
            render_comparator(results, lang, df_selected)

        else:
            st.info("👈 Selecione mais estudos.")

if __name__ == "__main__":
    main()
