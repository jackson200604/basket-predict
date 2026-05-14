# ============================================================
# app.py — Interface Streamlit Dark Mode
# ============================================================
import streamlit as st
import sys
# Compatibilité Python 3.14
import httpx
import asyncio
import plotly.graph_objects as go
from datetime import date

# ── Configuration page ────────────────────────────────────────
st.set_page_config(
    page_title = "🏀 BasketPredictAI",
    page_icon  = "🏀",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── URL API ───────────────────────────────────────────────────
import os
API_URL = st.secrets.get(
    "API_URL",
    os.getenv(
        "API_URL",
        "https://basketpredict-api-production.up.railway.app"
    )
)

# ── CSS Dark Mode Premium ─────────────────────────────────────
st.markdown("""
<style>
/* Fond général */
.stApp {
    background-color: #0a0a0f;
    color: #e8e8f0;
}
/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0d0d1a;
    border-right: 2px solid #FF6B35;
}
/* Bouton principal */
.stButton > button {
    background: linear-gradient(135deg, #FF6B35, #FF9800);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: bold;
    font-size: 16px;
    padding: 12px 32px;
    width: 100%;
    cursor: pointer;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #FF9800, #FF6B35);
    transform: scale(1.02);
}
/* Cards */
.metric-card {
    background: #1a1a2e;
    border: 1px solid #16213e;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin: 8px 0;
}
/* Value Bet badge */
.value-bet {
    background: linear-gradient(135deg, #00ff88, #00cc66);
    color: #000;
    padding: 6px 14px;
    border-radius: 20px;
    font-weight: bold;
    font-size: 13px;
    display: inline-block;
}
/* Titres sections */
.section-title {
    color: #FF6B35;
    font-size: 18px;
    font-weight: bold;
    border-bottom: 2px solid #FF6B35;
    padding-bottom: 6px;
    margin: 20px 0 12px 0;
}
/* Injury badge */
.injury-out      { color: #ff4444; font-weight: bold; }
.injury-doubtful { color: #ffaa00; font-weight: bold; }
.injury-question { color: #ffdd00; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# =============================================================
# SIDEBAR
# =============================================================

with st.sidebar:
    st.markdown("## 🏀 BasketPredictAI")
    st.markdown("*Analyse quantitative par IA*")
    st.divider()

    # Ligues
    league = st.selectbox(
        "🌍 Ligue",
        ["NBA", "EuroLeague", "NCAA", "BSL", "LNB"],
        index=0,
    )

    # Équipes
    st.markdown("### Équipes")
    team_home = st.text_input(
        "🏠 Équipe Domicile",
        placeholder="ex: Los Angeles Lakers",
    )
    team_away = st.text_input(
        "✈️ Équipe Extérieur",
        placeholder="ex: Boston Celtics",
    )

    # Date
    match_date = st.date_input(
        "📅 Date du match",
        value=date.today(),
    )

    # Cotes (optionnel)
    st.markdown("### 💰 Cotes (optionnel)")
    col1, col2 = st.columns(2)
    with col1:
        odds_home = st.number_input("Dom.", value=-110, step=5)
    with col2:
        odds_away = st.number_input("Ext.", value=-110, step=5)

    st.divider()

    # Bouton
    predict_btn = st.button("🔮 ANALYSER", use_container_width=True)

    st.divider()
    st.markdown("""
    <small>
    Propulsé par<br>
    🤖 Gemini 3 Flash<br>
    🔍 Serper.dev<br>
    📄 Jina AI Reader<br>
    📊 Monte Carlo 20K sims
    </small>
    """, unsafe_allow_html=True)


# =============================================================
# PAGE PRINCIPALE
# =============================================================

# Header
st.markdown("""
<div style='text-align:center; padding: 20px 0;'>
    <h1 style='color:#FF6B35; font-size:42px;'>🏀 BasketPredictAI</h1>
    <p style='color:#888; font-size:16px;'>
        Elo Dynamique · Four Factors · Monte Carlo 20K · Value Bet Detection
    </p>
</div>
""", unsafe_allow_html=True)


# =============================================================
# ANALYSE
# =============================================================

if predict_btn:

    # Validation
    if not team_home or not team_away:
        st.error("⚠️ Remplis les deux équipes dans la sidebar !")
        st.stop()

    if team_home.strip().lower() == team_away.strip().lower():
        st.error("⚠️ Les deux équipes doivent être différentes !")
        st.stop()

    # Appel API
    with st.spinner("🔄 Analyse en cours... (20 000 simulations)"):
        try:
            payload = {
                "team_home":  team_home.strip(),
                "team_away":  team_away.strip(),
                "league":     league,
                "match_date": str(match_date),
            }
            with httpx.Client(timeout=60) as client:
                resp = client.post(f"{API_URL}/predict", json=payload)
                resp.raise_for_status()
                data = resp.json()

        except httpx.ConnectError:
            st.error("❌ API non disponible. Vérifie que le backend tourne !")
            st.stop()
        except Exception as e:
            st.error(f"❌ Erreur : {e}")
            st.stop()

    # ── HEADER MATCH ─────────────────────────────────────────
    st.markdown("---")
    col_h, col_vs, col_a = st.columns([2, 1, 2])

    mc = data["monte_carlo"]
    with col_h:
        st.markdown(f"""
        <div class='metric-card'>
            <h2 style='color:#FF6B35;'>{data['team_home']}</h2>
            <h1 style='color:#fff; font-size:48px;'>
                {int(mc['avg_score_home'])}
            </h1>
            <p style='color:#4CAF50; font-size:20px;'>
                {mc['home_win_pct']*100:.1f}% victoire
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_vs:
        st.markdown("""
        <div style='text-align:center; padding-top:40px;'>
            <h2 style='color:#888;'>VS</h2>
            <p style='color:#FF6B35;'>🏀</p>
        </div>
        """, unsafe_allow_html=True)

    with col_a:
        st.markdown(f"""
        <div class='metric-card'>
            <h2 style='color:#2196F3;'>{data['team_away']}</h2>
            <h1 style='color:#fff; font-size:48px;'>
                {int(mc['avg_score_away'])}
            </h1>
            <p style='color:#4CAF50; font-size:20px;'>
                {mc['away_win_pct']*100:.1f}% victoire
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Confiance globale
    conf = data["confidence_global"]
    st.markdown(f"""
    <div style='text-align:center; margin:16px 0;'>
        <span style='color:#888;'>Confiance globale : </span>
        <span style='color:#FF6B35; font-size:20px; font-weight:bold;'>
            {conf*100:.1f}%
        </span>
    </div>
    """, unsafe_allow_html=True)
    st.progress(conf)

    # ── ONGLETS ───────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Analyse", "🔮 Scénarios", "📈 Stats", "💰 Paris"
    ])

    # ── TAB 1 : ANALYSE ──────────────────────────────────────
    with tab1:
        st.markdown("<div class='section-title'>🔑 Facteur Clé</div>",
                    unsafe_allow_html=True)
        st.info(data["key_factor"])

        st.markdown("<div class='section-title'>📋 Historique H2H</div>",
                    unsafe_allow_html=True)
        st.write(data["h2h_summary"])

        # Blessures
        col_inj1, col_inj2 = st.columns(2)
        with col_inj1:
            st.markdown(f"<div class='section-title'>🏥 Blessés {data['team_home']}</div>",
                        unsafe_allow_html=True)
            if data["injuries_home"]:
                for inj in data["injuries_home"]:
                    css = (
                        "injury-out" if inj["status"] == "Out"
                        else "injury-doubtful" if inj["status"] == "Doubtful"
                        else "injury-question"
                    )
                    st.markdown(
                        f"<span class='{css}'>● {inj['player_name']} "
                        f"({inj['status']}) "
                        f"Impact: {inj['impact_pts']:+.1f} pts</span>",
                        unsafe_allow_html=True,
                    )
            else:
                st.success("✅ Aucun blessé signalé")

        with col_inj2:
            st.markdown(f"<div class='section-title'>🏥 Blessés {data['team_away']}</div>",
                        unsafe_allow_html=True)
            if data["injuries_away"]:
                for inj in data["injuries_away"]:
                    css = (
                        "injury-out" if inj["status"] == "Out"
                        else "injury-doubtful" if inj["status"] == "Doubtful"
                        else "injury-question"
                    )
                    st.markdown(
                        f"<span class='{css}'>● {inj['player_name']} "
                        f"({inj['status']}) "
                        f"Impact: {inj['impact_pts']:+.1f} pts</span>",
                        unsafe_allow_html=True,
                    )
            else:
                st.success("✅ Aucun blessé signalé")

        # Graphique probabilités
        st.markdown("<div class='section-title'>📊 Probabilités Monte Carlo</div>",
                    unsafe_allow_html=True)
        fig_prob = go.Figure(go.Bar(
            x     = [data["team_home"], data["team_away"]],
            y     = [mc["home_win_pct"]*100, mc["away_win_pct"]*100],
            marker_color = ["#FF6B35", "#2196F3"],
            text  = [f"{mc['home_win_pct']*100:.1f}%",
                     f"{mc['away_win_pct']*100:.1f}%"],
            textposition = "auto",
        ))
        fig_prob.update_layout(
            paper_bgcolor = "#0a0a0f",
            plot_bgcolor  = "#0a0a0f",
            font_color    = "#e8e8f0",
            yaxis_title   = "Probabilité (%)",
            showlegend    = False,
            height        = 300,
        )
        st.plotly_chart(fig_prob, use_container_width=True)

    # ── TAB 2 : SCÉNARIOS ────────────────────────────────────
    with tab2:
        st.markdown("<div class='section-title'>🔮 Scores Probables</div>",
                    unsafe_allow_html=True)

        for scenario in data["scenarios"]:
            col_s1, col_s2, col_s3 = st.columns([2, 3, 2])
            with col_s1:
                st.markdown(f"**{scenario['label']}**")
            with col_s2:
                st.markdown(
                    f"<h3 style='color:#FF6B35; text-align:center;'>"
                    f"{scenario['score_home']} — {scenario['score_away']}"
                    f"</h3>",
                    unsafe_allow_html=True,
                )
            with col_s3:
                st.markdown(
                    f"<p style='text-align:right; color:#888;'>"
                    f"Confiance: {scenario['confidence']*100:.0f}%</p>",
                    unsafe_allow_html=True,
                )
            st.divider()

    # ── TAB 3 : STATS ────────────────────────────────────────
    with tab3:
        st.markdown("<div class='section-title'>📈 Statistiques Projetées</div>",
                    unsafe_allow_html=True)

        hs = data["home_stats"]
        as_ = data["away_stats"]

        col_st1, col_st2 = st.columns(2)
        with col_st1:
            st.markdown(f"**{data['team_home']}**")
            st.metric("Net Rating",  f"{hs['net_rating']:+.1f}")
            st.metric("Pace",        f"{hs['pace']:.1f}")
            st.metric("eFG%",        f"{hs['efg_pct']*100:.1f}%")
            st.metric("TOV%",        f"{hs['tov_pct']:.1f}%")
            st.metric("PPP",         f"{hs['ppp']:.3f}")
            st.metric("TS%",         f"{hs['ts_pct']*100:.1f}%")

        with col_st2:
            st.markdown(f"**{data['team_away']}**")
            st.metric("Net Rating",  f"{as_['net_rating']:+.1f}")
            st.metric("Pace",        f"{as_['pace']:.1f}")
            st.metric("eFG%",        f"{as_['efg_pct']*100:.1f}%")
            st.metric("TOV%",        f"{as_['tov_pct']:.1f}%")
            st.metric("PPP",         f"{as_['ppp']:.3f}")
            st.metric("TS%",         f"{as_['ts_pct']*100:.1f}%")

        # Radar Chart Four Factors
        st.markdown("<div class='section-title'>🕸️ Four Factors Radar</div>",
                    unsafe_allow_html=True)

        categories = ["eFG%", "TOV%", "ORB%", "FTr", "TS%"]
        fig_radar  = go.Figure()

        fig_radar.add_trace(go.Scatterpolar(
            r    = [
                hs["efg_pct"]*100,
                100 - hs["tov_pct"],
                hs["orb_pct"],
                hs["ftr"]*100,
                hs["ts_pct"]*100,
            ],
            theta = categories,
            fill  = "toself",
            name  = data["team_home"],
            line_color = "#FF6B35",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r    = [
                as_["efg_pct"]*100,
                100 - as_["tov_pct"],
                as_["orb_pct"],
                as_["ftr"]*100,
                as_["ts_pct"]*100,
            ],
            theta = categories,
            fill  = "toself",
            name  = data["team_away"],
            line_color = "#2196F3",
        ))
        fig_radar.update_layout(
            paper_bgcolor = "#0a0a0f",
            plot_bgcolor  = "#0a0a0f",
            font_color    = "#e8e8f0",
            polar         = dict(
                bgcolor   = "#1a1a2e",
                radialaxis = dict(visible=True, range=[0, 100]),
            ),
            height = 400,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── TAB 4 : PARIS ────────────────────────────────────────
    with tab4:
        st.markdown("<div class='section-title'>💰 Recommandations Paris</div>",
                    unsafe_allow_html=True)

        for bet in data["bet_recommendations"]:
            col_b1, col_b2, col_b3 = st.columns([2, 4, 2])
            with col_b1:
                st.markdown(f"**{bet['label']}**")
                st.caption(bet["bet_type"])
            with col_b2:
                st.write(bet["description"])
                if bet["is_value_bet"] and bet.get("edge"):
                    st.markdown(
                        f"<span class='value-bet'>"
                        f"⚡ EDGE +{bet['edge']*100:.1f}%</span>",
                        unsafe_allow_html=True,
                    )
            with col_b3:
                st.metric("Prob.", f"{bet['probability']*100:.1f}%")
            st.divider()

        # Sources
        st.markdown("<div class='section-title'>📡 Sources de données</div>",
                    unsafe_allow_html=True)
        for src in data.get("data_sources", []):
            st.markdown(f"✅ {src}")

# ── État initial ──────────────────────────────────────────────
else:
    st.markdown("""
    <div style='text-align:center; padding:60px 20px; color:#555;'>
        <h2>👈 Remplis les équipes dans la sidebar</h2>
        <p>puis clique <strong style='color:#FF6B35;'>ANALYSER</strong></p>
        <br>
        <p>🤖 Gemini AI · 🔍 Serper · 📄 Jina · 📊 20K simulations</p>
    </div>
    """, unsafe_allow_html=True)
