# ============================================================
# data_ingestion.py — Serper + Jina + Gemini + API réelles
# ============================================================
from typing import Optional
import httpx
import json
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
import google.generativeai as genai
from config import (
    GEMINI_API_KEY, SERPER_API_KEY, JINA_BASE_URL,
    GEMINI_MODEL, GEMINI_MAX_TOKENS,
)
from models import TeamStats, InjuryReport
from api_clients import (
    get_nba_team_stats, get_nba_team_form,
    get_tank01_injuries, get_tank01_team_stats,
)

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(GEMINI_MODEL)


# =============================================================
# 1. SERPER
# =============================================================

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=8))
async def serper_search(query: str) -> list[dict]:
    headers = {
        "X-API-KEY":    SERPER_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {"q": query, "num": 5, "hl": "en"}
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            "https://google.serper.dev/search",
            headers=headers, json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
    return [
        {"title": i.get("title",""), "snippet": i.get("snippet",""), "link": i.get("link","")}
        for i in data.get("organic", [])
    ]


async def search_team_stats(team: str, league: str) -> list[dict]:
    return await serper_search(f"{team} {league} stats last 15 games net rating pace eFG% 2025")

async def search_h2h(team_home: str, team_away: str, league: str) -> list[dict]:
    return await serper_search(f"{team_home} vs {team_away} {league} head to head 2024 2025")

async def search_injuries(team: str, league: str) -> list[dict]:
    return await serper_search(f"{team} injury report today 2025 player status out doubtful")

async def search_matchup(team_home: str, team_away: str) -> list[dict]:
    return await serper_search(f"{team_home} vs {team_away} matchup analysis preview 2025")

async def search_recent_form(team: str, league: str) -> list[dict]:
    return await serper_search(f"{team} {league} last 10 games results score margin 2025")


# =============================================================
# 2. JINA
# =============================================================

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=8))
async def jina_fetch(url: str) -> str:
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(f"{JINA_BASE_URL}{url}", headers={"Accept": "text/plain"})
        resp.raise_for_status()
        return resp.text[:6000]


# =============================================================
# 3. GEMINI — 1 SEUL APPEL PAR PRÉDICTION
# =============================================================

def _call_gemini(prompt: str) -> str:
    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=GEMINI_MAX_TOKENS, temperature=0.2,
            ),
        )
        return response.text.strip()
    except Exception as e:
        return f"[Gemini error: {e}]"


def _default_stats() -> dict:
    return {
        "elo_rating": 1500.0, "net_rating": 0.0, "pace": 100.0,
        "efg_pct": 0.50, "tov_pct": 13.0, "orb_pct": 25.0,
        "ftr": 0.25, "ts_pct": 0.55, "ppp": 1.10,
        "rest_days": 2, "back_to_back": False,
    }


def gemini_full_analysis(
    team_home: str, team_away: str, league: str,
    home_raw: str, away_raw: str,
    inj_home_raw: str, inj_away_raw: str,
    form_home_raw: str, form_away_raw: str,
    h2h_raw: str, matchup_raw: str,
) -> dict:
    """Unique appel Gemini par prédiction — remplace les 7 anciens appels séparés."""
    prompt = f"""
Tu es BasketPredictAI, analyste quantitatif basketball expert.
Match : {team_home} (domicile) vs {team_away} (extérieur) | Ligue : {league}

Réponds UNIQUEMENT avec un objet JSON valide, sans texte avant ou après,
sans balises markdown, sans commentaires.

Structure attendue :
{{
  "home_stats": {{
    "elo_rating": 1500.0, "net_rating": 0.0, "pace": 100.0,
    "efg_pct": 0.50, "tov_pct": 13.0, "orb_pct": 25.0,
    "ftr": 0.25, "ts_pct": 0.55, "ppp": 1.10,
    "rest_days": 2, "back_to_back": false
  }},
  "away_stats": {{
    "elo_rating": 1500.0, "net_rating": 0.0, "pace": 100.0,
    "efg_pct": 0.50, "tov_pct": 13.0, "orb_pct": 25.0,
    "ftr": 0.25, "ts_pct": 0.55, "ppp": 1.10,
    "rest_days": 2, "back_to_back": false
  }},
  "injuries_home": [{{"player_name": "Nom", "status": "Out", "impact_pts": -3.5}}],
  "injuries_away": [],
  "form_home": [-5.0, 8.0, 3.0, -12.0, 15.0],
  "form_away": [2.0, -4.0, 7.0, 1.0, -9.0],
  "h2h_summary": "Résumé factuel en 2-3 phrases de l'historique H2H en français.",
  "key_factor": "Facteur clé décisif en 3-4 phrases, précis et factuel, en français."
}}

Règles :
- Utilise les valeurs par défaut si une donnée est introuvable.
- injuries : statuts autorisés = Out | Doubtful | Questionable
- form : différentiels de score (marqués - encaissés), du plus ancien au plus récent, max 10 valeurs.
- h2h_summary et key_factor : jamais inventer, basé uniquement sur les données fournies.

=== DONNÉES SOURCE ===

[Stats {team_home}]
{home_raw[:2500]}

[Stats {team_away}]
{away_raw[:2500]}

[Blessures {team_home}]
{inj_home_raw[:1500]}

[Blessures {team_away}]
{inj_away_raw[:1500]}

[Forme récente {team_home}]
{form_home_raw[:1500]}

[Forme récente {team_away}]
{form_away_raw[:1500]}

[H2H]
{h2h_raw[:1500]}

[Analyse matchup]
{matchup_raw[:1000]}
"""
    raw = _call_gemini(prompt).replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "home_stats":    _default_stats(),
            "away_stats":    _default_stats(),
            "injuries_home": [],
            "injuries_away": [],
            "form_home":     [],
            "form_away":     [],
            "h2h_summary":   "Données H2H indisponibles.",
            "key_factor":    "Analyse indisponible — données insuffisantes.",
        }


# =============================================================
# 4. FUSION : API réelles > Gemini fallback
# =============================================================

def _merge_stats(gemini_dict: dict, nba_dict: Optional[dict], tank01_dict: Optional[dict]) -> dict:
    merged = dict(gemini_dict)
    if tank01_dict:
        if "elo_rating" in tank01_dict:
            merged["elo_rating"] = tank01_dict["elo_rating"]
    if nba_dict:
        for field in ("net_rating", "pace", "efg_pct", "tov_pct",
                      "orb_pct", "ftr", "ts_pct", "ppp"):
            if field in nba_dict and nba_dict[field] is not None:
                merged[field] = nba_dict[field]
    return merged


def _merge_injuries(gemini_list: list[dict], tank01_list: list[dict]) -> list[dict]:
    return tank01_list if tank01_list else gemini_list


# =============================================================
# 5. ORCHESTRATEUR PRINCIPAL
# =============================================================

async def collect_match_data(
    team_home:  str,
    team_away:  str,
    league:     str,
    match_date: str | None = None,
) -> dict:

    is_nba = league.upper() == "NBA"

    # ── 1. Toutes les requêtes en parallèle ───────────────────
    tasks = [
        search_team_stats(team_home, league),
        search_team_stats(team_away, league),
        search_h2h(team_home, team_away, league),
        search_injuries(team_home, league),
        search_injuries(team_away, league),
        search_matchup(team_home, team_away),
        search_recent_form(team_home, league),
        search_recent_form(team_away, league),
    ]
    if is_nba:
        tasks += [
            get_nba_team_stats(team_home),
            get_nba_team_stats(team_away),
            get_nba_team_form(team_home),
            get_nba_team_form(team_away),
        ]
    tasks += [
        get_tank01_injuries(team_home),
        get_tank01_injuries(team_away),
        get_tank01_team_stats(team_home),
        get_tank01_team_stats(team_away),
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    def safe(val, default):
        return default if isinstance(val, Exception) or val is None else val

    # ── 2. Déballage des résultats ────────────────────────────
    home_serper      = safe(results[0], [])
    away_serper      = safe(results[1], [])
    h2h_serper       = safe(results[2], [])
    inj_home_serper  = safe(results[3], [])
    inj_away_serper  = safe(results[4], [])
    matchup_serper   = safe(results[5], [])
    form_home_serper = safe(results[6], [])
    form_away_serper = safe(results[7], [])

    if is_nba:
        nba_stats_home  = safe(results[8],  None)
        nba_stats_away  = safe(results[9],  None)
        nba_form_home   = safe(results[10], [])
        nba_form_away   = safe(results[11], [])
        tank01_inj_home = safe(results[12], [])
        tank01_inj_away = safe(results[13], [])
        tank01_home     = safe(results[14], None)
        tank01_away     = safe(results[15], None)
    else:
        nba_stats_home = nba_stats_away = None
        nba_form_home  = nba_form_away  = []
        tank01_inj_home = safe(results[8],  [])
        tank01_inj_away = safe(results[9],  [])
        tank01_home     = safe(results[10], None)
        tank01_away     = safe(results[11], None)

    # ── 3. Snippets Serper ────────────────────────────────────
    def snippets(r: list[dict]) -> str:
        return "\n".join(f"{x.get('title','')}: {x.get('snippet','')}" for x in r)

    home_raw      = snippets(home_serper)
    away_raw      = snippets(away_serper)
    h2h_raw       = snippets(h2h_serper)
    inj_home_raw  = snippets(inj_home_serper)
    inj_away_raw  = snippets(inj_away_serper)
    matchup_raw   = snippets(matchup_serper)
    form_home_raw = snippets(form_home_serper)
    form_away_raw = snippets(form_away_serper)

    # ── 4. Gemini — 1 seul appel ──────────────────────────────
    gemini = gemini_full_analysis(
        team_home=team_home,   team_away=team_away,   league=league,
        home_raw=home_raw,     away_raw=away_raw,
        inj_home_raw=inj_home_raw, inj_away_raw=inj_away_raw,
        form_home_raw=form_home_raw, form_away_raw=form_away_raw,
        h2h_raw=h2h_raw,       matchup_raw=matchup_raw,
    )

    gemini_home   = gemini.get("home_stats",    _default_stats())
    gemini_away   = gemini.get("away_stats",    _default_stats())
    gemini_inj_h  = gemini.get("injuries_home", [])
    gemini_inj_a  = gemini.get("injuries_away", [])
    gemini_form_h = gemini.get("form_home",     [])
    gemini_form_a = gemini.get("form_away",     [])
    h2h_summary   = gemini.get("h2h_summary",   "Données H2H indisponibles.")
    key_factor    = gemini.get("key_factor",    "Analyse indisponible.")

    # ── 5. Fusion API réelles > Gemini ────────────────────────
    final_home_dict = _merge_stats(gemini_home, nba_stats_home, tank01_home)
    final_away_dict = _merge_stats(gemini_away, nba_stats_away, tank01_away)
    final_form_home = nba_form_home if nba_form_home else gemini_form_h
    final_form_away = nba_form_away if nba_form_away else gemini_form_a
    final_inj_home  = _merge_injuries(gemini_inj_h, tank01_inj_home)
    final_inj_away  = _merge_injuries(gemini_inj_a, tank01_inj_away)

    # ── 6. Construction objets ────────────────────────────────
    home_stats = TeamStats(team_name=team_home, recent_form=final_form_home, **final_home_dict)
    away_stats = TeamStats(team_name=team_away, recent_form=final_form_away, **final_away_dict)
    inj_home_list = [InjuryReport(**i) for i in final_inj_home]
    inj_away_list = [InjuryReport(**i) for i in final_inj_away]

    # ── 7. Ajustement blessures sur net_rating ────────────────
    home_stats.net_rating += sum(i.impact_pts for i in inj_home_list)
    away_stats.net_rating += sum(i.impact_pts for i in inj_away_list)

    # ── 8. Sources de données ─────────────────────────────────
    sources = ["Serper.dev", "Google Gemini 3 Flash"]  # ✏️ mis à jour
    if nba_stats_home or nba_stats_away:
        sources.append("NBA Stats (stats.nba.com)")
    if tank01_inj_home or tank01_inj_away:
        sources.append("Tank01 / RapidAPI")

    return {
        "home_stats":    home_stats,
        "away_stats":    away_stats,
        "injuries_home": inj_home_list,
        "injuries_away": inj_away_list,
        "h2h_summary":   h2h_summary,
        "key_factor":    key_factor,
        "data_sources":  sources,
    }