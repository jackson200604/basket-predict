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
# 3. GEMINI
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


def gemini_extract_stats(raw_text: str, team_name: str) -> dict:
    prompt = f"""
Tu es un analyste basketball expert. Extrais les statistiques
pour l'équipe "{team_name}" depuis le texte ci-dessous.

Réponds UNIQUEMENT avec un objet JSON valide, sans texte avant ou après.
Si une valeur est introuvable, utilise les valeurs par défaut.

Format JSON :
{{
  "elo_rating":   1500.0,
  "net_rating":   0.0,
  "pace":         100.0,
  "efg_pct":      0.50,
  "tov_pct":      13.0,
  "orb_pct":      25.0,
  "ftr":          0.25,
  "ts_pct":       0.55,
  "ppp":          1.10,
  "rest_days":    2,
  "back_to_back": false
}}

Texte source :
{raw_text[:4000]}
"""
    raw = _call_gemini(prompt).replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "elo_rating": 1500.0, "net_rating": 0.0, "pace": 100.0,
            "efg_pct": 0.50, "tov_pct": 13.0, "orb_pct": 25.0,
            "ftr": 0.25, "ts_pct": 0.55, "ppp": 1.10,
            "rest_days": 2, "back_to_back": False,
        }


def gemini_extract_recent_form(raw_text: str, team_name: str) -> list[float]:
    prompt = f"""
Extrais les différentiels de score (points marqués - points encaissés)
des derniers matchs de "{team_name}". Réponds UNIQUEMENT avec un tableau
JSON de nombres, du plus ancien au plus récent. Max 10 valeurs.
Si aucune donnée disponible, retourne [].
Exemple : [-5.0, 8.0, 3.0, -12.0, 15.0]

Texte : {raw_text[:3000]}
"""
    raw = _call_gemini(prompt).replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(raw)
        return [float(v) for v in data[:10]] if isinstance(data, list) else []
    except (json.JSONDecodeError, ValueError):
        return []


def gemini_extract_injuries(raw_text: str, team_name: str) -> list[dict]:
    prompt = f"""
Extrais la liste des blessés pour "{team_name}".
Réponds UNIQUEMENT avec un tableau JSON, sans texte avant ou après.
Si aucun blessé, retourne [].
Format : [{{"player_name": "Nom", "status": "Out", "impact_pts": -3.5}}]
Statuts : "Out", "Doubtful", "Questionable"
Texte : {raw_text[:3000]}
"""
    raw = _call_gemini(prompt).replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def gemini_key_factor(
    team_home: str, team_away: str,
    home_stats: dict, away_stats: dict,
    h2h_text: str, matchup_text: str, league: str,
) -> str:
    prompt = f"""
Tu es BasketPredictAI, analyste quantitatif expert.
Match : {team_home} vs {team_away} | Ligue : {league}
Stats domicile : {json.dumps(home_stats)}
Stats extérieur : {json.dumps(away_stats)}
H2H : {h2h_text[:1000]}
Matchup : {matchup_text[:1000]}
Rédige en 3-4 phrases le FACTEUR CLÉ décisif. Sois précis, factuel, en français.
"""
    return _call_gemini(prompt)


def gemini_h2h_summary(h2h_results: list[dict], team_home: str, team_away: str) -> str:
    snippets = "\n".join(
        f"- {r.get('title','')}: {r.get('snippet','')}" for r in h2h_results[:5]
    )
    prompt = f"""
Résume en 2-3 phrases l'historique entre {team_home} et {team_away}.
Sois factuel, ne jamais inventer. Langue : français.
Données : {snippets}
"""
    return _call_gemini(prompt)


# =============================================================
# 4. FUSION : API réelles > Gemini fallback
# =============================================================

def _merge_stats(gemini_dict: dict, nba_dict: Optional[dict], tank01_dict: Optional[dict]) -> dict:
    """
    Priorité : NBA Stats (officiel) > Tank01 > Gemini.
    On ne remplace que les champs effectivement présents dans la source
    prioritaire, pour ne jamais écraser un champ avec None.
    """
    merged = dict(gemini_dict)  # base Gemini

    # Tank01 : uniquement elo_rating (win-based)
    if tank01_dict:
        if "elo_rating" in tank01_dict:
            merged["elo_rating"] = tank01_dict["elo_rating"]

    # NBA Stats : remplace tous les champs disponibles (données officielles)
    if nba_dict:
        for field in ("net_rating", "pace", "efg_pct", "tov_pct",
                      "orb_pct", "ftr", "ts_pct", "ppp"):
            if field in nba_dict and nba_dict[field] is not None:
                merged[field] = nba_dict[field]

    return merged


def _merge_injuries(gemini_list: list[dict], tank01_list: list[dict]) -> list[dict]:
    """
    Tank01 est prioritaire (données structurées).
    Si Tank01 retourne des blessés, on l'utilise exclusivement.
    Sinon, on garde la liste Gemini.
    """
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

    # Appels API réelles (NBA uniquement pour NBA Stats, tous pour Tank01)
    if is_nba:
        tasks += [
            get_nba_team_stats(team_home),   # 8
            get_nba_team_stats(team_away),   # 9
            get_nba_team_form(team_home),    # 10
            get_nba_team_form(team_away),    # 11
        ]
    tasks += [
        get_tank01_injuries(team_home),      # 12 (ou 8)
        get_tank01_injuries(team_away),      # 13 (ou 9)
        get_tank01_team_stats(team_home),    # 14 (ou 10)
        get_tank01_team_stats(team_away),    # 15 (ou 11)
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
        nba_stats_home   = safe(results[8],  None)
        nba_stats_away   = safe(results[9],  None)
        nba_form_home    = safe(results[10], [])
        nba_form_away    = safe(results[11], [])
        tank01_inj_home  = safe(results[12], [])
        tank01_inj_away  = safe(results[13], [])
        tank01_home      = safe(results[14], None)
        tank01_away      = safe(results[15], None)
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

    # ── 4. Extraction Gemini (fallback) ───────────────────────
    gemini_home   = gemini_extract_stats(home_raw, team_home)
    gemini_away   = gemini_extract_stats(away_raw, team_away)
    gemini_inj_h  = gemini_extract_injuries(inj_home_raw, team_home)
    gemini_inj_a  = gemini_extract_injuries(inj_away_raw, team_away)
    h2h_summary   = gemini_h2h_summary(h2h_serper, team_home, team_away)
    key_factor    = gemini_key_factor(
        team_home, team_away, gemini_home, gemini_away,
        h2h_raw, matchup_raw, league,
    )
    gemini_form_h = gemini_extract_recent_form(form_home_raw, team_home)
    gemini_form_a = gemini_extract_recent_form(form_away_raw, team_away)

    # ── 5. Fusion API réelles > Gemini ────────────────────────
    final_home_dict = _merge_stats(gemini_home, nba_stats_home, tank01_home)
    final_away_dict = _merge_stats(gemini_away, nba_stats_away, tank01_away)

    # recent_form : NBA Stats > Gemini
    final_form_home = nba_form_home if nba_form_home else gemini_form_h
    final_form_away = nba_form_away if nba_form_away else gemini_form_a

    # blessures : Tank01 > Gemini
    final_inj_home = _merge_injuries(gemini_inj_h, tank01_inj_home)
    final_inj_away = _merge_injuries(gemini_inj_a, tank01_inj_away)

    # ── 6. Construction objets ────────────────────────────────
    home_stats = TeamStats(
        team_name=team_home, recent_form=final_form_home, **final_home_dict,
    )
    away_stats = TeamStats(
        team_name=team_away, recent_form=final_form_away, **final_away_dict,
    )
    inj_home_list = [InjuryReport(**i) for i in final_inj_home]
    inj_away_list = [InjuryReport(**i) for i in final_inj_away]

    # ── 7. Ajustement blessures sur net_rating ────────────────
    home_stats.net_rating += sum(i.impact_pts for i in inj_home_list)
    away_stats.net_rating += sum(i.impact_pts for i in inj_away_list)

    # ── 8. Sources de données ─────────────────────────────────
    sources = ["Serper.dev", "Google Gemini Flash"]
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
