# ============================================================
# data_ingestion.py — Serper + Jina + Gemini
# ============================================================
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

# ── Initialisation Gemini ─────────────────────────────────────
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(GEMINI_MODEL)


# =============================================================
# 1. SERPER — Recherche web
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
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
    return [
        {
            "title":   item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "link":    item.get("link", ""),
        }
        for item in data.get("organic", [])
    ]


async def search_team_stats(team: str, league: str) -> list[dict]:
    query = f"{team} {league} stats last 15 games net rating pace eFG% 2025"
    return await serper_search(query)


async def search_h2h(team_home: str, team_away: str, league: str) -> list[dict]:
    query = f"{team_home} vs {team_away} {league} head to head 2024 2025"
    return await serper_search(query)


async def search_injuries(team: str, league: str) -> list[dict]:
    query = f"{team} injury report today 2025 player status out doubtful"
    return await serper_search(query)


async def search_matchup(team_home: str, team_away: str) -> list[dict]:
    query = f"{team_home} vs {team_away} matchup analysis preview 2025"
    return await serper_search(query)


# =============================================================
# 2. JINA AI READER — Scraping
# =============================================================

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=8))
async def jina_fetch(url: str) -> str:
    jina_url = f"{JINA_BASE_URL}{url}"
    headers  = {"Accept": "text/plain"}
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(jina_url, headers=headers)
        resp.raise_for_status()
        return resp.text[:6000]


# =============================================================
# 3. GEMINI — Extraction structurée
# =============================================================

def _call_gemini(prompt: str) -> str:
    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=GEMINI_MAX_TOKENS,
                temperature=0.2,
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
    raw = _call_gemini(prompt)
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "elo_rating": 1500.0, "net_rating": 0.0,
            "pace": 100.0, "efg_pct": 0.50,
            "tov_pct": 13.0, "orb_pct": 25.0,
            "ftr": 0.25, "ts_pct": 0.55,
            "ppp": 1.10, "rest_days": 2,
            "back_to_back": False,
        }


def gemini_extract_injuries(raw_text: str, team_name: str) -> list[dict]:
    prompt = f"""
Extrais la liste des blessés pour "{team_name}".
Réponds UNIQUEMENT avec un tableau JSON, sans texte avant ou après.
Si aucun blessé, retourne [].

Format :
[
  {{"player_name": "Nom", "status": "Out", "impact_pts": -3.5}}
]

Statuts : "Out", "Doubtful", "Questionable"

Texte :
{raw_text[:3000]}
"""
    raw = _call_gemini(prompt)
    raw = raw.replace("```json", "").replace("```", "").strip()
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

Rédige en 3-4 phrases le FACTEUR CLÉ décisif.
Sois précis, factuel, en français.
"""
    return _call_gemini(prompt)


def gemini_h2h_summary(h2h_results: list[dict], team_home: str, team_away: str) -> str:
    snippets = "\n".join(
        f"- {r.get('title','')}: {r.get('snippet','')}"
        for r in h2h_results[:5]
    )
    prompt = f"""
Résume en 2-3 phrases l'historique entre {team_home} et {team_away}.
Sois factuel, ne jamais inventer. Langue : français.

Données :
{snippets}
"""
    return _call_gemini(prompt)


# =============================================================
# 4. ORCHESTRATEUR PRINCIPAL
# =============================================================

async def collect_match_data(
    team_home:  str,
    team_away:  str,
    league:     str,
    match_date: str | None = None,
) -> dict:

    # ── 1. Recherches parallèles ──────────────────────────────
    (
        home_serper, away_serper, h2h_serper,
        inj_home_serper, inj_away_serper, matchup_serper,
    ) = await asyncio.gather(
        search_team_stats(team_home, league),
        search_team_stats(team_away, league),
        search_h2h(team_home, team_away, league),
        search_injuries(team_home, league),
        search_injuries(team_away, league),
        search_matchup(team_home, team_away),
    )

    # ── 2. Concaténation snippets ─────────────────────────────
    def snippets(results: list[dict]) -> str:
        return "\n".join(
            f"{r.get('title','')}: {r.get('snippet','')}"
            for r in results
        )

    home_raw     = snippets(home_serper)
    away_raw     = snippets(away_serper)
    h2h_raw      = snippets(h2h_serper)
    inj_home_raw = snippets(inj_home_serper)
    inj_away_raw = snippets(inj_away_serper)
    matchup_raw  = snippets(matchup_serper)

    # ── 3. Extraction Gemini ──────────────────────────────────
    home_stats_dict = gemini_extract_stats(home_raw, team_home)
    away_stats_dict = gemini_extract_stats(away_raw, team_away)
    injuries_home   = gemini_extract_injuries(inj_home_raw, team_home)
    injuries_away   = gemini_extract_injuries(inj_away_raw, team_away)
    h2h_summary     = gemini_h2h_summary(h2h_serper, team_home, team_away)
    key_factor      = gemini_key_factor(
        team_home, team_away,
        home_stats_dict, away_stats_dict,
        h2h_raw, matchup_raw, league,
    )

    # ── 4. Construction objets ────────────────────────────────
    home_stats    = TeamStats(team_name=team_home, **home_stats_dict)
    away_stats    = TeamStats(team_name=team_away, **away_stats_dict)
    inj_home_list = [InjuryReport(**i) for i in injuries_home]
    inj_away_list = [InjuryReport(**i) for i in injuries_away]

    # ── 5. Ajustement Net Rating blessures ────────────────────
    home_stats.net_rating += sum(i.impact_pts for i in inj_home_list)
    away_stats.net_rating += sum(i.impact_pts for i in inj_away_list)

    return {
        "home_stats":    home_stats,
        "away_stats":    away_stats,
        "injuries_home": inj_home_list,
        "injuries_away": inj_away_list,
        "h2h_summary":   h2h_summary,
        "key_factor":    key_factor,
        "data_sources":  [
            "Serper.dev",
            "Jina AI Reader",
            "Google Gemini 3 Flash",
        ],
            }
