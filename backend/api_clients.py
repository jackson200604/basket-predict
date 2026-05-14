# ============================================================
# api_clients.py — NBA Stats · Tank01 · The Odds API
# ============================================================
"""
Trois clients indépendants, chacun avec son propre fallback.
Toutes les fonctions retournent None (ou dict vide / []) en cas
d'échec, ce qui permet à data_ingestion.py de basculer sur les
valeurs Gemini sans planter.

──────────────────────────────────────────────────────────────
NBA Stats  (stats.nba.com)  — GRATUIT, sans clé
  • get_nba_team_stats()     → dict de stats agrégées saison
  • get_nba_team_form()      → list[float] différentiels récents
  • get_nba_team_id()        → int team_id NBA interne

Tank01 / RapidAPI           — clé TANK01_API_KEY
  • get_tank01_injuries()    → list[dict] blessures structurées
  • get_tank01_team_stats()  → dict stats complémentaires

The Odds API                — clé ODDS_API_KEY
  • get_odds()               → dict {odds_home, odds_away} américains
──────────────────────────────────────────────────────────────
"""
import httpx
import asyncio
import logging
from typing import Optional
from config import (
    NBA_STATS_BASE, NBA_STATS_HEADERS,
    NBA_CURRENT_SEASON, NBA_SEASON_TYPE, NBA_FORM_LAST_N,
    TANK01_API_KEY, TANK01_BASE, TANK01_HOST,
    ODDS_API_KEY, ODDS_API_BASE, ODDS_SPORT,
    ODDS_REGIONS, ODDS_MARKETS, ODDS_BOOKMAKERS,
)

logger = logging.getLogger(__name__)


# =============================================================
# UTILITAIRES
# =============================================================

def _parse_nba_response(data: dict, table_index: int = 0) -> list[dict]:
    """
    Convertit la réponse stats.nba.com (format resultSets)
    en liste de dicts {colonne: valeur}.
    """
    try:
        result_set = data["resultSets"][table_index]
        headers    = result_set["headers"]
        rows       = result_set["rowSet"]
        return [dict(zip(headers, row)) for row in rows]
    except (KeyError, IndexError):
        return []


# Mapping nom d'équipe → abbreviation NBA officielle
# Utilisé pour matcher les résultats stats.nba.com
NBA_TEAM_ABBR: dict[str, str] = {
    "atlanta hawks":            "ATL", "boston celtics":          "BOS",
    "brooklyn nets":            "BKN", "charlotte hornets":        "CHA",
    "chicago bulls":            "CHI", "cleveland cavaliers":      "CLE",
    "dallas mavericks":         "DAL", "denver nuggets":           "DEN",
    "detroit pistons":          "DET", "golden state warriors":    "GSW",
    "houston rockets":          "HOU", "indiana pacers":           "IND",
    "los angeles clippers":     "LAC", "los angeles lakers":       "LAL",
    "memphis grizzlies":        "MEM", "miami heat":               "MIA",
    "milwaukee bucks":          "MIL", "minnesota timberwolves":   "MIN",
    "new orleans pelicans":     "NOP", "new york knicks":          "NYK",
    "oklahoma city thunder":    "OKC", "orlando magic":            "ORL",
    "philadelphia 76ers":       "PHI", "phoenix suns":             "PHX",
    "portland trail blazers":   "POR", "sacramento kings":         "SAC",
    "san antonio spurs":        "SAS", "toronto raptors":          "TOR",
    "utah jazz":                "UTA", "washington wizards":        "WAS",
    # Alias courants
    "lakers":    "LAL", "celtics":   "BOS", "warriors":  "GSW",
    "bulls":     "CHI", "heat":      "MIA", "bucks":     "MIL",
    "nets":      "BKN", "knicks":    "NYK", "suns":      "PHX",
    "nuggets":   "DEN", "clippers":  "LAC", "raptors":   "TOR",
    "mavericks": "DAL", "grizzlies": "MEM", "pelicans":  "NOP",
    "thunder":   "OKC", "timberwolves": "MIN", "pacers":  "IND",
    "cavaliers": "CLE", "pistons":   "DET", "rockets":   "HOU",
    "hawks":     "ATL", "hornets":   "CHA", "magic":     "ORL",
    "76ers":     "PHI", "sixers":    "PHI", "blazers":   "POR",
    "trail blazers": "POR", "kings":  "SAC", "spurs":    "SAS",
    "jazz":      "UTA", "wizards":   "WAS",
}


def _team_abbr(team_name: str) -> Optional[str]:
    return NBA_TEAM_ABBR.get(team_name.lower().strip())


# =============================================================
# 1. NBA STATS — stats.nba.com  (gratuit, sans clé)
# =============================================================

async def get_nba_team_id(team_name: str) -> Optional[int]:
    """Retourne l'ID interne NBA d'une équipe (ex: LAL → 1610612747)."""
    url    = f"{NBA_STATS_BASE}/commonteamroster"
    abbr   = _team_abbr(team_name)
    if not abbr:
        return None
    params = {
        "LeagueID": "00",
        "Season":   NBA_CURRENT_SEASON,
    }
    try:
        async with httpx.AsyncClient(
            headers=NBA_STATS_HEADERS, timeout=20, follow_redirects=True
        ) as client:
            resp = await client.get(
                f"{NBA_STATS_BASE}/commonallplayers",
                params={"LeagueID": "00", "Season": NBA_CURRENT_SEASON,
                        "IsOnlyCurrentSeason": "1"},
            )
            resp.raise_for_status()
            # On utilise leaguedashteamstats pour récupérer l'ID
            resp2 = await client.get(
                f"{NBA_STATS_BASE}/leaguedashteamstats",
                params={
                    "Conference":       "",
                    "DateFrom":         "",
                    "DateTo":           "",
                    "Division":         "",
                    "GameScope":        "",
                    "GameSegment":      "",
                    "LastNGames":       0,
                    "LeagueID":         "00",
                    "Location":         "",
                    "MeasureType":      "Base",
                    "Month":            0,
                    "OpponentTeamID":   0,
                    "Outcome":          "",
                    "PORound":          0,
                    "PaceAdjust":       "N",
                    "PerMode":          "PerGame",
                    "Period":           0,
                    "PlayerExperience": "",
                    "PlayerPosition":   "",
                    "PlusMinus":        "N",
                    "Rank":             "N",
                    "Season":           NBA_CURRENT_SEASON,
                    "SeasonSegment":    "",
                    "SeasonType":       NBA_SEASON_TYPE,
                    "ShotClockRange":   "",
                    "StarterBench":     "",
                    "TeamID":           0,
                    "TwoWay":           0,
                    "VsConference":     "",
                    "VsDivision":       "",
                },
            )
            resp2.raise_for_status()
            rows = _parse_nba_response(resp2.json())
            for row in rows:
                if row.get("TEAM_ABBREVIATION", "").upper() == abbr.upper():
                    return int(row["TEAM_ID"])
    except Exception as e:
        logger.warning(f"[NBA] get_nba_team_id({team_name}): {e}")
    return None


async def get_nba_team_stats(team_name: str) -> Optional[dict]:
    """
    Récupère les stats de saison depuis leaguedashteamstats.
    Retourne un dict compatible avec les champs de TeamStats.
    """
    abbr = _team_abbr(team_name)
    if not abbr:
        logger.warning(f"[NBA] Abréviation inconnue pour '{team_name}'")
        return None

    # ── Base stats (NET RTG, pace, etc.) ──────────────────────
    base_params = {
        "Conference": "", "DateFrom": "", "DateTo": "",
        "Division": "", "GameScope": "", "GameSegment": "",
        "LastNGames": 0, "LeagueID": "00", "Location": "",
        "MeasureType": "Base", "Month": 0, "OpponentTeamID": 0,
        "Outcome": "", "PORound": 0, "PaceAdjust": "N",
        "PerMode": "PerGame", "Period": 0,
        "PlayerExperience": "", "PlayerPosition": "",
        "PlusMinus": "N", "Rank": "N",
        "Season": NBA_CURRENT_SEASON, "SeasonSegment": "",
        "SeasonType": NBA_SEASON_TYPE, "ShotClockRange": "",
        "StarterBench": "", "TeamID": 0, "TwoWay": 0,
        "VsConference": "", "VsDivision": "",
    }
    adv_params  = {**base_params, "MeasureType": "Advanced"}
    ff_params   = {**base_params, "MeasureType": "Four Factors"}

    try:
        async with httpx.AsyncClient(
            headers=NBA_STATS_HEADERS, timeout=25, follow_redirects=True
        ) as client:
            r_base, r_adv, r_ff = await asyncio.gather(
                client.get(f"{NBA_STATS_BASE}/leaguedashteamstats", params=base_params),
                client.get(f"{NBA_STATS_BASE}/leaguedashteamstats", params=adv_params),
                client.get(f"{NBA_STATS_BASE}/leaguedashteamstats", params=ff_params),
            )

        rows_base = _parse_nba_response(r_base.json())
        rows_adv  = _parse_nba_response(r_adv.json())
        rows_ff   = _parse_nba_response(r_ff.json())

        def find(rows: list[dict]) -> Optional[dict]:
            for r in rows:
                if r.get("TEAM_ABBREVIATION", "").upper() == abbr.upper():
                    return r
            return None

        b = find(rows_base)
        a = find(rows_adv)
        f = find(rows_ff)

        if not b:
            logger.warning(f"[NBA] Équipe '{abbr}' introuvable dans leaguedashteamstats")
            return None

        # Points par possession ≈ OffRating / 100
        off_rating = float(a.get("OFF_RATING", 110.0)) if a else 110.0
        ppp        = round(off_rating / 100.0, 4)

        return {
            # Champs TeamStats
            "net_rating":   round(float(a.get("NET_RATING",  0.0))  if a else 0.0,  2),
            "pace":         round(float(a.get("PACE",        100.0)) if a else 100.0, 2),
            "efg_pct":      round(float(f.get("EFG_PCT",     0.50))  if f else 0.50, 4),
            "tov_pct":      round(float(f.get("TM_TOV_PCT",  13.0))  if f else 13.0, 2),
            "orb_pct":      round(float(f.get("OREB_PCT",    25.0))  if f else 25.0, 2),
            "ftr":          round(float(f.get("FTA_RATE",    0.25))  if f else 0.25, 4),
            "ts_pct":       round(float(b.get("TS_PCT",      0.55))  if b else 0.55, 4),
            "ppp":          ppp,
            # Méta utile
            "_team_id":     int(b.get("TEAM_ID", 0)),
            "_off_rating":  off_rating,
            "_def_rating":  round(float(a.get("DEF_RATING", 110.0)) if a else 110.0, 2),
            "_wins":        int(b.get("W", 0)),
            "_losses":      int(b.get("L", 0)),
        }

    except Exception as e:
        logger.warning(f"[NBA] get_nba_team_stats({team_name}): {e}")
        return None


async def get_nba_team_form(team_name: str, last_n: int = NBA_FORM_LAST_N) -> list[float]:
    """
    Retourne les différentiels de score des last_n derniers matchs
    (positif = victoire, négatif = défaite) via teamgamelog.
    Ordre chronologique (du plus ancien au plus récent).
    """
    # On a besoin du team_id
    team_id = await get_nba_team_id(team_name)
    if not team_id:
        return []

    params = {
        "TeamID":     team_id,
        "Season":     NBA_CURRENT_SEASON,
        "SeasonType": NBA_SEASON_TYPE,
        "LeagueID":   "00",
    }
    try:
        async with httpx.AsyncClient(
            headers=NBA_STATS_HEADERS, timeout=20, follow_redirects=True
        ) as client:
            resp = await client.get(
                f"{NBA_STATS_BASE}/teamgamelog", params=params
            )
            resp.raise_for_status()
            rows = _parse_nba_response(resp.json())

        # rows[0] = match le plus récent → on prend les last_n et on inverse
        diffs = []
        for row in rows[:last_n]:
            pts     = row.get("PTS", 0)
            pts_opp = row.get("PTS_OPP", None)
            if pts_opp is None:
                # Calculer depuis MATCHUP et WL si PTS_OPP absent
                plus_minus = row.get("PLUS_MINUS", 0)
                diff = float(plus_minus)
            else:
                diff = float(pts) - float(pts_opp)
            diffs.append(round(diff, 1))

        diffs.reverse()   # ordre chronologique
        return diffs

    except Exception as e:
        logger.warning(f"[NBA] get_nba_team_form({team_name}): {e}")
        return []


# =============================================================
# 2. TANK01 / RAPIDAPI — Blessures & stats complémentaires
# =============================================================

async def get_tank01_injuries(team_name: str) -> list[dict]:
    """
    Retourne la liste des blessés pour une équipe via Tank01.
    Format retourné : [{"player_name": str, "status": str, "impact_pts": float}]
    """
    if not TANK01_API_KEY:
        return []

    abbr = _team_abbr(team_name)
    if not abbr:
        return []

    headers = {
        "x-rapidapi-key":  TANK01_API_KEY,
        "x-rapidapi-host": TANK01_HOST,
    }
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{TANK01_BASE}/getNBAInjuryList",
                headers=headers,
            )
            resp.raise_for_status()
            data    = resp.json()
            players = data.get("body", [])

        injuries = []
        for p in players:
            # Filtrer par équipe
            if p.get("teamAbv", "").upper() != abbr.upper():
                continue

            status_raw = p.get("injuryStatus", "").strip()
            # Normaliser le statut
            if "out" in status_raw.lower():
                status = "Out"
            elif "doubtful" in status_raw.lower():
                status = "Doubtful"
            elif "questionable" in status_raw.lower():
                status = "Questionable"
            else:
                continue   # "probable" ou vide → on ignore

            # Impact estimé selon statut et position
            pos = p.get("pos", "").upper()
            base_impact = {
                "Out":          -4.0,
                "Doubtful":     -2.5,
                "Questionable": -1.0,
            }.get(status, 0.0)

            # Stars (starters) pèsent plus
            if pos in ("PG", "SG", "SF"):
                base_impact *= 1.3
            elif pos in ("PF", "C"):
                base_impact *= 1.1

            injuries.append({
                "player_name": p.get("playerName", "Unknown"),
                "status":      status,
                "impact_pts":  round(base_impact, 1),
            })

        return injuries

    except Exception as e:
        logger.warning(f"[Tank01] get_tank01_injuries({team_name}): {e}")
        return []


async def get_tank01_team_stats(team_name: str) -> Optional[dict]:
    """
    Récupère les stats d'équipe via Tank01 (complément NBA Stats).
    Utilisé principalement pour les ligues hors-NBA (EuroLeague, etc.)
    ou comme source de recoupement pour elo_rating.
    """
    if not TANK01_API_KEY:
        return None

    abbr = _team_abbr(team_name)
    if not abbr:
        return None

    headers = {
        "x-rapidapi-key":  TANK01_API_KEY,
        "x-rapidapi-host": TANK01_HOST,
    }
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{TANK01_BASE}/getNBATeamList",
                headers=headers,
            )
            resp.raise_for_status()
            teams = resp.json().get("body", [])

        for t in teams:
            if t.get("teamAbv", "").upper() == abbr.upper():
                wins   = int(t.get("wins",   0))
                losses = int(t.get("losses", 0))
                games  = wins + losses
                # Elo synthétique basé sur win% si pas d'autre source
                win_pct    = wins / max(games, 1)
                elo_approx = round(1500.0 + (win_pct - 0.50) * 400.0, 1)
                return {
                    "elo_rating": elo_approx,
                    "_wins":      wins,
                    "_losses":    losses,
                    "_win_pct":   round(win_pct, 4),
                }
    except Exception as e:
        logger.warning(f"[Tank01] get_tank01_team_stats({team_name}): {e}")
    return None


# =============================================================
# 3. THE ODDS API — Cotes moneyline en temps réel
# =============================================================

def _american_to_decimal(american: float) -> float:
    if american > 0:
        return round(american / 100.0 + 1.0, 4)
    return round(100.0 / abs(american) + 1.0, 4)


async def get_odds(team_home: str, team_away: str) -> Optional[dict]:
    """
    Retourne les meilleures cotes moneyline américaines pour le match.
    Stratégie : consensus sur ODDS_BOOKMAKERS, fallback sur le premier
    bookmaker disponible.

    Retourne :
      {
        "odds_home": float,   # ex: -150  (négatif = favori)
        "odds_away": float,   # ex: +130
        "bookmaker": str,
        "game_id":   str,
      }
    ou None si introuvable.
    """
    if not ODDS_API_KEY:
        return None

    params = {
        "apiKey":  ODDS_API_KEY,
        "regions": ODDS_REGIONS,
        "markets": ODDS_MARKETS,
        "oddsFormat": "american",
        "dateFormat":  "iso",
    }
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{ODDS_API_BASE}/sports/{ODDS_SPORT}/odds",
                params=params,
            )
            resp.raise_for_status()
            games = resp.json()

        home_key = team_home.lower().strip()
        away_key = team_away.lower().strip()

        def _name_match(api_name: str, query: str) -> bool:
            """Correspondance souple : 'Lakers' ↔ 'Los Angeles Lakers'."""
            api_lower = api_name.lower()
            return (
                query in api_lower
                or api_lower in query
                or any(w in api_lower for w in query.split() if len(w) > 3)
            )

        for game in games:
            h = game.get("home_team", "")
            a = game.get("away_team", "")
            if not (_name_match(h, home_key) and _name_match(a, away_key)):
                continue

            # Chercher d'abord dans nos bookmakers préférés
            bookmakers = game.get("bookmakers", [])
            selected   = None
            for bk_name in ODDS_BOOKMAKERS:
                for bk in bookmakers:
                    if bk.get("key", "") == bk_name:
                        selected = bk
                        break
                if selected:
                    break
            if not selected and bookmakers:
                selected = bookmakers[0]
            if not selected:
                continue

            # Extraire h2h markets
            for market in selected.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                outcomes   = market.get("outcomes", [])
                odds_home  = None
                odds_away  = None
                for o in outcomes:
                    if _name_match(o.get("name", ""), home_key):
                        odds_home = float(o["price"])
                    elif _name_match(o.get("name", ""), away_key):
                        odds_away = float(o["price"])

                if odds_home is not None and odds_away is not None:
                    return {
                        "odds_home": odds_home,
                        "odds_away": odds_away,
                        "bookmaker": selected.get("title", selected.get("key", "?")),
                        "game_id":   game.get("id", ""),
                    }

        logger.info(f"[Odds] Aucune cote trouvée pour {team_home} vs {team_away}")
        return None

    except Exception as e:
        logger.warning(f"[Odds] get_odds({team_home}, {team_away}): {e}")
        return None
