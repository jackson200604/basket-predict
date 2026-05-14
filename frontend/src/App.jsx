import { useState } from "react";

const API_URL = "https://basket-predict.onrender.com/predict";

const TEAMS = [
  "Los Angeles Lakers", "Boston Celtics", "Golden State Warriors",
  "Chicago Bulls", "Miami Heat", "Brooklyn Nets",
  "Milwaukee Bucks", "Phoenix Suns", "Denver Nuggets",
  "Philadelphia 76ers", "Dallas Mavericks", "Memphis Grizzlies",
  "New Orleans Pelicans", "Atlanta Hawks", "Cleveland Cavaliers",
  "Toronto Raptors", "Sacramento Kings", "Oklahoma City Thunder",
  "Minnesota Timberwolves", "New York Knicks",
];

function StatBar({ label, home, away, higherIsBetter = true }) {
  const total = (home || 0) + (away || 0);
  const homeW = total > 0 ? (home / total) * 100 : 50;
  const awayW = total > 0 ? (away / total) * 100 : 50;
  const homeWins = higherIsBetter ? home >= away : home <= away;
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#aaa", marginBottom: 4 }}>
        <span style={{ color: homeWins ? "#f97316" : "#94a3b8", fontWeight: homeWins ? 700 : 400 }}>{(home ?? "—").toString().slice(0, 5)}</span>
        <span style={{ color: "#64748b", textTransform: "uppercase", letterSpacing: 1, fontSize: 10 }}>{label}</span>
        <span style={{ color: !homeWins ? "#3b82f6" : "#94a3b8", fontWeight: !homeWins ? 700 : 400 }}>{(away ?? "—").toString().slice(0, 5)}</span>
      </div>
      <div style={{ display: "flex", height: 4, borderRadius: 2, overflow: "hidden", background: "#1e293b" }}>
        <div style={{ width: `${homeW}%`, background: "#f97316", transition: "width 0.8s ease" }} />
        <div style={{ width: `${awayW}%`, background: "#3b82f6", transition: "width 0.8s ease" }} />
      </div>
    </div>
  );
}

function TeamInput({ label, value, onChange, color }) {
  const [open, setOpen] = useState(false);
  const filtered = TEAMS.filter(t => t.toLowerCase().includes(value.toLowerCase()));
  return (
    <div style={{ flex: 1, position: "relative" }}>
      <div style={{ fontSize: 10, letterSpacing: 2, color, textTransform: "uppercase", marginBottom: 6, fontWeight: 700 }}>{label}</div>
      <input
        value={value}
        onChange={e => { onChange(e.target.value); setOpen(true); }}
        onFocus={() => setOpen(true)}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
        placeholder="Nom d'équipe…"
        style={{
          width: "100%", padding: "10px 14px", background: "#0f172a",
          border: `1.5px solid ${color}33`, borderRadius: 10, color: "#f1f5f9",
          fontSize: 14, outline: "none", boxSizing: "border-box",
          transition: "border-color 0.2s",
        }}
        onMouseEnter={e => e.target.style.borderColor = color}
        onMouseLeave={e => e.target.style.borderColor = `${color}33`}
      />
      {open && filtered.length > 0 && (
        <div style={{
          position: "absolute", top: "100%", left: 0, right: 0, zIndex: 99,
          background: "#0f172a", border: `1.5px solid ${color}44`, borderRadius: 10,
          maxHeight: 180, overflowY: "auto", marginTop: 4,
          boxShadow: `0 8px 32px ${color}22`,
        }}>
          {filtered.slice(0, 8).map(t => (
            <div
              key={t}
              onMouseDown={() => { onChange(t); setOpen(false); }}
              style={{
                padding: "9px 14px", cursor: "pointer", fontSize: 13, color: "#cbd5e1",
                transition: "background 0.1s",
              }}
              onMouseEnter={e => e.currentTarget.style.background = `${color}22`}
              onMouseLeave={e => e.currentTarget.style.background = "transparent"}
            >
              {t}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function WinGauge({ homeWin, awayWin, homeTeam, awayTeam }) {
  const hw = Math.round(homeWin * 100);
  const aw = Math.round(awayWin * 100);
  return (
    <div style={{ textAlign: "center", margin: "24px 0" }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 10 }}>
        <div>
          <div style={{ fontSize: 28, fontWeight: 800, color: "#f97316", fontFamily: "'Bebas Neue', cursive" }}>{hw}%</div>
          <div style={{ fontSize: 11, color: "#64748b", textTransform: "uppercase", letterSpacing: 1 }}>{homeTeam?.split(" ").pop()}</div>
        </div>
        <div style={{ fontSize: 12, color: "#475569", alignSelf: "center", padding: "0 12px" }}>VS</div>
        <div style={{ textAlign: "right" }}>
          <div style={{ fontSize: 28, fontWeight: 800, color: "#3b82f6", fontFamily: "'Bebas Neue', cursive" }}>{aw}%</div>
          <div style={{ fontSize: 11, color: "#64748b", textTransform: "uppercase", letterSpacing: 1 }}>{awayTeam?.split(" ").pop()}</div>
        </div>
      </div>
      <div style={{ display: "flex", height: 8, borderRadius: 4, overflow: "hidden", background: "#1e293b" }}>
        <div style={{ width: `${hw}%`, background: "linear-gradient(90deg,#ea580c,#f97316)", transition: "width 1s ease" }} />
        <div style={{ width: `${aw}%`, background: "linear-gradient(90deg,#3b82f6,#2563eb)", transition: "width 1s ease" }} />
      </div>
    </div>
  );
}

export default function App() {
  const [teamHome, setTeamHome] = useState("");
  const [teamAway, setTeamAway] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const predict = async () => {
    if (!teamHome.trim() || !teamAway.trim()) {
      setError("Entrez les deux équipes.");
      return;
    }
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ team_home: teamHome, team_away: teamAway, league: "NBA" }),
      });
      if (!res.ok) throw new Error(`Erreur ${res.status}`);
      const data = await res.json();
      setResult(data);
    } catch (e) {
      setError("Impossible de contacter le serveur. Vérifiez que le backend est démarré.");
    } finally {
      setLoading(false);
    }
  };

  const mc = result?.monte_carlo;
  const ff = result?.four_factors;
  const hs = result?.home_stats;
  const as_ = result?.away_stats;

  return (
    <div style={{
      minHeight: "100vh", background: "#020617", color: "#f1f5f9",
      fontFamily: "'DM Sans', 'Helvetica Neue', sans-serif",
      padding: "0 0 40px",
    }}>
      {/* Header */}
      <div style={{
        padding: "28px 24px 20px",
        background: "linear-gradient(180deg,#0f172a,#020617)",
        borderBottom: "1px solid #1e293b",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
          <span style={{ fontSize: 22 }}>🏀</span>
          <span style={{ fontSize: 20, fontWeight: 800, letterSpacing: -0.5 }}>BasketPredict<span style={{ color: "#f97316" }}>AI</span></span>
        </div>
        <div style={{ fontSize: 12, color: "#475569", letterSpacing: 1 }}>ANALYSE & PRÉDICTION NBA</div>
      </div>

      <div style={{ padding: "20px 20px 0" }}>
        {/* Team inputs */}
        <div style={{ display: "flex", gap: 12, alignItems: "flex-end", marginBottom: 16 }}>
          <TeamInput label="Domicile" value={teamHome} onChange={setTeamHome} color="#f97316" />
          <div style={{ paddingBottom: 12, color: "#334155", fontWeight: 700, fontSize: 12 }}>@</div>
          <TeamInput label="Extérieur" value={teamAway} onChange={setTeamAway} color="#3b82f6" />
        </div>

        {/* Predict button */}
        <button
          onClick={predict}
          disabled={loading}
          style={{
            width: "100%", padding: "14px", background: loading ? "#1e293b" : "linear-gradient(135deg,#f97316,#ea580c)",
            border: "none", borderRadius: 12, color: "#fff", fontWeight: 700, fontSize: 15,
            cursor: loading ? "not-allowed" : "pointer", letterSpacing: 0.5,
            boxShadow: loading ? "none" : "0 4px 24px #f9731644",
            transition: "all 0.2s", marginBottom: 12,
          }}
        >
          {loading ? "Analyse en cours…" : "⚡ Prédire le match"}
        </button>

        {error && (
          <div style={{ background: "#7f1d1d22", border: "1px solid #ef444433", borderRadius: 10, padding: "10px 14px", color: "#fca5a5", fontSize: 13, marginBottom: 16 }}>
            {error}
          </div>
        )}

        {/* Loading shimmer */}
        {loading && (
          <div style={{ textAlign: "center", padding: 40 }}>
            <div style={{ fontSize: 32, animation: "spin 1s linear infinite", display: "inline-block" }}>⚙️</div>
            <div style={{ fontSize: 12, color: "#475569", marginTop: 10, letterSpacing: 1 }}>SIMULATION MONTE CARLO…</div>
            <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
          </div>
        )}

        {/* Results */}
        {result && !loading && (
          <div style={{ animation: "fadeIn 0.5s ease" }}>
            <style>{`@keyframes fadeIn { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:translateY(0); } }`}</style>

            {/* Win probability */}
            <div style={{ background: "#0f172a", borderRadius: 14, padding: "18px 16px", marginBottom: 12 }}>
              <div style={{ fontSize: 10, color: "#475569", letterSpacing: 2, textTransform: "uppercase", marginBottom: 12 }}>Probabilité de victoire</div>
              {mc && <WinGauge homeWin={mc.home_win_pct} awayWin={mc.away_win_pct} homeTeam={result.team_home} awayTeam={result.team_away} />}
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#475569" }}>
                <span>Confiance globale</span>
                <span style={{ color: "#22c55e", fontWeight: 700 }}>{Math.round((result.confidence_global || 0) * 100)}%</span>
              </div>
            </div>

            {/* Score scenarios */}
            {result.scenarios?.length > 0 && (
              <div style={{ background: "#0f172a", borderRadius: 14, padding: "18px 16px", marginBottom: 12 }}>
                <div style={{ fontSize: 10, color: "#475569", letterSpacing: 2, textTransform: "uppercase", marginBottom: 12 }}>Scénarios de score</div>
                {result.scenarios.map((s, i) => (
                  <div key={i} style={{
                    display: "flex", justifyContent: "space-between", alignItems: "center",
                    padding: "10px 0", borderBottom: i < result.scenarios.length - 1 ? "1px solid #1e293b" : "none",
                  }}>
                    <span style={{ fontSize: 12, color: "#94a3b8" }}>{s.label}</span>
                    <span style={{ fontWeight: 700, fontSize: 15, letterSpacing: 1 }}>
                      <span style={{ color: "#f97316" }}>{s.score_home}</span>
                      <span style={{ color: "#334155", margin: "0 6px" }}>–</span>
                      <span style={{ color: "#3b82f6" }}>{s.score_away}</span>
                    </span>
                    <span style={{ fontSize: 11, color: "#22c55e" }}>{Math.round(s.confidence * 100)}%</span>
                  </div>
                ))}
              </div>
            )}

            {/* Stats comparaison */}
            {hs && as_ && (
              <div style={{ background: "#0f172a", borderRadius: 14, padding: "18px 16px", marginBottom: 12 }}>
                <div style={{ fontSize: 10, color: "#475569", letterSpacing: 2, textTransform: "uppercase", marginBottom: 12 }}>Statistiques comparées</div>
                <StatBar label="ELO" home={hs.elo_rating} away={as_.elo_rating} />
                <StatBar label="NET RTG" home={hs.net_rating} away={as_.net_rating} />
                <StatBar label="eFG%" home={hs.efg_pct} away={as_.efg_pct} />
                <StatBar label="PPP" home={hs.ppp} away={as_.ppp} />
                <StatBar label="TOV%" home={hs.tov_pct} away={as_.tov_pct} higherIsBetter={false} />
              </div>
            )}

            {/* Facteur clé */}
            {result.key_factor && (
              <div style={{ background: "#0f172a", borderRadius: 14, padding: "14px 16px", marginBottom: 12, display: "flex", gap: 10, alignItems: "flex-start" }}>
                <span style={{ fontSize: 18 }}>🔑</span>
                <div>
                  <div style={{ fontSize: 10, color: "#475569", letterSpacing: 2, textTransform: "uppercase", marginBottom: 4 }}>Facteur clé</div>
                  <div style={{ fontSize: 13, color: "#cbd5e1", lineHeight: 1.5 }}>{result.key_factor}</div>
                </div>
              </div>
            )}

            {/* Paris recommandés */}
            {result.bet_recommendations?.length > 0 && (
              <div style={{ background: "#0f172a", borderRadius: 14, padding: "18px 16px", marginBottom: 12 }}>
                <div style={{ fontSize: 10, color: "#475569", letterSpacing: 2, textTransform: "uppercase", marginBottom: 12 }}>Paris recommandés</div>
                {result.bet_recommendations.map((b, i) => (
                  <div key={i} style={{
                    display: "flex", justifyContent: "space-between", alignItems: "flex-start",
                    padding: "10px 0", borderBottom: i < result.bet_recommendations.length - 1 ? "1px solid #1e293b" : "none",
                    gap: 8,
                  }}>
                    <div style={{ flex: 1 }}>
                      <div style={{ fontSize: 12, fontWeight: 600, color: b.is_value_bet ? "#fbbf24" : "#cbd5e1" }}>
                        {b.is_value_bet && "⭐ "}{b.label}
                      </div>
                      <div style={{ fontSize: 11, color: "#64748b", marginTop: 2 }}>{b.description}</div>
                    </div>
                    <div style={{ textAlign: "right", flexShrink: 0 }}>
                      <div style={{ fontSize: 12, fontWeight: 700, color: "#22c55e" }}>{Math.round(b.probability * 100)}%</div>
                      <div style={{ fontSize: 10, color: "#475569" }}>{b.bet_type}</div>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Disclaimer */}
            <div style={{ fontSize: 10, color: "#1e293b", textAlign: "center", padding: "8px 0" }}>
              À titre informatif uniquement. Jouez responsable.
            </div>
          </div>
        )}
      </div>
    </div>
  );
                   }
    
