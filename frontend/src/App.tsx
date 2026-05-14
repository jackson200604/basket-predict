import { useState } from "react";

export default function App() {
  const [team, setTeam] = useState("");
  const [result, setResult] = useState("");

  const predict = async () => {
    const res = await fetch("TON_URL_RENDER/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ team })
    });
    const data = await res.json();
    setResult(data.prediction);
  };

  return (
    <div style={{ textAlign: "center", marginTop: "50px" }}>
      <h1>Prédiction du panier</h1>
      <input
        value={team}
        onChange={(e) => setTeam(e.target.value)}
        placeholder="Entrer équipe"
      />
      <button onClick={predict}>Prédire</button>
      <p>{result}</p>
    </div>
  );
}
