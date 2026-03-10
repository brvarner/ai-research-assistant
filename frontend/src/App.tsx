import { useState } from "react";
import "./App.css";

interface ChatResponse {
  answer: string;
  sources: {
    content: string;
    source: string;
    page: number;
    similarity: number;
  }[];
}

function App() {
  const [query, setQuery] = useState("");
  const [selectedSource, setSelectedSource] = useState("");
  const [response, setResponse] = useState<ChatResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function askQuestion() {
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    setResponse(null);

    try {
      const res = await fetch("http://localhost:8000/search/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query,
          source: selectedSource || null,
        }),
      });

      if (!res.ok) throw new Error(`API error: ${res.status}`);
      const data = await res.json();
      setResponse(data);
    } catch (err) {
      setError("Something went wrong. Please try again.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <div className="container">
        <div>
          <h1>Keyboard Research App</h1>
          <p>
            Ask the app a question about these keyboards, based on their
            manuals. I'll soon add some web-scraped content to the backend to
            add nuance to the answers.
          </p>
        </div>
        <div className="query-display">
          <div>
            <label htmlFor="query" className="label">
              Query
            </label>
            <input
              type="text"
              id="query"
              value={query}
              onChange={(e) => setQuery(e.currentTarget.value)}
              onKeyDown={(e) => e.key === "Enter" && askQuestion()}
            />
          </div>
          <div>
            <label htmlFor="selectedSource" className="label">
              Source? (Optional)
            </label>
            <input
              type="text"
              id="selectedSource"
              value={selectedSource}
              onChange={(e) => setSelectedSource(e.currentTarget.value)}
            />
          </div>
          <button onClick={askQuestion}>
            {loading ? "Thinking..." : "Ask the Question"}
          </button>
        </div>

        {error && <p style={{ color: "red" }}>{error}</p>}

        {response && (
          <div>
            <h2>Answer</h2>
            <p>{response.answer}</p>

            <h3>Sources</h3>
            {response.sources.map((source, i) => (
              <div key={i}>
                <p>
                  {source.source} — Page {source.page} — Score:{" "}
                  {source.similarity}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>
    </>
  );
}

export default App;
