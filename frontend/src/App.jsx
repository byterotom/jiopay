import { useState, useEffect, useRef } from "react";
import axios from "axios";
import { FiSend } from "react-icons/fi";
import "./App.css";

function App() {
  const [messages, setMessages] = useState([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const chatEndRef = useRef(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    const newMessages = [...messages, { role: "user", content: query }];
    setMessages(newMessages);
    setQuery("");
    setLoading(true);

    try {
      const res = await axios.post("https://jiopay-chatbot-2.onrender.com/chat", {
        query,
        top_k: 5,
        embed_model: "all-MiniLM-L6-v2",
      });

      setMessages([
        ...newMessages,
        { role: "assistant", content: res.data.answer || "No response." },
      ]);
    } catch (err) {
      console.error(err);
      setMessages([
        ...newMessages,
        { role: "assistant", content: "❌ Error: Could not fetch response." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  return (
    <div className="app-container">
      {/* Chat Card */}
      <div className="chat-card">
        <header className="chat-header">💬 JioPay Assistant</header>

        <main className="chat-window">
          {messages.map((msg, idx) => (
            <div key={idx} className={`message ${msg.role}`}>
              {msg.content}
            </div>
          ))}
          {loading && <div className="message assistant">Thinking...</div>}
          <div ref={chatEndRef}></div>
        </main>

        <form onSubmit={handleSubmit} className="input-bar">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Type your question..."
          />
          <button type="submit" disabled={loading}>
            {loading ? "..." : <FiSend size={18} />}
          </button>
        </form>
      </div>
    </div>
  );
}

export default App;
