import Link from "next/link";

const featureCards = [
  {
    title: "Duplicate Intelligence",
    text: "Exact and semantic duplicate detection catches repeated samples before they inflate confidence.",
    tag: "Deduplication"
  },
  {
    title: "Label Reliability",
    text: "Class imbalance and centroid-based mismatch checks surface label noise early.",
    tag: "Label Quality"
  },
  {
    title: "Toxicity and Safety",
    text: "Content moderation signals estimate unsafe rows and show distribution by label.",
    tag: "Safety"
  },
  {
    title: "Domain Mixing Alerts",
    text: "Embedding clustering reveals hidden domain drift inside a single dataset.",
    tag: "Domain Drift"
  },
  {
    title: "Train/Test Leakage",
    text: "Similarity scanning across splits quantifies leakage risk before training.",
    tag: "Leakage"
  },
  {
    title: "Actionable Reporting",
    text: "Get a quality score, issue severity, row-level flags, and direct cleanup actions.",
    tag: "Reporting"
  }
];

export default function LandingPage() {
  return (
    <main className="landing-shell">
      <header className="landing-nav">
        <div className="brand-mark">Dataset Quality Analyzer</div>
        <Link href="/analyze" className="nav-cta">
          Open Analyzer
        </Link>
      </header>

      <section className="hero hero-background">
        <div className="hero-panel">
          <p className="hero-kicker">Pre-training Dataset Intelligence</p>
          <h1 className="hero-title">Train Models on Better Data, Not Guesswork</h1>
          <p className="hero-subtitle">
            Detect duplicates, label noise, toxicity, domain drift, and leakage before model training. Use one pipeline to gate dataset quality.
          </p>

          <div className="hero-actions">
            <Link href="/analyze" className="action-primary">
              Start Analysis
            </Link>
            <a href="#features" className="action-secondary">
              View Features
            </a>
          </div>
        </div>
      </section>

      <section id="features" className="features">
        <div className="section-heading">
          <p className="section-kicker">Feature Stack</p>
          <h2>Built as Independent Analyzer Modules</h2>
        </div>

        <div className="bento-grid">
          {featureCards.map((card, idx) => (
            <article key={card.title} className={`bento-card card-${(idx % 6) + 1}`}>
              <span className="card-tag">{card.tag}</span>
              <h3>{card.title}</h3>
              <p>{card.text}</p>
            </article>
          ))}
        </div>
      </section>
    </main>
  );
}
