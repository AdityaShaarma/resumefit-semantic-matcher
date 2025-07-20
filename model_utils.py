import re
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords


# Import OpenAI for rewrite suggestions
try:
    from openai import OpenAI
    import os
    from dotenv import load_dotenv

    # Load environment variables from .env (works locally & on HF Spaces if set in settings)
    load_dotenv()

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
    OPENAI_ENABLED = bool(OPENAI_API_KEY)
except ImportError:
    OPENAI_CLIENT = None
    OPENAI_ENABLED = False

# download stopwords if not already available
nltk.download("stopwords", quiet=True)
STOPWORDS = {w.lower() for w in stopwords.words("english")}

# load SBERT model for semantic similarity – forced CPU for Hugging Face compatibility
sbert_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# curated set of technical keywords (extended for better detection)
TECH_KEYWORDS = {
    # Programming Languages
    "python", "java", "c++", "c#", "r", "sql", "nosql", "hadoop", "spark",
    "go", "rust", "typescript", "javascript", "bash", "shell", "perl",
    "matlab", "sas", "scala", "objective-c", "swift", "vb.net", "fortran",
    "cobol", "dart", "lua", "haskell", "julia", "php", "ruby", "assembly",
    "f#", "ocaml", "prolog", "vhdl", "verilog",

    # Data Science & ML
    "pandas", "numpy", "scipy", "scikit-learn", "tensorflow", "keras", "pytorch",
    "xgboost", "lightgbm", "catboost", "transformers", "bert", "gpt",
    "llama", "stable diffusion", "huggingface", "mlflow", "optuna",
    "hyperopt", "ray tune", "prophet", "statsmodels", "autoML", "dataiku",
    "rapidminer", "weka", "feature engineering", "eda", "bayesian",
    "markov chains", "reinforcement learning", "deep learning",
    "graph neural networks", "gan", "cnn", "rnn", "lstm", "gru", "vae",
    "kmeans", "dbscan", "svm", "random forest", "gradient boosting",
    "shap", "lime", "feature importance", "model explainability",
    "predictive modeling", "time series forecasting", "ensemble methods",

    # Big Data & Data Engineering
    "databricks", "airflow", "dbt", "snowflake", "redshift", "bigquery",
    "delta lake", "hive", "pig", "impala", "flink", "storm", "kafka",
    "zookeeper", "elasticsearch", "logstash", "kibana", "splunk",
    "nifi", "lakehouse", "data lake", "data warehouse", "parquet",
    "orc", "avro", "columnar storage", "etl", "elt", "data pipeline",
    "data governance", "data catalog", "data lineage", "cdc", "change data capture",

    # Databases
    "postgresql", "mysql", "mariadb", "oracle", "sql server", "teradata",
    "cassandra", "couchdb", "neo4j", "mongodb", "dynamodb", "cosmosdb",
    "redis", "memcached", "hbase", "arangodb", "influxdb", "timescaledb",
    "faunadb", "realm", "orientdb", "tidb", "cockroachdb",

    # Cloud Platforms
    "azure", "aws", "gcp", "digitalocean", "ibm cloud", "oracle cloud",
    "aliyun", "heroku", "vercel", "netlify", "cloudflare", "linode",
    "aws lambda", "cloud functions", "s3", "ec2", "eks", "emr",
    "cloudformation", "terraform", "pulumi", "ansible", "chef", "puppet",

    # DevOps & CI/CD
    "docker", "kubernetes", "helm", "jenkins", "gitlab ci", "circleci",
    "travisci", "github actions", "argoCD", "tekton", "nexus", "sonarqube",
    "prometheus", "grafana", "nagios", "new relic", "datadog",
    "site reliability", "chaos engineering", "canary deployment",
    "blue-green deployment", "monitoring", "log aggregation",

    # Web Development & APIs
    "react", "next.js", "vue", "angular", "svelte", "jquery", "d3.js",
    "chart.js", "three.js", "bootstrap", "tailwind", "flask", "django",
    "fastapi", "express", "spring", "node.js", "graphql", "rest api",
    "soap", "openapi", "swagger", "postman", "grpc", "websocket",
    "microservices", "service mesh", "istio", "envoy",

    # Mobile Development
    "android", "ios", "kotlin", "swiftui", "jetpack compose", "flutter",
    "react native", "cordova", "ionic", "unity", "unreal engine",

    # Cybersecurity & Networking
    "wireshark", "metasploit", "nessus", "splunk", "crowdstrike",
    "snort", "palo alto", "checkpoint", "nmap", "burpsuite", "owasp",
    "threat modeling", "penetration testing", "zero trust",
    "iam", "mfa", "ldap", "kerberos", "vpn", "tls", "ssl", "ipsec",
    "cissp", "iso27001", "nist", "soc2", "fedramp", "firewalls",

    # Product & Project Management
    "jira", "confluence", "trello", "asana", "monday.com", "clickup",
    "ms project", "visio", "figma", "mural", "lucidchart",
    "scrum", "kanban", "safe agile", "prince2", "pmp", "okrs", "kpis",

    # Finance & Quant
    "bloomberg", "quantlib", "alpaca", "ibkr", "yfinance",
    "backtrader", "risk modeling", "value at risk", "black scholes",
    "monte carlo", "options pricing", "portfolio optimization",
    "algorithmic trading", "quantitative research", "hedge funds",

    # Healthcare & Bioinformatics
    "biopython", "bioconductor", "genomics", "rna-seq", "crispr",
    "epic systems", "cerner", "hl7", "fhir", "clinical trials",
    "electronic health records", "drug discovery", "protein folding",

    # Blockchain & Web3
    "solidity", "ethereum", "polygon", "chainlink", "uniswap",
    "defi", "nft", "smart contracts", "truffle", "hardhat",
    "web3.js", "ethers.js", "metamask", "hyperledger", "ipfs",

    # IoT, Robotics, Edge
    "edge computing", "iot", "embedded systems", "raspberry pi",
    "arduino", "robotics", "autonomous vehicles", "lidar", "slam",
    "ros", "coppeliasim", "gazebo",

    # Visualization
    "powerbi", "tableau", "qlik", "lookerstudio", "looker", "superset",
    "ggplot2", "matplotlib", "seaborn", "plotly", "bokeh", "holoviews",
    "altair", "vega-lite", "d3", "echarts",

    # Other Keywords
    "a/b testing", "feature flags", "experimentation", "customer segmentation",
    "recommendation engine", "search ranking", "personalization",
    "ocr", "nlp", "speech recognition", "voice ai", "computer vision",
    "digital twin", "smart city", "5g", "quantum computing"
}

# cleans and normalizes text for semantic analysis
def preprocess(text: str) -> str:
    # lowercases, removes punctuation, removes stopwords
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = [t for t in text.split() if t not in STOPWORDS]
    return " ".join(tokens)

# converts text into a semantic vector using SBERT
def get_embedding(text: str) -> np.ndarray:
    clean_text = preprocess(text)
    return sbert_model.encode(clean_text)

# computes cosine similarity between two vectors
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))

# extracts meaningful technical keywords using regex and the curated list
def extract_keywords(text: str) -> set:
    text_lower = text.lower()
    single_tokens = set(re.findall(r"\b[a-zA-Z\+\#\d]+\b", text_lower))
    matched = {t for t in single_tokens if t in TECH_KEYWORDS}

    # check for multi-word keywords
    for phrase in TECH_KEYWORDS:
        if " " in phrase and phrase in text_lower:
            matched.add(phrase)
    return matched

# computes weighted keyword overlap, giving higher weight to specialized skills
def weighted_keyword_score(resume: str, jd: str) -> float:
    resume_skills = extract_keywords(resume)
    jd_skills = extract_keywords(jd)
    if not jd_skills:
        return 0.0

    weights = {skill: 2.0 if skill in {"pytorch", "xgboost", "transformers"} else 1.0 for skill in jd_skills}
    matched_weight = sum(weights[s] for s in resume_skills & jd_skills if s in weights)
    total_weight = sum(weights.values())
    return matched_weight / total_weight if total_weight > 0 else 0.0

# blends semantic similarity and keyword overlap into a final weighted score
def compute_final_score(resume: str, jd: str, alpha: float = 0.6) -> tuple:
    resume_vec = get_embedding(resume)
    jd_vec = get_embedding(jd)
    semantic_score = cosine_similarity(resume_vec, jd_vec)
    keyword_score = weighted_keyword_score(resume, jd)
    final_score = alpha * semantic_score + (1 - alpha) * keyword_score
    return final_score, semantic_score, keyword_score

# provides recruiter-style suggestions (no awkward numbers)
def generate_feedback(resume: str, jd: str, semantic_score: float, keyword_score: float) -> str:
    missing_keywords = extract_keywords(jd) - extract_keywords(resume)
    feedback = []

    if semantic_score >= 0.7 and keyword_score >= 0.6:
        feedback.append("Your resume aligns well with the job description. Great job!")
    elif semantic_score >= 0.5:
        feedback.append("Your resume is contextually relevant but could highlight technical expertise better.")
    else:
        feedback.append("Your resume shows limited alignment. Consider tailoring it to match the role more closely.")

    if missing_keywords:
        feedback.append(f"Consider adding or emphasizing these skills: {', '.join(sorted(missing_keywords))}.")
    else:
        feedback.append("You have covered the essential technical skills for this role.")

    return " ".join(feedback)

# Logic for resume bullet point feedback

ACTION_VERBS = {
    # Results & Impact
    "achieved", "accelerated", "boosted", "drove", "delivered", "exceeded",
    "increased", "maximized", "optimized", "outperformed", "reduced", 
    "surpassed", "transformed", "enhanced", "expanded", "improved", 
    "strengthened", "streamlined", "saved", "cut", "amplified",

    # Leadership & Initiative
    "led", "spearheaded", "orchestrated", "directed", "pioneered", 
    "coordinated", "oversaw", "supervised", "organized", "executed", 
    "mobilized", "facilitated", "initiated", "chaired", "managed",
    "mentored", "trained", "guided", "inspired", "delegated",

    # Strategy & Planning
    "developed", "designed", "architected", "devised", "formulated", 
    "mapped", "conceptualized", "prioritized", "planned", "strategized",
    "structured", "outlined", "positioned", "standardized", "automated",

    # Analysis & Research
    "analyzed", "evaluated", "assessed", "benchmarked", "forecasted", 
    "diagnosed", "modeled", "predicted", "quantified", "researched", 
    "interpreted", "synthesized", "validated", "investigated",

    # Innovation & Problem-Solving
    "created", "invented", "engineered", "launched", "deployed", 
    "customized", "innovated", "resolved", "troubleshot", 
    "debugged", "built", "prototyped", "deconstructed", "simulated",

    # Communication & Collaboration
    "collaborated", "negotiated", "partnered", "liaised", "advocated",
    "aligned", "consulted", "presented", "pitched", "conveyed",
    "influenced", "persuaded", "articulated", "demonstrated",
    "documented", "reported",

    # Teaching & Mentorship
    "trained", "mentored", "coached", "taught", "educated", 
    "empowered", "supported", "advised", "counseled",

    # Product & Business
    "commercialized", "marketed", "monetized", "negotiated", 
    "scaled", "capitalized", "differentiated", "targeted", 
    "promoted", "drove adoption", "launched products",

    # Creative & Design
    "designed", "illustrated", "visualized", "modeled", 
    "crafted", "edited", "curated", "authored"
}

# LLM-BASED REWRITE (only for weak bullet points)
def rewrite_with_llm(bullet: str) -> str:
    if not OPENAI_ENABLED:
        return "(Enable OpenAI for rewrite suggestions)"
    prompt = (
        f"Rewrite this resume bullet to be stronger, using an action verb, numbers, "
        f"and technical skills where possible. Keep it concise and recruiter-friendly:\n\n"
        f"Original: {bullet}\nRewritten:"
    )
    response = OPENAI_CLIENT.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=60,
        temperature=0.5
    )
    return response.choices[0].message.content.strip()

# analyzes resume bullet points and returns only suggestions for weak ones
def bullet_point_feedback(resume_text: str) -> str:
    # split text into lines and filter likely bullets (short sentences or bullet symbols)
    lines = [line.strip() for line in resume_text.split("\n") if line.strip()]
    bullet_points = [l for l in lines if l.startswith(("•", "-", "*")) or len(l.split()) <= 15]

    if not bullet_points:
        return "No clear bullet points detected. Consider formatting experience using concise, action-driven bullet points."

    suggestions = []
    for point in bullet_points:
        p_lower = point.lower()
        has_action = any(verb in p_lower for verb in ACTION_VERBS)
        has_number = any(char.isdigit() for char in point)
        has_tech = any(tech in p_lower for tech in TECH_KEYWORDS)

        # only critique weak bullets
        if not (has_action and has_number and has_tech):
            feedback_parts = []
            if not has_action:
                feedback_parts.append("start with a strong action verb")
            if not has_number:
                feedback_parts.append("include measurable results or numbers")
            if not has_tech:
                feedback_parts.append("highlight relevant technical skills")

            # optional: use LLM to rewrite
            llm_suggestion = rewrite_with_llm(point) if OPENAI_ENABLED else ""
            if llm_suggestion:
                suggestions.append(f"• \"{point}\" → {', '.join(feedback_parts)}.\n   Suggested rewrite: {llm_suggestion}")
            else:
                suggestions.append(f"• \"{point}\" → Consider to {', '.join(feedback_parts)}.")
        # strong bullets are skipped (no redundant praise)

    return "\n".join(suggestions) if suggestions else "All bullet points are strong as written."