# MODEL UTILS

import re
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords

# download stopwords if not already downloaded
# stopwords help remove common words like "and", "the", etc., for better semantic analysis
nltk.download("stopwords", quiet=True)

# store all stopwords in lowercase for easier matching
STOPWORDS = {w.lower() for w in stopwords.words("english")}

# load SBERT (Sentence-BERT) model for semantic similarity calculations
# forced to use CPU for maximum compatibility with Hugging Face Spaces
sbert_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# curated technical keywords for keyword overlap scoring
TECH_KEYWORDS = {
    # PROGRAMMING LANGUAGES (Tech)
    "python", "java", "c", "c++", "c#", "r", "sql", "pl/sql", "tsql", "nosql", 
    "hadoop", "spark", "go", "golang", "rust", "typescript", "javascript", "bash", 
    "shell", "powershell", "perl", "matlab", "sas", "stata", "scala", "objective-c", 
    "swift", "vb.net", "fortran", "cobol", "dart", "lua", "haskell", "julia", "php", 
    "ruby", "assembly", "f#", "ocaml", "prolog", "vhdl", "verilog", "abap", "apex", 
    "kql", "groovy", "elixir", "crystal",

    # DATA SCIENCE, ML & AI (Tech)
    "pandas", "numpy", "scipy", "scikit-learn", "tensorflow", "keras", "pytorch", 
    "torchvision", "xgboost", "lightgbm", "catboost", "transformers", "bert", "gpt", 
    "gpt-4", "llama", "mistral", "falcon", "stable diffusion", "huggingface", "trl", 
    "peft", "langchain", "llamaindex", "mlflow", "dvc", "optuna", "hyperopt", 
    "ray tune", "prophet", "statsmodels", "automl", "dataiku", "h2o.ai", "rapidminer", 
    "weka", "feature engineering", "eda", "bayesian", "monte carlo", "markov chains", 
    "kalman filters", "reinforcement learning", "deep learning", "graph neural networks", 
    "gan", "cnn", "rnn", "lstm", "gru", "vae", "tabnet", "kmeans", "dbscan", "hdbscan", 
    "svm", "random forest", "gradient boosting", "shap", "lime", "feature importance", 
    "model explainability", "predictive modeling", "time series forecasting", 
    "sequence modeling", "ensemble methods", "meta learning", "few shot learning", 
    "active learning", "self supervised learning", "speech to text", "text to speech", 
    "image segmentation", "object detection", "pose estimation", "video analytics", 
    "ocr", "nlp", "named entity recognition", "topic modeling", "sentiment analysis", 
    "recommendation systems", "search ranking", "personalization", "causal inference", 
    "a/b testing", "experimentation",

    # BIG DATA & DATA ENGINEERING (Tech)
    "databricks", "pyspark", "spark sql", "airflow", "dbt", "snowflake", "redshift", 
    "bigquery", "delta lake", "lakehouse", "data lake", "data warehouse", "hive", 
    "pig", "impala", "flink", "storm", "kafka", "zookeeper", "schema registry", 
    "ksql", "elasticsearch", "logstash", "kibana", "splunk", "nifi", "beam", "orc", 
    "parquet", "avro", "iceberg", "columnar storage", "cdc", "change data capture", 
    "data governance", "data catalog", "data lineage", "data mesh", "real time analytics", 
    "stream processing", "batch processing", "etl", "elt", "data pipeline", 
    "orchestration",

    # DATABASES (Tech)
    "postgresql", "mysql", "mariadb", "oracle", "sql server", "teradata", "greenplum", 
    "cassandra", "couchdb", "neo4j", "graphdb", "mongodb", "dynamodb", "cosmosdb", 
    "redis", "memcached", "hbase", "arangodb", "influxdb", "timescaledb", "faunadb", 
    "realm", "orientdb", "tidb", "cockroachdb", "clickhouse", "vector db", "pinecone", 
    "milvus", "weaviate", "chroma", "pgvector",

    # CLOUD PLATFORMS & TOOLS (Tech)
    "aws", "gcp", "azure", "digitalocean", "ibm cloud", "oracle cloud", "aliyun", 
    "heroku", "vercel", "netlify", "cloudflare", "linode", "vultr",
    "s3", "ec2", "eks", "emr", "aws lambda", "batch", "fargate", "sagemaker", 
    "cloudformation", "cdk", "terraform", "pulumi", "ansible", "chef", "puppet",
    "vertex ai", "dataproc", "bigtable", "lookml", "powerbi service", "database",

    # DEVOPS & CI/CD (Tech)
    "docker", "kubernetes", "helm", "jenkins", "gitlab ci", "circleci", "travisci", 
    "github actions", "argoCD", "tekton", "nexus", "artifactory", "sonarqube", 
    "prometheus", "grafana", "nagios", "new relic", "datadog", "sentry", 
    "site reliability", "chaos engineering", "canary deployment", "blue green deployment", 
    "load balancing", "log aggregation",

    # WEB & API DEVELOPMENT (Tech)
    "react", "next.js", "vue", "nuxt.js", "angular", "svelte", "jquery", 
    "astro", "d3.js", "chart.js", "three.js", "bootstrap", "tailwind",
    "flask", "django", "fastapi", "tornado", "express", "spring", 
    "spring boot", "node.js", "graphql", "rest api", "soap", "openapi", 
    "swagger", "postman", "grpc", "websocket", "web components",
    "microservices", "service mesh", "istio", "envoy", "API", "APIs",

    # MOBILE DEVELOPMENT & AR/VR (Tech)
    "android", "ios", "kotlin", "swiftui", "jetpack compose", "flutter",
    "react native", "cordova", "ionic", "unity", "unreal engine", "arkit",
    "arcore", "vr", "xr", "hololens",

    # CYBERSECURITY & NETWORKING (Tech)
    "wireshark", "metasploit", "nessus", "splunk", "crowdstrike", "snort",
    "palo alto", "checkpoint", "nmap", "burpsuite", "owasp", 
    "threat modeling", "penetration testing", "zero trust", 
    "iam", "mfa", "sso", "ldap", "kerberos", "vpn", "tls", "ssl", "ipsec",
    "cissp", "iso27001", "nist", "soc2", "fedramp", "firewalls",

    # PRODUCT & PROJECT MANAGEMENT (Tech)
    "jira", "confluence", "trello", "asana", "monday.com", "clickup", 
    "ms project", "visio", "figma", "mural", "lucidchart",
    "scrum", "kanban", "safe agile", "scaled agile", "prince2", "pmp",
    "okrs", "kpis", "roadmapping", "stakeholder management",

    # FINANCE & QUANT (Tech/Finance)
    "bloomberg", "quantlib", "alpaca", "ibkr", "yfinance", "backtrader",
    "risk modeling", "value at risk", "black scholes", "monte carlo",
    "options pricing", "portfolio optimization", "algorithmic trading",
    "quantitative research", "hedge funds", "factor models",

    # HEALTHCARE & BIOINFORMATICS (Tech/Science)
    "biopython", "bioconductor", "genomics", "rna-seq", "crispr", 
    "epic systems", "cerner", "hl7", "fhir", "clinical trials", 
    "electronic health records", "drug discovery", "protein folding",

    # BLOCKCHAIN & WEB3 (Tech)
    "solidity", "ethereum", "polygon", "chainlink", "uniswap", "defi",
    "nft", "smart contracts", "truffle", "hardhat", "web3.js", 
    "ethers.js", "metamask", "hyperledger", "ipfs", "zk rollups",

    # IOT, ROBOTICS & EDGE COMPUTING (Tech)
    "edge computing", "iot", "mqtt", "embedded systems", "raspberry pi",
    "arduino", "robotics", "autonomous vehicles", "lidar", "slam", 
    "ros", "coppeliasim", "gazebo", "tinyml",

    # VISUALIZATION & BI (Tech)
    "powerbi", "tableau", "qlik", "lookerstudio", "looker", "superset",
    "ggplot2", "matplotlib", "seaborn", "plotly", "bokeh", "holoviews",
    "altair", "vega-lite", "d3", "echarts", "dash", "streamlit",

    # OTHER EMERGING TECH (Tech)
    "digital twin", "smart city", "5g", "6g", "quantum computing", 
    "qiskit", "braket", "neuromorphic computing", "synthetic data",

    # MARKETING & ADVERTISING (Non-Tech)
    "seo", "sem", "search engine optimization", "content marketing", "digital marketing", 
    "social media marketing", "branding", "brand management", "market research", 
    "marketing strategy", "marketing campaigns", "email marketing", "google analytics", 
    "ppc", "pay per click", "advertising", "copywriting", "public relations", "media relations",
    "social media", "influencer marketing", "crm", "customer relationship management", 
    "lead generation", "lead nurturing", "competitive analysis", "product marketing", 
    "go-to-market strategy", "marketing analytics",

    # SALES & BUSINESS DEVELOPMENT (Non-Tech)
    "sales strategy", "business development", "b2b sales", "b2c sales", "account management", 
    "territory management", "pipeline management", "prospecting", "cold calling", 
    "negotiation", "sales presentations", "sales forecasting", "quotas", "customer acquisition", 
    "relationship management", "client retention", "solution selling", "consultative selling", 
    "cross-selling", "upselling", "salesforce", "salesforce.com", "hubspot", "crm software", 
    "sales operations", "channel sales", "inside sales", "outside sales",

    # FINANCE & ACCOUNTING (Non-Tech)
    "financial reporting", "financial analysis", "budgeting", "forecasting", "financial modeling", 
    "accounts payable", "accounts receivable", "general ledger", "bookkeeping", "reconciliation", 
    "auditing", "taxation", "tax compliance", "us gaap", "ifrs", "accounting", "cost accounting", 
    "management accounting", "quickbooks", "sap fi/co", "oracle financials", "excel", "advanced excel", 
    "financial planning", "portfolio management", "investment analysis", "risk management", 
    "variance analysis", "mergers and acquisitions", "capital budgeting",

    # HUMAN RESOURCES & RECRUITING (Non-Tech)
    "recruiting", "talent acquisition", "full-cycle recruiting", "sourcing", "interviewing", 
    "onboarding", "new hire orientation", "hris", "human resources information system", 
    "payroll", "benefits administration", "compensation", "employee relations", 
    "performance management", "training and development", "succession planning", 
    "labor law", "employment law", "compliance training", "diversity and inclusion", 
    "hr policies", "organizational development", "change management", "employee engagement", 
    "hr analytics", "workforce planning", "applicant tracking system", "ATS", 

    # OPERATIONS, SUPPLY CHAIN & LOGISTICS (Non-Tech)
    "supply chain management", "logistics", "operations management", "procurement", 
    "purchasing", "strategic sourcing", "vendor management", "inventory management", 
    "warehouse management", "distribution", "transportation management", "fleet management", 
    "demand forecasting", "production planning", "materials management", "mrp", 
    "erp systems", "sap scm", "lean manufacturing", "six sigma", "continuous improvement", 
    "kaizen", "quality control", "quality assurance", "capacity planning", "scheduling", 
    "order fulfillment", "import/export", "trade compliance", "osha compliance", "iso 9001", 
    "iso 14001", "just-in-time", "supply chain optimization",

    # ENGINEERING & MANUFACTURING (Non-Tech)
    "cad", "autocad", "solidworks", "catia", "ansys", "3d modeling", "product design", 
    "mechanical design", "electrical design", "schematics", "circuit design", 
    "pcb design", "plc programming", "hvac", "cnc machining", "machining", "welding", 
    "fabrication", "blueprint reading", "gd&t", "metrology", "labview", "simulink", 
    "feasibility analysis", "prototyping", "industrial automation", "robotics", 
    "industrial safety", "preventive maintenance", "maintenance management", 
    "root cause analysis", "process engineering", "manufacturing engineering", 
    "plant operations", "project engineering", "civil engineering", "structural engineering", 
    "piping design", "revit", "bim", "construction management", "project scheduling", 
    "cost estimation", "contract management",

    # DESIGN & CREATIVE (Non-Tech)
    "graphic design", "visual design", "ui design", "ux design", "user experience", 
    "user interface", "wireframing", "prototyping", "user research", "usability testing", 
    "adobe photoshop", "adobe illustrator", "adobe indesign", "adobe xd", "figma", "sketch", 
    "invision", "adobe premiere", "after effects", "video editing", "motion graphics", 
    "animation", "3d modeling", "blender", "maya", "autodesk 3ds max", "illustration", 
    "brand design", "branding", "typography", "color theory", "storyboarding", "photography", 
    "photo editing", "creative direction", "art direction",

    # EDUCATION & TRAINING (Non-Tech)
    "teaching", "curriculum development", "lesson planning", "instructional design", 
    "classroom management", "educational technology", "e-learning", "online teaching", 
    "training delivery", "workshop facilitation", "public speaking", "presentations", 
    "learning management system", "LMS", "student assessment", "grading", "tutoring", 
    "mentoring", "special education", "ESL", "english as a second language", 
    "early childhood education", "adult education", "higher education administration", 
    "academic advising", "course design", "faculty development", "education policy", 
    "standardized testing", "common core", 

    # LEGAL & COMPLIANCE (Non-Tech)
    "legal research", "legal writing", "litigation", "civil litigation", "criminal litigation", 
    "commercial litigation", "case management", "contract drafting", "contract negotiation", 
    "document review", "compliance", "regulatory compliance", "corporate law", "labor law", 
    "employment law", "intellectual property", "patents", "trademarks", "copyrights", 
    "licensing", "due diligence", "mediation", "arbitration", "court filings", 
    "legal documentation", "brief writing", "pleadings", "discovery", "trial preparation", 
    "westlaw", "lexisnexis", "compliance auditing", "risk assessment", "gdpr", "hipaa", 
    "aml", "anti money laundering", "kYC", "know your customer", "corporate governance",

    # HEALTHCARE & MEDICAL (Non-Tech)
    "patient care", "vital signs", "patient assessment", "clinical skills", "nursing", 
    "acute care", "critical care", "emergency care", "surgery", "medication administration", 
    "phlebotomy", "iv therapy", "diagnostic imaging", "radiology", "lab testing", 
    "medical terminology", "electronic medical records", "emr", "ehr", "hipaa", 
    "insurance billing", "icd-10", "cpt coding", "medical coding", "scheduling (medical)", 
    "triage", "infection control", "telemedicine", "healthcare management", 
    "hospital administration", "clinical research", "public health", "rehabilitation", 
    "physical therapy", "occupational therapy", "dental care", "pharmacy", "patient education", 
    "disease management",

    # ADMINISTRATIVE & OFFICE SUPPORT (Non-Tech)
    "office administration", "office management", "calendar management", "scheduling", 
    "meeting coordination", "travel arrangements", "clerical work", "data entry", 
    "file management", "record keeping", "reception", "front office", "ms office", 
    "microsoft office", "word processing", "excel (spreadsheets)", "powerpoint", 
    "outlook", "google workspace", "shorthand", "typing", "document preparation", 
    "report writing", "transcription", "office equipment", "printing and scanning", 
    "email management", "bookkeeping", "invoicing", "purchase orders", 
    "multitasking", "time management", "confidentiality"
}

# preprocesses text by normalizing it for semantic similarity calculations
# steps:
# 1. convert to lowercase
# 2. remove punctuation and special characters
# 3. remove common stopwords to keep only meaningful words
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = [t for t in text.split() if t not in STOPWORDS]
    return " ".join(tokens)

# generates a numerical vector (embedding) for the text using SBERT
# embeddings are used to compute semantic similarity between two texts
def get_embedding(text: str) -> np.ndarray:
    return sbert_model.encode(preprocess(text))

# computes cosine similarity between two vectors
# returns a value between 0 and 1 where:
# 1 = identical meaning, 0 = completely unrelated
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))

# extracts all technical keywords present in a given text
# uses the curated TECH_KEYWORDS list for matching
def extract_keywords(text: str) -> set:
    text_lower = text.lower()
    tokens = set(re.findall(r"\b[a-zA-Z\+\#\d]+\b", text_lower))
    return {t for t in tokens if t in TECH_KEYWORDS}

# computes weighted keyword score between resume and job description
# gives higher weight to specialized or in-demand keywords (like pytorch and xgboost)
def weighted_keyword_score(resume: str, jd: str) -> float:
    resume_skills = extract_keywords(resume)
    jd_skills = extract_keywords(jd)
    if not jd_skills:
        return 0.0
    weights = {s: 2.0 if s in {"pytorch", "xgboost"} else 1.0 for s in jd_skills}
    matched_weight = sum(weights[s] for s in resume_skills & jd_skills if s in weights)
    total_weight = sum(weights.values())
    return matched_weight / total_weight if total_weight > 0 else 0.0

# combines semantic similarity and keyword score into a final score
# alpha is a weight that determines the importance of semantic similarity
# default alpha = 0.6 (semantic similarity is slightly more important than keyword overlap)
def compute_final_score(resume: str, jd: str, alpha: float = 0.6) -> tuple:
    semantic_score = cosine_similarity(get_embedding(resume), get_embedding(jd))
    keyword_score = weighted_keyword_score(resume, jd)
    final_score = alpha * semantic_score + (1 - alpha) * keyword_score
    return final_score, semantic_score, keyword_score

# generates recruiter-style feedback based on the scores and missing technical keywords
# provides actionable suggestions to improve resume alignment with the job description
def generate_feedback(resume: str, jd: str, semantic_score: float, keyword_score: float) -> str:
    missing_keywords = extract_keywords(jd) - extract_keywords(resume)
    feedback = []

    if semantic_score >= 0.7 and keyword_score >= 0.6:
        feedback.append("Your resume aligns well with the job description. Great job.")
    elif semantic_score >= 0.5:
        feedback.append("Your resume is contextually relevant but could highlight technical expertise better.")
    else:
        feedback.append("Your resume shows limited alignment. Consider tailoring it more closely to the role.")

    if missing_keywords:
        feedback.append(f"Consider adding these technical skills: {', '.join(sorted(missing_keywords))}.")
    else:
        feedback.append("You have covered the essential technical skills for this role.")

    return " ".join(feedback)