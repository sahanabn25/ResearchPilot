# ResearchPilot: An Autonomous Multi-Agent AI System for Research Discovery, Analysis, and Report Generation

ResearchPilot is an AI-powered research assistant designed to simplify academic research by discovering scholarly papers, generating research questions, summarizing research topics, and identifying potential research limitations.

The current prototype integrates the **Crossref API** for academic publication discovery, **Google Gemini** for AI-powered research planning, and **Streamlit** for an interactive web interface. The project is designed to evolve into a multi-agent system capable of analyzing research papers and generating structured research reports.

## Features

- **Research Paper Discovery:** Retrieve scholarly publication records using the Crossref API.
- **AI-Powered Research Planning:** Generate research questions, topic summaries, and potential limitations using Google Gemini.
- **Relevant Paper Selection:** Identify relevant papers from the retrieved results.
- **Interactive Web Interface:** Explore research results through a Streamlit dashboard.
- **Publication Metadata:** Display paper titles, authors, publication years, DOIs, and source links when available.
- **Structured Output:** Generate research plans in JSON format.
- **Export Support:** Download generated research results for further use.
- **Error Handling:** Handle API failures, invalid responses, and missing configuration.

## System Architecture

```text
                User
                  |
                  v
          Streamlit Interface
                  |
                  v
          Python Backend
                  |
          +-------+-------+
          |               |
          v               v
    Crossref API      Google Gemini
          |               |
          v               v
    Paper Metadata   Research Planning
          |               |
          +-------+-------+
                  |
                  v
        Structured Research Plan
                  |
                  v
          Results Dashboard
             and JSON Export
```

The current implementation provides research discovery and AI-generated planning. Specialized autonomous agents for detailed paper analysis, report generation, and validation are planned extensions.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| AI Model | Google Gemini |
| Research Paper Discovery | Crossref REST API |
| Data Exchange | JSON |
| API Communication | Requests |
| Environment Configuration | python-dotenv |
| Version Control | Git and GitHub |
| Deployment | Streamlit Community Cloud |

## Project Structure

```text
ResearchPilot/
├── agent.py
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .env
```

- `agent.py` — Backend functions for searching papers and generating research plans.
- `app.py` — Streamlit web application.
- `requirements.txt` — Python dependencies.
- `README.md` — Project documentation.
- `.gitignore` — Excludes secrets and unnecessary files.
- `.env` — Local environment variables and API credentials; not committed to GitHub.

## Getting Started

### Prerequisites

- Python 3.10 or later, compatible with the installed dependencies.
- Git.
- A Google Gemini API key.
- An internet connection for API access.

### 1. Clone the Repository

```bash
git clone https://github.com/sahanabn25/ResearchPilot.git
cd ResearchPilot
```

### 2. Create a Virtual Environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux or macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` file should include the dependencies used by the application, such as:

```text
streamlit
google-genai
requests
python-dotenv
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=your_working_gemini_model
```

Replace the placeholders with your actual Gemini API key and the model identifier supported by your account.

**Security:** Never commit `.env` or expose your API key in source code, screenshots, or public repositories.

### 5. Run the Backend

To run the command-line research workflow, use:

```bash
python agent.py "Machine learning for weather forecasting"
```

The backend searches Crossref for relevant publication records and uses Gemini to generate a structured research plan.

### 6. Launch the Web Application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

Enter a research topic in the interface to explore the retrieved papers and generated research plan.

## Example Use Case

**Research Topic:** Machine Learning for Weather Forecasting

ResearchPilot can:

1. Retrieve publication records related to machine learning and weather forecasting.
2. Display available paper metadata and source links.
3. Generate research questions related to weather prediction.
4. Summarize the research topic.
5. Identify relevant papers from the retrieved records.
6. Highlight potential limitations in the available literature or retrieved results.
7. Export the structured research plan as JSON.

AI-generated summaries and relevance assessments should be verified against the original publications. Crossref metadata alone does not establish that a paper is peer-reviewed or that its findings are reliable.

## Deployment

ResearchPilot can be deployed using Streamlit Community Cloud.

1. Push the project to GitHub.
2. Visit [Streamlit Community Cloud](https://share.streamlit.io/).
3. Sign in with GitHub and create a new app.
4. Select the repository `sahanabn25/ResearchPilot`.
5. Set the branch to `main` and the main file path to `app.py`.
6. Add `GEMINI_API_KEY` and `GEMINI_MODEL` in the app's Secrets settings.
7. Deploy the application.

Do not upload the `.env` file to GitHub. Configure deployment secrets through Streamlit Community Cloud.

## Current Development Status

**Implemented in the current prototype**
- Crossref-based publication metadata retrieval.
- Gemini-powered research planning.
- Structured JSON output and validation.
- Command-line research workflow.
- Streamlit frontend implementation.

**Next development milestones**
- Test the complete Streamlit application end to end.
- Add automated tests for search, input validation, and AI response validation.
- Improve paper relevance ranking and metadata verification.
- Extract and analyze accessible full-text research papers.
- Introduce specialized agents for discovery, screening, analysis, report generation, and validation.
- Generate research reports with traceable citations and evidence-supported findings.

## Future Scope

ResearchPilot aims to evolve into a multi-agent research workflow in which specialized agents collaborate to:

- Discover and screen research papers.
- Extract important methodologies, findings, and limitations.
- Compare findings across publications.
- Identify potential research gaps.
- Generate structured literature reviews and research reports.
- Validate claims and citations against available source documents.

These capabilities will be developed incrementally and should not be considered implemented until they are tested.

## Security and Responsible Use

- Keep API credentials in environment variables or deployment secrets.
- Respect API usage limits and source terms.
- Verify publication metadata and references.
- Treat AI-generated summaries and research gaps as suggestions, not verified conclusions.
- Base research claims on accessible source material and provide citations wherever possible.


