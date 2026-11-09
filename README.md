# iRacing Setup RAG & Telemetry Engineering MCP Server (`iracing-engineer-mcp`)

A production-ready **Model Context Protocol (MCP)** server that combines **Garage61 60 Hz telemetry ingestion**, a **deterministic NumPy/SciPy vehicle dynamics engine**, and a **ChromaDB Vector RAG** built from the [Learn Car Setup for iRacing GitBook](https://learnsetup.gitbook.io/learn-car-setup-for-iracing).

Designed around an **Elite Performance Driver Coach & Chief Race Engineer** persona (Peter "Bono" Bonnington / Rob Wilson style), the server enforces a strict **Driver-First Engineering Decision Gate**: it mathematically isolates driver input flaws from mechanical balance limitations and holds garage setup adjustments until driving inputs are verified clean.

---

## 1. Repository Structure

```text
iracing-engineer-mcp/
├── README.md
├── pyproject.toml
├── .env.example
├── config.py
├── data/
│   └── gitbook_docs/         # Complete 18-page Markdown corpus from learnsetup.gitbook.io
├── scraper/
│   └── ingest_rag.py         # Markdown ## / ### / #### chunker & metadata classifier
├── telemetry/
│   ├── g61_client.py         # Authenticated Garage61 API v1 client & 60 Hz CSV parser
│   ├── signal_math.py        # 1.0m SciPy spatial resampling & 7 deterministic physics checks
│   └── track_sectors.py      # Pre-mapped corners (Spa, Nürburgring GP, Watkins Glen, Road America) + |a_y| > 0.5G auto-segmentation
├── rag/
│   └── vector_store.py       # Persistent ChromaDB vector store (`all-MiniLM-L6-v2`) & query interface
├── mcp_server/
│   ├── server.py             # FastMCP("iracing-setup-engineer") tools & decision gate orchestration
│   └── persona_prompts.py    # Bono / Rob Wilson coaching system prompt & diagnostics formatter
└── tests/
    ├── test_math.py          # Synthetic telemetry unit tests for deterministic physics checks
    └── test_server.py        # RAG ingestion, Garage61 API client, and MCP decision gate tests
```

---

## 2. Installation & Configuration

### Prerequisites
- Python **3.11+**
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`
- A [Garage61](https://garage61.net) Personal Access / Bearer Token

### Setup

```bash
# 1. Create and activate a virtual environment
uv venv .venv
source .venv/bin/activate

# 2. Install dependencies
uv pip install -e .

# 3. Configure environment variables
cp .env.example .env
```

Edit `.env` and set your Garage61 bearer token:

```ini
GARAGE61_API_KEY=your_garage61_bearer_token_here
CHROMA_DB_PATH=./data/chroma_db
GITBOOK_CACHE_DIR=./data/gitbook_docs
```

---

## 3. Vector RAG Ingestion (`data/gitbook_docs/` -> ChromaDB)

All 18 pages from `https://learnsetup.gitbook.io/learn-car-setup-for-iracing` are pre-scraped and committed directly under `./data/gitbook_docs/`.

`scraper/ingest_rag.py` splits the Markdown files on `##`, `###`, and `####` headers (producing 139 setup chunks), tags every chunk with structured metadata (`phase`, `symptom`, `component`), embeds them with `all-MiniLM-L6-v2`, and persists them to `./data/chroma_db` (which also auto-populates on first query if empty):

```bash
python -m scraper.ingest_rag
```

Each chunk is tagged with:
- **`phase`**: `"entry"`, `"mid"`, `"exit"`, or `"general"`
- **`symptom`**: `"understeer"`, `"oversteer"`, `"tire_wear"`, `"bottoming"`, or `"general"`
- **`component`**: `"arb"`, `"springs"`, `"dampers_low"`, `"dampers_fast"`, `"diff"`, `"camber_toe"`, `"brake_bias"`, `"aero_rake"`

---

## 4. Deterministic Telemetry Physics Engine (`telemetry/signal_math.py`)

Both the user's latest lap and the fastest comparison lap (team lap first, falling back to fastest clean public lap) are interpolated onto a common **1.0-meter spatial distance axis** $s \in [0, \text{TrackLength}]$ using `scipy.interpolate.interp1d`.

For every corner sector, the engine computes 7 deterministic metrics:

| Check | Mathematical Definition | Fault Threshold | Flagged Diagnosis |
| :--- | :--- | :--- | :--- |
| **Braking Rate** | Initial rise time $\Delta t_{\text{rise}}$ to reach $\ge 80\%$ brake pressure | `user_rise > ref_rise + 0.10s` | `lazy brake hit` (`lazy_brake_hit`) |
| **Trail-Off Gradient** | Peak release rate $\left|\frac{d\,\text{Brake}}{ds}\right|$ between turn-in and apex | `> 10%/m` release rate vs smooth bleed | `abrupt brake release unweighting front axle` (`trail_off_warning`) |
| **Coasting / Hesitation** | Contiguous meters where `Brake < 0.05` and `Throttle < 0.05` | `user_coast > ref_coast + 8.0m` | `dead-time coasting` (`dead_time_coasting`) |
| **Friction Circle (G-Sum)** | Combined acceleration $G_{\text{total}} = \sqrt{a_x^2 + a_y^2}$ in transition | $G_{\text{total}} < 1.0\text{G}$ during pedal transition | `G-sum sag during pedal transition` (`g_sum_sag`) |
| **Steering Scrub Index** | Excess lock $\Delta \delta = |\delta_{\text{user}}| - |\delta_{\text{ref}}|$ from turn-in to apex | $\Delta \delta > 12^\circ$ and $v_{\text{min, user}} < v_{\text{min, ref}}$ | `steering scrub / thermal understeer` (`steering_scrub`) |
| **Apex Delta** | Minimum speed delta $\Delta v_{\text{min}}$ and spatial offset $\Delta s_{\text{apex}}$ | $\Delta s_{\text{apex}} < -10.0\text{m}$ | `early apex pinch choking exit` (`early_apex_pinch`) |
| **Exit Traction** | Steering wheel angle $|\delta|$ when throttle crosses $50\%$ | $|\delta| > 25^\circ$ | `premature throttle application with excessive lock` (`premature_throttle_lock`) |

---

## 5. Registering the MCP Server in Claude Desktop & Cursor

### Exposed MCP Tools
1. **`analyze_last_session(car: str = None, track: str = None, specific_corner: int = None)`**
   - Resolves the user's latest lap and the fastest reference lap from Garage61, downloads 60 Hz CSV streams, aligns them onto a 1.0m distance grid, and runs the 7 physics checks.
   - **Decision Gate**:
     - If driving faults are detected (`technique_fault_isolated: true`), sets `allow_setup_adjustments: false` and returns only driver coaching diagnostics.
     - If driving inputs are clean (`technique_fault_isolated: false`), sets `allow_setup_adjustments: true` and attaches targeted setup adjustments from the GitBook RAG.
2. **`query_setup_guide(corner_phase: str, symptom: str, component: str = None)`**
   - Direct vector store RAG lookup for mechanical or aerodynamic setup adjustments.

### Claude Desktop Configuration (`claude_desktop_config.json`)

Add the following entry to your `claude_desktop_config.json` (macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`, Windows: `%APPDATA%\Claude\claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "iracing-setup-engineer": {
      "command": "/absolute/path/to/iRacing-Coach/.venv/bin/python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/absolute/path/to/iRacing-Coach",
      "env": {
        "GARAGE61_API_KEY": "your_garage61_bearer_token_here",
        "CHROMA_DB_PATH": "/absolute/path/to/iRacing-Coach/data/chroma_db",
        "GITBOOK_CACHE_DIR": "/absolute/path/to/iRacing-Coach/data/gitbook_docs"
      }
    }
  }
}
```

### Cursor Configuration (`.cursor/mcp.json`)

Create or update `.cursor/mcp.json` in your workspace (or `~/.cursor/mcp.json` globally):

```json
{
  "mcpServers": {
    "iracing-setup-engineer": {
      "command": "/absolute/path/to/iRacing-Coach/.venv/bin/python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/absolute/path/to/iRacing-Coach",
      "env": {
        "GARAGE61_API_KEY": "your_garage61_bearer_token_here",
        "CHROMA_DB_PATH": "./data/chroma_db",
        "GITBOOK_CACHE_DIR": "./data/gitbook_docs"
      }
    }
  }
}
```

---

## 6. Running Unit Tests

Run the synthetic telemetry math and MCP server test suite with `pytest`:

```bash
.venv/bin/pytest -v
```
