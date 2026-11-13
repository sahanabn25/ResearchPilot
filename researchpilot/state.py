"""Shared workflow state: the single structured object agents read from and write to.

Each agent receives the state (or the part it needs), returns structured output,
and the orchestrator stores it here. Everything except `store` (a live vector
index) is JSON-serialisable so a run can be saved, resumed or exported.
"""
import copy
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from typing import Any


@dataclass
class WorkflowState:
    mode: str = "topic"                                     # "topic" | "upload"
    topic: str = ""
    papers: list = field(default_factory=list)              # merged discovery records
    source_status: dict = field(default_factory=dict)       # per-source ok/error
    analyses: dict = field(default_factory=dict)            # paper_id -> analysis
    analysis_stats: dict = field(default_factory=dict)
    gaps: dict = field(default_factory=dict)                # gap-agent output
    paper: dict = field(default_factory=dict)               # Option 2: extraction of the uploaded PDF + parsed profile
    questions: list = field(default_factory=list)           # questionnaire shown to the user
    answers: dict = field(default_factory=dict)             # question_id -> user's answer
    profile: dict = field(default_factory=dict)             # structured user project (from answers)
    evaluation: dict = field(default_factory=dict)          # comparison/evaluation output
    report: dict = field(default_factory=dict)              # structured final report (see reporting/)
    errors: list = field(default_factory=list)              # {"stage", "message"}
    events: list = field(default_factory=list)              # trace of what happened
    completed_stages: list = field(default_factory=list)
    steps_used: int = 0
    store: Any = field(default=None, repr=False, compare=False)  # not serialised

    def log(self, stage: str, message: str) -> None:
        self.events.append({"time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "stage": stage, "message": message})

    def add_error(self, stage: str, message: str) -> None:
        self.errors.append({"stage": stage, "message": message})
        self.log(stage, f"ERROR: {message}")

    def to_dict(self) -> dict:
        return {f.name: copy.deepcopy(getattr(self, f.name)) for f in fields(self) if f.name != "store"}

    @classmethod
    def from_dict(cls, data: dict) -> "WorkflowState":
        known = {f.name for f in fields(cls)} - {"store"}
        return cls(**{k: v for k, v in data.items() if k in known})
