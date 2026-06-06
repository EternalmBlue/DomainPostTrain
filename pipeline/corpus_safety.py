from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from pipeline.cpt_discovery import discover_corpus
from pipeline.utils import (
    SAFETY_PREAMBLE,
    clean_text_noise,
    load_config,
    normalize_path_for_report,
    read_json,
    read_text,
    resolve_input_path,
    resolve_training_path,
    setup_logging,
    utc_now,
    write_json,
    write_text,
)


REFUSAL_MARKERS = [
    "cannot",
    "can not",
    "do not",
    "should not",
    "must not",
    "refuse",
    "not allowed",
    "documentation does not specify",
    "approved process",
]

DANGEROUS_TERMS = [
    "bypass access",
    "avoid audit",
    "disable audit",
    "steal credential",
    "extract secret",
    "private key",
    "production password",
    "delete evidence",
    "impersonate",
]

SOURCE_PATH_TERMS = [
    "training_audit_not_for_training",
    "src/main/java",
    "src/main/kotlin",
    "src/main/resources",
    ".git/",
    ".git\\",
]

PRIVATE_REPO_PATTERNS = [
    re.compile(r"https?://(?:[^/\s]+/){1,2}[^/\s]+/(?:private|internal|gitlab|gitea)[^)\s]*", re.I),
    re.compile(r"git@[^:\s]+:[^/\s]+/[^)\s]+\.git", re.I),
]

SECRET_PATTERNS = [
    ("password_assignment", re.compile(r"\b(password|passwd|pwd)\b\s*[:=]\s*['\"]?[^'\"\s]{6,}", re.I)),
    ("secret_assignment", re.compile(r"\b(secret|client_secret)\b\s*[:=]\s*['\"]?[^'\"\s]{8,}", re.I)),
    ("token_assignment", re.compile(r"\b(access[_-]?token|refresh[_-]?token|token)\b\s*[:=]\s*['\"]?[A-Za-z0-9_\-\.]{16,}", re.I)),
    ("api_key_assignment", re.compile(r"\b(api[_-]?key|apikey)\b\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}", re.I)),
    ("webhook_url", re.compile(r"https?://[^\s]*(webhook|hooks)[^\s]+", re.I)),
    ("license_key_assignment", re.compile(r"\blicense[_ -]?key\b\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{12,}", re.I)),
    ("private_key_block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("jdbc_connection", re.compile(r"jdbc:(mysql|mariadb|postgresql|sqlserver)://[^\s]+", re.I)),
    ("mongodb_connection", re.compile(r"mongodb(?:\+srv)?://[^\s]+", re.I)),
    ("redis_connection", re.compile(r"redis://[^\s]+", re.I)),
    ("ip_port_account_combo", re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}:\d{2,5}\b.{0,80}\b(user|username|account|password|pwd)\b", re.I)),
    ("authorization_server", re.compile(r"(license server|licensing server|auth server).{0,80}(https?://|[A-Za-z0-9.-]+:\d{2,5})", re.I)),
]

CODE_LINE_PATTERN = re.compile(
    r"^\s*(package\s+[\w.]+;|import\s+[\w.*]+;|public\s+|private\s+|protected\s+|class\s+\w+|"
    r"interface\s+\w+|fun\s+\w+\(|def\s+\w+\(|from\s+\w+\s+import|SELECT\s+|INSERT\s+|UPDATE\s+|DELETE\s+|"
    r"CREATE\s+TABLE|@\w+|if\s*\(.+\)\s*\{|for\s*\(.+\)\s*\{|while\s*\(.+\)\s*\{|return\s+.+;|"
    r"[A-Za-z0-9_<>]+\s+[A-Za-z0-9_]+\s*=\s*.+;)\s*$",
    re.I,
)

FENCED_CODE_START = re.compile(r"^\s*```([A-Za-z0-9_+-]*)\s*$")


def _line_has_refusal_marker(line: str) -> bool:
    lower = line.lower()
    return any(marker.lower() in lower for marker in REFUSAL_MARKERS)


def _line_snippet(line: str) -> str:
    clean = line.strip()
    if len(clean) <= 140:
        return clean
    return clean[:137] + "..."


def _line_is_public_permission_reference(line: str) -> bool:
    lower = line.lower()
    access_context = any(marker in lower for marker in ["access", "approval", "permission"])
    unsafe_instruction = any(marker in lower for marker in ["how to", "guide", "steps", "bypass", "avoid audit"])
    return access_context and not unsafe_instruction


def _issue(
    issues: list[dict[str, Any]],
    severity: str,
    category: str,
    path: Path,
    line_number: int | None,
    message: str,
    pattern: str | None = None,
    line: str | None = None,
) -> None:
    issues.append(
        {
            "severity": severity,
            "category": category,
            "path": str(path),
            "display_path": normalize_path_for_report(path),
            "line": line_number,
            "message": message,
            "pattern": pattern,
            "snippet": _line_snippet(line or "") if line else None,
        }
    )


def has_safety_boundary(text: str) -> bool:
    lower = text.lower()
    required_markers = ["must not", "credentials", "documentation does not specify", "approved"]
    return sum(1 for marker in required_markers if marker in lower) >= 2 or "safety" in lower


def inspect_text(path: Path, text: str, max_consecutive_code_lines: int) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    lines = text.splitlines()
    consecutive_code_start = None
    consecutive_count = 0
    in_fence = False
    fence_lang = ""
    fence_start = 0
    fence_lines = 0

    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        fence_match = FENCED_CODE_START.match(line)
        if fence_match:
            if not in_fence:
                in_fence = True
                fence_lang = fence_match.group(1).lower()
                fence_start = idx
                fence_lines = 0
            else:
                high_risk_lang = fence_lang in {"java", "kotlin", "kt", "py", "python", "sql"}
                if high_risk_lang or fence_lines > max_consecutive_code_lines:
                    _issue(
                        issues,
                        "medium",
                        "source_code_block",
                        path,
                        fence_start,
                        f"Fenced code block may contain source-like material ({fence_lines} lines, lang={fence_lang or 'plain'}).",
                        pattern=fence_lang or "fenced_code",
                    )
                in_fence = False
                fence_lang = ""
                fence_lines = 0
            continue
        if in_fence:
            fence_lines += 1

        is_code_line = bool(CODE_LINE_PATTERN.match(line))
        if is_code_line:
            if consecutive_code_start is None:
                consecutive_code_start = idx
            consecutive_count += 1
        else:
            if consecutive_count > max_consecutive_code_lines and consecutive_code_start is not None:
                _issue(
                    issues,
                    "medium",
                    "consecutive_code_lines",
                    path,
                    consecutive_code_start,
                    f"More than {max_consecutive_code_lines} consecutive source-like lines detected.",
                    pattern="source_like_lines",
                )
            consecutive_code_start = None
            consecutive_count = 0

        lower = stripped.lower()
        for term in SOURCE_PATH_TERMS:
            if term.lower() in lower:
                severity = "high" if term != "src/main/resources" else "medium"
                _issue(
                    issues,
                    severity,
                    "excluded_or_source_path",
                    path,
                    idx,
                    f"Corpus mentions a path that should not be trained without review: {term}",
                    pattern=term,
                    line=line,
                )

        for pattern_name, pattern in SECRET_PATTERNS:
            if pattern.search(line):
                _issue(
                    issues,
                    "high",
                    "sensitive_pattern",
                    path,
                    idx,
                    f"Potential sensitive value matched pattern {pattern_name}. Value is intentionally not printed.",
                    pattern=pattern_name,
                )

        for pattern in PRIVATE_REPO_PATTERNS:
            if pattern.search(line):
                _issue(
                    issues,
                    "high",
                    "private_repository",
                    path,
                    idx,
                    "Potential private repository URL detected.",
                    pattern="private_repo_url",
                )

        for term in DANGEROUS_TERMS:
            if term.lower() in lower:
                if _line_is_public_permission_reference(line):
                    _issue(
                        issues,
                        "info",
                        "public_permission_reference",
                        path,
                        idx,
                        f"Dangerous-looking term appears in a public permission node/reference: {term}",
                        pattern=term,
                        line=line,
                    )
                elif _line_has_refusal_marker(line):
                    _issue(
                        issues,
                        "info",
                        "safety_boundary_mention",
                        path,
                        idx,
                        f"Dangerous term appears in a refusal/safety-boundary sentence: {term}",
                        pattern=term,
                        line=line,
                    )
                else:
                    _issue(
                        issues,
                        "high",
                        "dangerous_training_content",
                        path,
                        idx,
                        f"Dangerous or abuse-enabling topic appears without clear refusal framing: {term}",
                        pattern=term,
                        line=line,
                    )

    if consecutive_count > max_consecutive_code_lines and consecutive_code_start is not None:
        _issue(
            issues,
            "medium",
            "consecutive_code_lines",
            path,
            consecutive_code_start,
            f"More than {max_consecutive_code_lines} consecutive source-like lines detected.",
            pattern="source_like_lines",
        )
    if in_fence:
        _issue(issues, "medium", "unclosed_code_fence", path, fence_start, "Unclosed fenced code block detected.")
    return issues


def run_preflight(config: dict[str, Any], config_path: Path, input_paths: list[Path] | None = None) -> dict[str, Any]:
    if input_paths is None:
        discovered = discover_corpus(config, config_path)
        input_paths = [Path(path) for path in discovered.get("selected_paths", [])]
    else:
        discovered = {
            "status": "ok" if input_paths else "not_found",
            "selected_paths": [str(path) for path in input_paths],
            "strategy": "explicit_input_paths",
        }

    safety_cfg = config.get("safety", {})
    corpus_cfg = config.get("corpus", {})
    max_code_lines = int(safety_cfg.get("max_consecutive_code_lines", 3))
    min_chars = int(corpus_cfg.get("min_text_chars", 1000))
    append_safety_preamble = bool(corpus_cfg.get("append_safety_preamble", True))

    issues: list[dict[str, Any]] = []
    total_chars = 0
    safety_boundary_present = False
    file_reports = []
    for path in input_paths:
        if not path.exists():
            _issue(issues, "high", "missing_input", path, None, "Selected corpus file is missing.")
            continue
        text = clean_text_noise(read_text(path))
        total_chars += len(text)
        if has_safety_boundary(text):
            safety_boundary_present = True
        file_issues = inspect_text(path, text, max_code_lines)
        issues.extend(file_issues)
        file_reports.append(
            {
                "path": str(path),
                "display_path": normalize_path_for_report(path),
                "chars": len(text),
                "lines": len(text.splitlines()),
                "issue_count": len(file_issues),
            }
        )

    if total_chars < min_chars:
        issues.append(
            {
                "severity": "warning",
                "category": "short_corpus",
                "path": None,
                "display_path": None,
                "line": None,
                "message": f"Corpus has {total_chars} chars, below configured min_text_chars={min_chars}. Training is allowed but overfitting risk is high.",
                "pattern": None,
                "snippet": None,
            }
        )

    if not safety_boundary_present and append_safety_preamble:
        issues.append(
            {
                "severity": "warning",
                "category": "missing_safety_boundary",
                "path": None,
                "display_path": None,
                "line": None,
                "message": "Corpus lacks an explicit safety boundary. Dataset preparation will prepend the configured safety preamble.",
                "pattern": "SAFETY_PREAMBLE",
                "snippet": SAFETY_PREAMBLE[:120] + "...",
            }
        )

    high_risk_count = sum(1 for issue in issues if issue["severity"] == "high")
    medium_count = sum(1 for issue in issues if issue["severity"] == "medium")
    warning_count = sum(1 for issue in issues if issue["severity"] == "warning")
    info_count = sum(1 for issue in issues if issue["severity"] == "info")
    block_on_high_risk = bool(safety_cfg.get("block_on_high_risk", True))

    status = "blocked" if block_on_high_risk and high_risk_count > 0 else "passed"
    return {
        "status": status,
        "discovery": discovered,
        "selected_paths": [str(path) for path in input_paths],
        "files": file_reports,
        "total_chars": total_chars,
        "safety_boundary_present": safety_boundary_present,
        "append_safety_preamble": append_safety_preamble,
        "high_risk_count": high_risk_count,
        "medium_count": medium_count,
        "warning_count": warning_count,
        "info_count": info_count,
        "issues": issues,
        "timestamp_utc": utc_now(),
    }


def write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# CPT Corpus Preflight Report",
        "",
        f"Status: **{report['status']}**",
        f"Timestamp UTC: `{report['timestamp_utc']}`",
        f"Total characters: `{report['total_chars']}`",
        f"Safety boundary present: `{report['safety_boundary_present']}`",
        "",
        "## Selected Files",
        "",
    ]
    for item in report["files"]:
        lines.append(f"- `{item['display_path']}`: {item['chars']} chars, {item['lines']} lines, {item['issue_count']} issues")
    if not report["files"]:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Issue Summary",
            "",
            f"- High risk: {report['high_risk_count']}",
            f"- Medium: {report['medium_count']}",
            f"- Warning: {report['warning_count']}",
            f"- Info: {report['info_count']}",
            "",
            "## Issues",
            "",
        ]
    )
    if not report["issues"]:
        lines.append("No issues detected.")
    else:
        for issue in report["issues"]:
            location = issue.get("display_path") or "corpus"
            if issue.get("line"):
                location += f":{issue['line']}"
            lines.append(f"- **{issue['severity']}** `{issue['category']}` at `{location}`: {issue['message']}")
            if issue.get("snippet") and issue["category"] != "sensitive_pattern":
                lines.append(f"  - Context: {issue['snippet']}")
    lines.append("")
    write_text(path, "\n".join(lines))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run safety preflight checks for CPT corpus files.")
    parser.add_argument("--config", default="configs/domain_post_training.yaml", help="Path to domain_post_training.yaml.")
    parser.add_argument("--input_paths", nargs="*", default=None, help="Explicit corpus file paths.")
    parser.add_argument("--allow_unsafe_corpus", action="store_true", help="Write reports but do not exit non-zero on high-risk findings.")
    parser.add_argument("--output_json", default=None, help="Output JSON path.")
    parser.add_argument("--output_md", default=None, help="Output Markdown path.")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    logger = setup_logging("preflight_corpus_check", args.verbose)
    config, config_path = load_config(args.config)
    input_paths = None
    if args.input_paths:
        input_paths = [resolve_input_path(path, config_path) for path in args.input_paths]
    else:
        discovered_path = resolve_training_path("outputs/logs/discovered_corpus.json", "outputs/logs/discovered_corpus.json")
        discovered = read_json(discovered_path, default=None)
        if discovered and discovered.get("status") == "ok":
            input_paths = [Path(path) for path in discovered.get("selected_paths", [])]
    report = run_preflight(config, config_path, input_paths)
    json_path = resolve_training_path(args.output_json, "outputs/logs/preflight_report.json")
    md_path = resolve_training_path(args.output_md, "outputs/logs/preflight_report.md")
    write_json(json_path, report)
    write_markdown_report(md_path, report)
    logger.info("Preflight report: %s", md_path)
    if report["status"] == "blocked" and not args.allow_unsafe_corpus:
        logger.error("High-risk corpus findings detected. Training must stop unless --allow_unsafe_corpus is passed.")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
