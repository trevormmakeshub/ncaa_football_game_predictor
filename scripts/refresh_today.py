"""Pull NCAA football results and retrain the prior-form table through today.

ESPN FBS standings identify the teams. ESPN team schedules supply dates,
teams, venues, and final scores. Ranking weeks come from the ESPN core
API. A sportsbook is not called. A game's own score is not a feature.
This does not price a bet.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

try:
    from scipy.special import expit
    from sklearn.linear_model import LinearRegression, RidgeClassifier
except ImportError:
    def expit(value):
        array = np.asarray(value, dtype=float)
        return 1.0 / (1.0 + np.exp(-array))

    class RidgeClassifier:
        def __init__(self, alpha: float = 1.0) -> None:
            self.alpha = alpha
            self.classes_ = np.array([0, 1])
            self.coef_ = None
            self.intercept_ = 0.0

        def fit(self, samples, target):
            samples = np.asarray(samples, dtype=float)
            target = np.asarray(target, dtype=float).reshape(-1)
            self.classes_ = np.unique(target.astype(int))
            center = samples.mean(axis=0)
            gram = (samples - center).T @ (samples - center)
            gram = gram + (self.alpha * np.eye(samples.shape[1]))
            self.coef_ = np.linalg.solve(gram, (samples - center).T @ (target - target.mean()))
            self.intercept_ = float(target.mean() - center @ self.coef_)
            return self

        def decision_function(self, samples):
            samples = np.asarray(samples, dtype=float)
            return samples @ self.coef_ + self.intercept_

    class LinearRegression:
        def fit(self, samples, target):
            samples = np.asarray(samples, dtype=float)
            target = np.asarray(target, dtype=float).reshape(-1)
            ones = np.column_stack([np.ones(len(samples)), samples])
            coef, *_ = np.linalg.lstsq(ones, target, rcond=None)
            self.intercept_ = float(coef[0])
            self.coef_ = coef[1:]
            return self

        def predict(self, samples):
            samples = np.asarray(samples, dtype=float)
            return samples @ self.coef_ + self.intercept_


EASTERN = ZoneInfo("America/New_York")
AS_OF = datetime.now(EASTERN).date()
NEXT_THROUGH = AS_OF + timedelta(days=7)
HOLDOUT_START = date(2026, 8, 1)
ROOT = Path(__file__).resolve().parents[1]
SCHEDULE_PATH = ROOT / "data" / "schedules" / "games.csv"
TEAMS_PATH = ROOT / "data" / "teams" / "fbs_teams.csv"
TEAM_STATS_PATH = ROOT / "data" / "stats_team" / "stats_team_reg_2026.csv"
PLAYER_STATS_PATH = ROOT / "data" / "stats_player" / "stats_player_reg_2026.csv"
RANKINGS_PATH = ROOT / "data" / "rankings" / "rankings.csv"
COMPLETED_PATH = ROOT / "artifacts" / "ncaa_completed.csv"
NEXT_PATH = ROOT / "artifacts" / "ncaa_next_games.csv"
SEEDS_PATH = ROOT / "artifacts" / "seeds.csv"
MAX_DATE_PATH = ROOT / "artifacts" / "last_max_date.txt"
README_PATH = ROOT / "README.md"
NOTES_PATH = ROOT / "NOTES.md"
STANDINGS_URL = "https://site.api.espn.com/apis/v2/sports/football/college-football/standings"
SCHEDULE_URL = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/teams/{team_id}/schedule?season={season}"
RANKINGS_WEEK_URL = "https://sports.core.api.espn.com/v2/sports/football/leagues/college-football/seasons/{year}/types/2/weeks/{week}/rankings"
PLAYER_URL = "https://site.web.api.espn.com/apis/common/v3/sports/football/college-football/statistics/byathlete?limit=250&season=2026&seasontype=2"
POLL_NAMES = {"AP Top 25", "AFCA Coaches Poll", "Playoff Committee Rankings"}
FEATURES = [
    "home_win_rate",
    "away_win_rate",
    "home_ppg",
    "away_ppg",
    "home_papg",
    "away_papg",
    "home_rest_days",
    "away_rest_days",
]
RECENT_FEATURES = [
    "home_recent_win_rate",
    "away_recent_win_rate",
    "home_recent_margin",
    "away_recent_margin",
]
LINE = "not a bet"
CALIBRATION_PATH = ROOT / "artifacts" / "calibration.csv"
HIGH_PATH = ROOT / "artifacts" / "high_confidence.csv"


def fetch_json(url: str, timeout: int = 40) -> dict:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ncaa-football-game-predictor/1.0", "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_retry(url: str, attempts: int = 3) -> dict | None:
    last = None
    for _ in range(attempts):
        try:
            return fetch_json(url)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
    print(f"fetch failed {type(last).__name__} {url}")
    return None


def fbs_teams() -> list[dict]:
    payload = fetch_json(STANDINGS_URL)
    teams: dict[str, dict] = {}

    def walk(node: dict, conference: str) -> None:
        name = node.get("name") or conference
        current = name if node.get("isConference") else conference
        for entry in (node.get("standings") or {}).get("entries") or []:
            team = entry.get("team") or {}
            team_id = str(team.get("id") or "")
            abbreviation = team.get("abbreviation") or ""
            if not team_id or not abbreviation:
                continue
            teams[team_id] = {
                "team_id": team_id,
                "abbreviation": abbreviation,
                "name": team.get("displayName") or team.get("name") or "",
                "conference": current,
            }
        for child in node.get("children") or []:
            walk(child, current)

    for child in payload.get("children") or []:
        walk(child, child.get("name") or "")
    rows = list(teams.values())
    rows.sort(key=lambda row: row["abbreviation"])
    return rows


def schedule_urls(teams: list[dict]) -> list[str]:
    urls = []
    for team in teams:
        for season in (2025, 2026):
            base = SCHEDULE_URL.format(team_id=team["team_id"], season=season)
            urls.append(base)
            urls.append(base + "&seasontype=3")
    return urls


def collect_events(teams: list[dict]) -> tuple[list[dict], int]:
    events: dict[str, dict] = {}
    failures = 0
    urls = schedule_urls(teams)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch_retry, url): url for url in urls}
        for future in as_completed(futures):
            payload = future.result()
            if payload is None:
                failures += 1
                continue
            for event in payload.get("events") or []:
                season_type = season_type_id(event)
                if season_type not in (2, 3):
                    continue
                events[str(event["id"])] = event
    return list(events.values()), failures


def season_type_id(event: dict) -> int | None:
    season_type = event.get("seasonType")
    value = season_type.get("type", season_type.get("id")) if isinstance(season_type, dict) else season_type
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def score_value(score: object) -> int | None:
    if isinstance(score, dict):
        score = score.get("displayValue", score.get("value"))
    if score in (None, ""):
        return None
    return int(float(score))


def game_rows(events: list[dict]) -> list[dict]:
    rows = []
    for event in events:
        competition = event["competitions"][0]
        competitors = competition.get("competitors", [])
        home = next((team for team in competitors if team.get("homeAway") == "home"), None)
        away = next((team for team in competitors if team.get("homeAway") == "away"), None)
        if home is None or away is None:
            continue
        stamp = datetime.fromisoformat(event["date"].replace("Z", "+00:00")).astimezone(EASTERN)
        status = competition.get("status", {}).get("type", {})
        status_name = status.get("name") or ""
        home_score = score_value(home.get("score"))
        away_score = score_value(away.get("score"))
        final = (
            status_name == "STATUS_FINAL"
            and stamp.date() <= AS_OF
            and home_score is not None
            and away_score is not None
            and not (home_score == 0 and away_score == 0)
        )
        season_year = int((event.get("season") or {}).get("year") or 0)
        rows.append(
            {
                "event_id": str(event["id"]),
                "stamp": stamp,
                "date": stamp.date().isoformat(),
                "season": season_year,
                "game_type": "POST" if season_type_id(event) == 3 else "REG",
                "away": away["team"].get("abbreviation") or "",
                "home": home["team"].get("abbreviation") or "",
                "away_name": away["team"].get("displayName") or "",
                "home_name": home["team"].get("displayName") or "",
                "away_score": away_score if final else None,
                "home_score": home_score if final else None,
                "played": final,
                "status": status_name,
                "venue": (competition.get("venue") or {}).get("fullName") or "",
            }
        )
    rows.sort(key=lambda row: (row["stamp"], row["event_id"]))
    return rows


def add_prior_form(rows: list[dict]) -> None:
    history: dict[str, list[tuple[date, int, int, int]]] = {}
    for row in rows:
        for side in ("home", "away"):
            games = history.get(row[side], [])
            if games:
                window = games[-4:]
                row[f"{side}_win_rate"] = sum(item[3] for item in games) / len(games)
                row[f"{side}_ppg"] = sum(item[1] for item in games) / len(games)
                row[f"{side}_papg"] = sum(item[2] for item in games) / len(games)
                row[f"{side}_rest_days"] = (date.fromisoformat(row["date"]) - games[-1][0]).days
                row[f"{side}_recent_win_rate"] = sum(item[3] for item in window) / len(window)
                row[f"{side}_recent_margin"] = sum(item[1] - item[2] for item in window) / len(window)
            else:
                row[f"{side}_win_rate"] = None
                row[f"{side}_ppg"] = None
                row[f"{side}_papg"] = None
                row[f"{side}_rest_days"] = None
                row[f"{side}_recent_win_rate"] = None
                row[f"{side}_recent_margin"] = None
        if not row["played"]:
            continue
        if row["home_score"] == row["away_score"]:
            row["home_win"] = None
            home_win_flag = 0
            away_win_flag = 0
        else:
            home_win_flag = int(row["home_score"] > row["away_score"])
            away_win_flag = 1 - home_win_flag
            row["home_win"] = home_win_flag
        history.setdefault(row["home"], []).append(
            (date.fromisoformat(row["date"]), row["home_score"], row["away_score"], home_win_flag)
        )
        history.setdefault(row["away"], []).append(
            (date.fromisoformat(row["date"]), row["away_score"], row["home_score"], away_win_flag)
        )


def win_probability(row: dict, center, scale, ridge, positive_is_home: bool, features: list[str] | None = None) -> float | None:
    features = FEATURES if features is None else features
    if any(row.get(feature) is None for feature in features):
        return None
    sample = (np.array([row[feature] for feature in features], dtype=float) - center) / scale
    decision = float(ridge.decision_function(sample.reshape(1, -1))[0])
    if not positive_is_home:
        decision = -decision
    return float(expit(decision))


def fit_model(rows: list[dict], features: list[str] | None = None):
    features = FEATURES if features is None else features
    frame = pd.DataFrame(rows)
    matrix = frame[features].to_numpy(dtype=float)
    center = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale == 0] = 1.0
    scaled = (matrix - center) / scale
    target = frame["home_win"].to_numpy(dtype=int)
    ridge = RidgeClassifier(alpha=1.0)
    ridge.fit(scaled, target)
    home_points = LinearRegression().fit(scaled, frame["home_score"].to_numpy(dtype=float))
    away_points = LinearRegression().fit(scaled, frame["away_score"].to_numpy(dtype=float))
    positive_is_home = int(ridge.classes_[1]) == 1
    return center, scale, ridge, home_points, away_points, positive_is_home


def holdout_accuracy(completed: list[dict]) -> tuple[int, int, float | None]:
    train_rows = [
        row for row in completed
        if date.fromisoformat(row["date"]) < HOLDOUT_START
        and row.get("home_win") is not None
        and all(row.get(feature) is not None for feature in FEATURES)
    ]
    test_rows = [
        row for row in completed
        if date.fromisoformat(row["date"]) >= HOLDOUT_START
        and row.get("home_win") is not None
        and all(row.get(feature) is not None for feature in FEATURES)
    ]
    if len(train_rows) < 2 or len(test_rows) == 0 or len({row["home_win"] for row in train_rows}) < 2:
        return len(train_rows), len(test_rows), None
    center, scale, ridge, _, _, positive_is_home = fit_model(train_rows)
    correct = 0
    for row in test_rows:
        probability = win_probability(row, center, scale, ridge, positive_is_home)
        correct += int(probability is not None and int(probability >= 0.5) == row["home_win"])
    return len(train_rows), len(test_rows), correct / len(test_rows)


def score_rows(completed: list[dict], upcoming: list[dict], points_mode: str = "regression") -> int:
    trainable = [
        row for row in completed
        if row.get("home_win") is not None and all(row.get(feature) is not None for feature in FEATURES)
    ]
    if len(trainable) < 2 or len({row["home_win"] for row in trainable}) < 2:
        return 0
    center, scale, ridge, home_points, away_points, positive_is_home = fit_model(trainable)
    residual_home = residual_away = None
    residual_center = residual_scale = None
    if points_mode == "residual":
        frame = pd.DataFrame(trainable)
        matrix = frame[FEATURES].to_numpy(dtype=float)
        residual_center = matrix.mean(axis=0)
        residual_scale = matrix.std(axis=0)
        residual_scale[residual_scale == 0] = 1.0
        scaled = (matrix - residual_center) / residual_scale
        residual_home = LinearRegression().fit(
            scaled,
            frame["home_score"].to_numpy(dtype=float) - frame["home_ppg"].to_numpy(dtype=float),
        )
        residual_away = LinearRegression().fit(
            scaled,
            frame["away_score"].to_numpy(dtype=float) - frame["away_ppg"].to_numpy(dtype=float),
        )
    for row in completed:
        if row.get("home_win") is None and row["played"] and row["home_score"] == row["away_score"]:
            row["home_win_prob"] = None
            continue
        row["home_win_prob"] = win_probability(row, center, scale, ridge, positive_is_home)
    for row in upcoming:
        probability = win_probability(row, center, scale, ridge, positive_is_home)
        row["home_win_prob"] = probability
        if probability is None:
            annotate_prediction(row)
            continue
        sample = ((np.array([row[feature] for feature in FEATURES], dtype=float) - center) / scale).reshape(1, -1)
        home = float(home_points.predict(sample)[0])
        away = float(away_points.predict(sample)[0])
        if points_mode == "blend" and row.get("home_ppg") is not None and row.get("away_ppg") is not None:
            regression_total = home + away
            blended_total = 0.5 * regression_total + 0.5 * (float(row["home_ppg"]) + float(row["away_ppg"]))
            margin = home - away
            home = (blended_total + margin) / 2
            away = (blended_total - margin) / 2
        elif (
            points_mode == "residual"
            and residual_home is not None
            and row.get("home_ppg") is not None
            and row.get("away_ppg") is not None
        ):
            residual_sample = ((np.array([row[feature] for feature in FEATURES], dtype=float) - residual_center) / residual_scale).reshape(1, -1)
            home = float(row["home_ppg"]) + float(residual_home.predict(residual_sample)[0])
            away = float(row["away_ppg"]) + float(residual_away.predict(residual_sample)[0])
        row["predicted_home_points"] = home
        row["predicted_away_points"] = away
        row["predicted_points"] = home + away
        annotate_prediction(row)
    return len(trainable)


def annotate_prediction(row: dict) -> None:
    home_points = row.get("predicted_home_points")
    away_points = row.get("predicted_away_points")
    if home_points is not None and away_points is not None:
        row["predicted_total"] = float(home_points) + float(away_points)
    else:
        row["predicted_total"] = row.get("predicted_points")
    probability = row.get("home_win_prob")
    if probability is None:
        row["pick_side"] = None
        row["pick_team"] = None
        row["confidence"] = None
        row["confidence_tier"] = None
        return
    probability = float(probability)
    if probability >= 0.5:
        row["pick_side"] = "home"
        row["pick_team"] = row.get("home")
    else:
        row["pick_side"] = "away"
        row["pick_team"] = row.get("away")
    row["confidence"] = abs(probability - 0.5)
    if probability >= 0.60 or probability <= 0.40:
        row["confidence_tier"] = "high"
    elif probability >= 0.55 or probability <= 0.45:
        row["confidence_tier"] = "medium"
    else:
        row["confidence_tier"] = "low"


def feature_metrics(rows: list[dict], features: list[str], required: list[str]) -> dict | None:
    def ready(start: date | None, end: date | None) -> list[dict]:
        chosen = []
        for row in rows:
            if row.get("home_win") is None or any(row.get(name) is None for name in required):
                continue
            game_date = date.fromisoformat(row["date"])
            if start is not None and game_date < start:
                continue
            if end is not None and game_date >= end:
                continue
            chosen.append(row)
        return chosen

    train_rows = ready(None, HOLDOUT_START)
    test_rows = ready(HOLDOUT_START, None)
    if len(train_rows) < 2 or not test_rows or len({row["home_win"] for row in train_rows}) < 2:
        return None
    center, scale, ridge, home_points, away_points, positive_is_home = fit_model(train_rows, features)
    train_frame = pd.DataFrame(train_rows)
    train_matrix = train_frame[features].to_numpy(dtype=float)
    residual_center = train_matrix.mean(axis=0)
    residual_scale = train_matrix.std(axis=0)
    residual_scale[residual_scale == 0] = 1.0
    residual_scaled = (train_matrix - residual_center) / residual_scale
    home_delta = LinearRegression().fit(
        residual_scaled,
        train_frame["home_score"].to_numpy(dtype=float) - train_frame["home_ppg"].to_numpy(dtype=float),
    )
    away_delta = LinearRegression().fit(
        residual_scaled,
        train_frame["away_score"].to_numpy(dtype=float) - train_frame["away_ppg"].to_numpy(dtype=float),
    )
    correct = 0
    total_error = 0.0
    blend_error = 0.0
    residual_error = 0.0
    pairs = []
    for row in test_rows:
        probability = win_probability(row, center, scale, ridge, positive_is_home, features)
        if probability is None:
            continue
        sample = ((np.array([row[name] for name in features], dtype=float) - center) / scale).reshape(1, -1)
        home = float(home_points.predict(sample)[0])
        away = float(away_points.predict(sample)[0])
        actual = float(row["home_score"]) + float(row["away_score"])
        total_error += abs((home + away) - actual)
        blend_error += abs((0.5 * (home + away) + 0.5 * (float(row["home_ppg"]) + float(row["away_ppg"]))) - actual)
        residual_sample = ((np.array([row[name] for name in features], dtype=float) - residual_center) / residual_scale).reshape(1, -1)
        residual_total = (
            float(row["home_ppg"]) + float(home_delta.predict(residual_sample)[0])
            + float(row["away_ppg"]) + float(away_delta.predict(residual_sample)[0])
        )
        residual_error += abs(residual_total - actual)
        correct += int(int(probability >= 0.5) == int(row["home_win"]))
        pairs.append({"home_win": int(row["home_win"]), "home_win_prob": probability})
    counted = len(pairs)
    if counted == 0:
        return None
    return {
        "train_rows": len(train_rows),
        "test_rows": counted,
        "accuracy": correct / counted,
        "total_mae": total_error / counted,
        "blend_mae": blend_error / counted,
        "residual_mae": residual_error / counted,
        "pairs": pairs,
    }


def choose_model(completed: list[dict]) -> dict:
    """Keep recent form only when the holdout does not get worse and no extra rows go blank."""
    global FEATURES
    base = [name for name in FEATURES if name not in RECENT_FEATURES]
    expanded = base + RECENT_FEATURES

    def countable(features: list[str]) -> int:
        return sum(
            row.get("home_win") is not None and all(row.get(name) is not None for name in features)
            for row in completed
        )

    baseline = feature_metrics(completed, base, expanded)
    candidate = feature_metrics(completed, expanded, expanded)
    adopted = False
    if baseline is not None and candidate is not None and countable(expanded) == countable(base):
        if candidate["accuracy"] + 1e-12 >= baseline["accuracy"] and candidate["total_mae"] <= baseline["total_mae"] + 0.5:
            adopted = True
    FEATURES = expanded if adopted else base
    active = candidate if adopted else baseline
    points_mode = "regression"
    if active is not None:
        options = [("regression", active["total_mae"]), ("blend", active["blend_mae"])]
        if active.get("residual_mae") is not None:
            options.append(("residual", active["residual_mae"]))
        name, value = min(options, key=lambda item: (item[1], item[0]))
        if name != "regression" and value + 0.05 < active["total_mae"]:
            points_mode = name
    return {
        "baseline": baseline,
        "recent": candidate,
        "adopted_recent": adopted,
        "use_blend": points_mode == "blend",
        "points_mode": points_mode,
        "chosen": active,
    }


def calibration_rows(rows: list[dict], split: str) -> list[dict]:
    bins = (("50-55", 0.50, 0.55), ("55-60", 0.55, 0.60), ("60-65", 0.60, 0.65), ("65-plus", 0.65, 1.01))
    out = []
    for name, low, high in bins:
        group = [
            row for row in rows
            if row.get("home_win") is not None
            and row.get("home_win_prob") is not None
            and low <= float(row["home_win_prob"]) < high
        ]
        count = len(group)
        if count == 0:
            rate = None
            calibrated = "no"
            note = "no rows"
        else:
            rate = sum(int(row["home_win"]) for row in group) / count
            calibrated = "yes" if count >= 20 and rate >= low else "no"
            if count < 20:
                note = "count under 20"
            elif rate >= low:
                note = "observed home-win rate is at least the bin floor"
            else:
                note = "observed home-win rate is below the bin floor"
        out.append({"split": split, "bin": name, "count": count, "home_win_rate": rate, "calibrated": calibrated, "note": note})
    return out


def season_team_stats(completed: list[dict], teams: list[dict]) -> pd.DataFrame:
    fbs = {team["abbreviation"]: team for team in teams}
    current = [row for row in completed if row["season"] == 2026 and row["game_type"] == "REG"]
    totals: dict[str, dict] = {}
    for row in current:
        for side, other in (("home", "away"), ("away", "home")):
            abbreviation = row[side]
            if abbreviation not in fbs:
                continue
            bucket = totals.setdefault(
                abbreviation,
                {
                    "team": abbreviation,
                    "name": fbs[abbreviation]["name"],
                    "conference": fbs[abbreviation]["conference"],
                    "games": 0,
                    "wins": 0,
                    "losses": 0,
                    "ties": 0,
                    "points_for": 0,
                    "points_against": 0,
                },
            )
            bucket["games"] += 1
            bucket["points_for"] += row[f"{side}_score"]
            bucket["points_against"] += row[f"{other}_score"]
            if row[f"{side}_score"] == row[f"{other}_score"]:
                bucket["ties"] += 1
            elif row[f"{side}_score"] > row[f"{other}_score"]:
                bucket["wins"] += 1
            else:
                bucket["losses"] += 1
    frame = pd.DataFrame(totals.values())
    if frame.empty:
        return frame
    frame["ppg"] = frame["points_for"] / frame["games"]
    frame["papg"] = frame["points_against"] / frame["games"]
    return frame.sort_values("team")


def team_abbreviation(team: dict, cache: dict[str, str]) -> str:
    abbreviation = team.get("abbreviation") or ""
    if abbreviation:
        return abbreviation
    ref = team.get("$ref") or ""
    if not ref:
        return ""
    if ref not in cache:
        payload = fetch_retry(ref)
        cache[ref] = (payload or {}).get("abbreviation") or ""
    return cache[ref]


def pull_rankings() -> tuple[pd.DataFrame, str]:
    jobs = []
    for year in (2025, 2026):
        for week in range(1, 17):
            jobs.append((year, week, RANKINGS_WEEK_URL.format(year=year, week=week)))
    indexes = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(fetch_retry, url): (year, week) for year, week, url in jobs}
        for future in as_completed(futures):
            payload = future.result()
            year, week = futures[future]
            if payload is None:
                continue
            for item in payload.get("items") or []:
                ref = item.get("$ref")
                if ref:
                    indexes.append((year, week, ref))
    polls = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(fetch_retry, ref): (year, week) for year, week, ref in indexes}
        for future in as_completed(futures):
            poll = future.result()
            year, week = futures[future]
            if not poll or poll.get("name") not in POLL_NAMES:
                continue
            polls.append((year, week, poll))
    cache: dict[str, str] = {}
    rows = []
    for year, week, poll in polls:
        occurrence = poll.get("occurrence") or {}
        poll_week = occurrence.get("number") if isinstance(occurrence, dict) else week
        for rank in poll.get("ranks") or []:
            abbreviation = team_abbreviation(rank.get("team") or {}, cache)
            if not abbreviation:
                continue
            rows.append(
                {
                    "season": year,
                    "week": int(poll_week or week),
                    "poll": poll.get("name"),
                    "date": str(poll.get("date") or "")[:10],
                    "rank": rank.get("current"),
                    "team": abbreviation,
                    "previous": rank.get("previous"),
                    "points": rank.get("points"),
                    "first_place_votes": rank.get("firstPlaceVotes"),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, "Ranking weeks were not returned."
    frame = frame.drop_duplicates(["season", "week", "poll", "team"])
    frame = frame.sort_values(["season", "week", "poll", "rank"])
    return frame, f"Ranking rows: {len(frame)}. Weeks are the occurrence on each poll."


def pull_player_stats() -> tuple[int, str]:
    payload = fetch_retry(PLAYER_URL)
    if not payload:
        return 0, "Player totals were not available."
    catalogs = {
        category["name"]: category.get("names") or []
        for category in payload.get("categories") or []
        if category.get("name")
    }
    rows = []
    for athlete in payload.get("athletes") or []:
        person = athlete.get("athlete") or {}
        teams = person.get("teams") or []
        record = {
            "player": person.get("displayName") or "",
            "team": (teams[0].get("abbreviation") if teams else "") or person.get("teamShortName") or "",
        }
        by_name = {category.get("name"): category for category in athlete.get("categories") or []}
        for category_name, names in catalogs.items():
            values = (by_name.get(category_name) or {}).get("values") or []
            for name, value in zip(names, values):
                record[f"{category_name}_{name}"] = value
        if record["player"]:
            rows.append(record)
    if not rows:
        return 0, "Player totals were not available."
    PLAYER_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(PLAYER_STATS_PATH, index=False)
    print(f"player rows {len(rows)}")
    return len(rows), "Player rows are the ESPN 2026 regular-season leader table."


def write_table(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = None
    frame[columns].to_csv(path, index=False)


def main() -> None:
    print(f"as_of {AS_OF.isoformat()} next_through {NEXT_THROUGH.isoformat()}", flush=True)
    teams = fbs_teams()
    print(f"fbs teams {len(teams)}")
    if len(teams) < 120:
        raise SystemExit("FBS standings returned too few teams")
    events, failures = collect_events(teams)
    print(f"events {len(events)} schedule failures {failures}")
    if failures > 25:
        raise SystemExit("too many schedule requests failed")
    rows = game_rows(events)
    completed = [row for row in rows if row["played"]]
    upcoming = [
        row for row in rows
        if not row["played"]
        and row["status"] != "STATUS_POSTPONED"
        and AS_OF <= date.fromisoformat(row["date"]) <= NEXT_THROUGH
    ]
    ordered = sorted(completed + upcoming, key=lambda row: (row["date"], row["event_id"], row["away"], row["home"]))
    add_prior_form(ordered)
    model_report = choose_model(completed)
    train_rows = score_rows(completed, upcoming, model_report["points_mode"])
    hold_train, hold_test, accuracy = holdout_accuracy(completed)
    player_rows, player_note = pull_player_stats()
    rankings, ranking_note = pull_rankings()
    team_stats = season_team_stats(completed, teams)

    for row in completed + upcoming:
        row["game_id"] = f"{row['date']}_{row['away']}_{row['home']}"

    TEAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(teams).to_csv(TEAMS_PATH, index=False)
    TEAM_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    team_stats.to_csv(TEAM_STATS_PATH, index=False)
    RANKINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    rankings.to_csv(RANKINGS_PATH, index=False)

    schedule_columns = [
        "date", "season", "game_type", "away", "home", "away_name", "home_name",
        "away_score", "home_score", "venue",
    ]
    future = []
    seen = {(row["date"], row["away"], row["home"]) for row in completed + upcoming}
    for row in rows:
        key = (row["date"], row["away"], row["home"])
        if key in seen or row["status"] == "STATUS_POSTPONED" or row["date"] < AS_OF.isoformat():
            continue
        seen.add(key)
        future.append(row)
    write_table(SCHEDULE_PATH, completed + upcoming + future, schedule_columns)

    completed_columns = [
        "game_id", "date", "season", "game_type", "away", "home", "away_score", "home_score", "home_win",
        *FEATURES, "home_win_prob",
    ]
    next_columns = completed_columns + [
        "predicted_away_points", "predicted_home_points", "predicted_points", "predicted_total",
        "pick_side", "pick_team", "confidence", "confidence_tier",
    ]
    write_table(COMPLETED_PATH, completed, completed_columns)
    write_table(NEXT_PATH, upcoming, next_columns)
    seeds = [
        {
            "date": row["date"],
            "away": row["away"],
            "home": row["home"],
            "home_win_prob": row.get("home_win_prob"),
            "predicted_points": row.get("predicted_points"),
            "predicted_total": row.get("predicted_total"),
            "pick_side": row.get("pick_side"),
            "pick_team": row.get("pick_team"),
            "confidence": row.get("confidence"),
            "confidence_tier": row.get("confidence_tier"),
            "spread_line": "",
            "total_line": "",
            "lean": "",
            "line": LINE,
        }
        for row in upcoming
    ]
    write_table(
        SEEDS_PATH,
        seeds,
        [
            "date", "away", "home", "home_win_prob", "predicted_points", "predicted_total",
            "pick_side", "pick_team", "confidence", "confidence_tier",
            "spread_line", "total_line", "lean", "line",
        ],
    )
    scored = [row for row in completed if row.get("home_win") is not None and row.get("home_win_prob") is not None]
    in_sample = None if not scored else sum(int(float(row["home_win_prob"]) >= 0.5) == int(row["home_win"]) for row in scored) / len(scored)
    holdout_pairs = [] if model_report.get("chosen") is None else model_report["chosen"]["pairs"]
    write_table(
        CALIBRATION_PATH,
        calibration_rows(completed, "in_sample") + calibration_rows(holdout_pairs, "holdout"),
        ["split", "bin", "count", "home_win_rate", "calibrated", "note"],
    )
    high_rows = [row for row in upcoming if row.get("confidence_tier") == "high"]
    write_table(
        HIGH_PATH,
        high_rows,
        [
            "game_id", "date", "away", "home", "home_win_prob", "pick_side", "pick_team",
            "confidence", "confidence_tier", "predicted_away_points", "predicted_home_points", "predicted_total",
        ],
    )

    max_date = max(row["date"] for row in completed)
    min_date = min(row["date"] for row in completed)
    MAX_DATE_PATH.write_text(max_date + "\n", encoding="utf-8", newline="\n")
    reg_2025 = sum(row["season"] == 2025 and row["game_type"] == "REG" for row in completed)
    post_2025 = sum(row["season"] == 2025 and row["game_type"] == "POST" for row in completed)
    reg_2026 = sum(row["season"] == 2026 and row["game_type"] == "REG" for row in completed)
    ties = sum(row["played"] and row["home_score"] == row["away_score"] for row in completed)
    zero_scores = sum(row["away_score"] == 0 and row["home_score"] == 0 for row in completed)
    scored_next = sum(row.get("home_win_prob") is not None for row in upcoming)
    today = sum(row["date"] == AS_OF.isoformat() for row in upcoming)
    accuracy_text = "not computed" if accuracy is None else f"{accuracy:.4f}"
    bowl = [
        row for row in completed
        if row["away"] == "MISS" and row["home"] == "UGA" and row["season"] == 2025 and row["game_type"] == "POST"
    ]
    bowl_text = "not found"
    if bowl:
        bowl_text = f"{bowl[0]['date']} {bowl[0]['away']} {bowl[0]['away_score']} {bowl[0]['home']} {bowl[0]['home_score']}"

    README_PATH.write_text(
        "# NCAA Football Game Predictor 2026\n\n"
        f"Completed games run from {min_date} through {max_date}. "
        "ESPN FBS standings identify the teams. ESPN team schedules are the score source "
        "for the 2025 regular season, the 2025 postseason, and the 2026 regular season. "
        "Ranking weeks come from the ESPN core API. CollegeFootballData was not used because it requires a key. "
        "A sportsbook is not called. Refresh is manual. This does not price a bet.\n\n"
        "Run `python scripts/refresh_today.py` during the season. "
        f"The script keeps results through its as-of date, {AS_OF.isoformat()}.\n\n"
        "Output files use the same seeds columns as nfl_game_predictor: "
        "date, away, home, home_win_prob, predicted_points, spread_line, total_line, lean, line. "
        "spread_line and total_line are blank. The line column says this is not a bet.\n",
        encoding="utf-8",
        newline="\n",
    )
    NOTES_PATH.write_text(
        "Run python scripts/refresh_today.py during the season. It does not run on a timer.\n\n"
        "FBS teams come from the ESPN standings tree, including division children such as the Sun Belt. "
        "Each team schedule is requested for 2025 and 2026, and again for postseason games. "
        "A final score is stored only when ESPN marks the game STATUS_FINAL, the Eastern date is on or before "
        f"{AS_OF.isoformat()}, and the score is not 0-0. Postponed games are not stored as finals. "
        f"Schedule request failures: {failures}. "
        "Games against non-FBS opponents stay on the FBS schedule. A non-FBS team's prior form includes only "
        "the games that appear on those schedules.\n\n"
        "Form uses only earlier games: win rate, points scored, points allowed, rest days, and the last four games "
        "when that check improves the holdout. "
        "A game does not train on its own score. Ties are stored and are left out of the win classifier. "
        "The published home_win_prob on a completed game is in-sample. "
        "The held-out accuracy below is a separate fit on games from "
        f"{HOLDOUT_START.isoformat()}. "
        "The win model is a ridge classifier. Points use linear regression. A half-weight of prior points "
        "per game, or a residual on top of prior points per game, is used only when that choice lowers "
        "the holdout total error. Opponent win rate is the other side's prior win rate.\n\n"
        f"{ranking_note} {player_note} "
        "The site rankings endpoint ignores the requested week and returns the latest poll, so it is not used "
        "for past weeks. There is no recorded public closing line in this table. "
        "spread_line and total_line are blank. The line column says this is not a bet. This does not price a bet.\n\n"
        f"FBS teams: {len(teams)}\n"
        f"Completed rows: {len(completed)}\n"
        f"2025 regular-season games: {reg_2025}\n"
        f"2025 postseason games: {post_2025}\n"
        f"2026 regular-season games: {reg_2026}\n"
        f"Ties: {ties}\n"
        f"Zero-score rows stored: {zero_scores}\n"
        f"Training rows: {train_rows}\n"
        f"Feature count: {len(FEATURES)}\n"
        f"Minimum date: {min_date}\n"
        f"Maximum date: {max_date}\n"
        f"Next games through {NEXT_THROUGH.isoformat()}: {len(upcoming)}\n"
        f"Next games dated {AS_OF.isoformat()}: {today}\n"
        f"Next games with home_win_prob: {scored_next}\n"
        f"Held-out games from {HOLDOUT_START.isoformat()}: {hold_test} of {hold_train} training rows, accuracy {accuracy_text}\n"
        f"In-sample accuracy: {'not computed' if in_sample is None else f'{in_sample:.4f}'} on {len(scored)} scored rows\n"
        f"High-confidence next games: {len(high_rows)}\n"
        f"{card_line('Baseline holdout', model_report.get('baseline'))}\n"
        f"{card_line('Recent-form holdout', model_report.get('recent'))}\n"
        f"Recent form adopted: {'yes' if model_report['adopted_recent'] else 'no'}\n"
        f"Points blend adopted: {'yes' if model_report['use_blend'] else 'no'}\n"
        f"Points method: {model_report['points_mode']}\n"
        f"Current-season team rows: {len(team_stats)}\n"
        f"Player rows: {player_rows}\n"
        f"Checked bowl: {bowl_text}\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"completed {len(completed)} {min_date} {max_date}")
    print(f"2025 reg {reg_2025} post {post_2025} 2026 reg {reg_2026} ties {ties}")
    print(f"next {len(upcoming)} today {today} scored {scored_next}")
    print(f"train_rows {train_rows}")
    print(f"holdout {hold_train} {hold_test} {accuracy_text}")
    print(f"teams in snapshot {len(team_stats)}")
    print(f"rankings {len(rankings)}")
    print(f"bowl {bowl_text}")
    print("recent_form", "yes" if model_report["adopted_recent"] else "no")
    print("points_blend", "yes" if model_report["use_blend"] else "no")
    print("points_method", model_report["points_mode"])
    print("features", len(FEATURES))
    print("high", len(high_rows))
    log_path = ROOT / "artifacts" / "refresh.log"
    with log_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            f"{datetime.now().isoformat(timespec='seconds')} completed {len(completed)} "
            f"max {max_date} next {len(upcoming)} recent {model_report['adopted_recent']} "
            f"blend {model_report['use_blend']} points {model_report['points_mode']}\n"
        )
    import subprocess
    import sys
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "validate_artifacts.py")], cwd=ROOT, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def card_line(label: str, block: dict | None) -> str:
    if not block:
        return f"{label}: not computed"
    return (
        f"{label}: accuracy {block['accuracy']:.4f}, total MAE {block['total_mae']:.3f}, "
        f"prior-points blend MAE {block['blend_mae']:.3f}, "
        f"residual MAE {block.get('residual_mae', 0):.3f}, "
        f"{block['test_rows']} test games after {block['train_rows']} earlier games"
    )


if __name__ == "__main__":
    main()
