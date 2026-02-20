import json
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup
import pandas as pd

# Prefer the maintained fork (thefuzz). Falls back to fuzzywuzzy if needed.
try:
    from thefuzz import process
except ImportError:
    from fuzzywuzzy import process  # type: ignore


KTC_POSITIONS = ["qb", "rb", "wr", "te"]
GITHUB_CSV_URL = "https://raw.githubusercontent.com/nzylakffa/sleepercalc/main/All%20Dynasty%20Rankings.csv"

OUTPUT_BEST_JSON = "best_values.json"
OUTPUT_WORST_JSON = "worst_values.json"
OUTPUT_LAST_UPDATED = "last_updated.txt"


def scrape_rankings(url: str) -> pd.DataFrame:
    """
    Scrapes KeepTradeCut dynasty positional rankings page.
    Returns DataFrame with columns:
    Industry Rank, Name, Rookie, Team
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; FFA-Bot/1.0; +https://thefantasyfootballadvice.com)"
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    containers = soup.find_all("div", class_="single-ranking-wrapper")

    rows = []
    for c in containers:
        rank_el = c.find("div", class_="rank-number")
        rank = rank_el.text.strip() if rank_el else ""

        name_el = c.find("div", class_="player-name")
        name = ""
        rookie = ""
        if name_el:
            a = name_el.find("a")
            if a:
                name = a.text.strip()
            if name_el.find("span", class_="rookie-badge"):
                rookie = "Rookie"

        team_el = c.find("span", class_="player-team")
        team = team_el.text.strip() if team_el else ""

        rows.append(
            {
                "Industry Rank": rank,
                "Name": name,
                "Rookie": rookie,
                "Team": team,
            }
        )

    df = pd.DataFrame(rows)

    # Normalize / clean rank
    df["Industry Rank"] = pd.to_numeric(df["Industry Rank"], errors="coerce")

    # Drop obviously broken rows
    df = df.dropna(subset=["Industry Rank"])
    df["Industry Rank"] = df["Industry Rank"].astype(int)

    return df


def fuzzy_merge(df_left: pd.DataFrame, df_right: pd.DataFrame, left_key: str, right_key: str, threshold: int = 80) -> pd.DataFrame:
    """
    Adds a matched_name column to df_left by fuzzy matching df_left[left_key] to df_right[right_key].
    """
    choices = df_right[right_key].dropna().astype(str).tolist()

    def match_one(x):
        if pd.isna(x) or str(x).strip() == "":
            return None
        match = process.extractOne(str(x), choices)
        if not match:
            return None
        name, score = match[0], match[1]
        return name if score >= threshold else None

    df_left = df_left.copy()
    df_left["matched_name"] = df_left[left_key].apply(match_one)
    return df_left


def build_merged_rankings() -> pd.DataFrame:
    """
    1) Scrape KTC for QB/RB/WR/TE
    2) Load GitHub dynasty CSV
    3) Fuzzy match and merge
    4) Return merged dataframe with core columns needed downstream
    """
    # Scrape KTC pages
    rankings_dfs = {}
    for pos in KTC_POSITIONS:
        url = f"https://keeptradecut.com/dynasty-rankings/{pos}-rankings"
        rankings_dfs[pos] = scrape_rankings(url)

    # Load your rankings
    github_df = pd.read_csv(GITHUB_CSV_URL)

    # Drop unused columns if present
    github_df = github_df.drop(columns=["TEP", "SF TEP", "SF"], errors="ignore")

    # Ensure required columns exist
    required = {"Player", "Position", "1 QB"}
    missing = required - set(github_df.columns)
    if missing:
        raise ValueError(f"GitHub CSV is missing required columns: {sorted(list(missing))}")

    # Create positional rank based on your 1 QB values (descending)
    github_df["Rank"] = github_df.groupby("Position")["1 QB"].rank(method="first", ascending=False)

    # Merge per position
    final_rankings = pd.DataFrame()

    for pos in KTC_POSITIONS:
        left = rankings_dfs[pos]

        # Fuzzy match KTC names to your Player column
        left = fuzzy_merge(left, github_df, left_key="Name", right_key="Player", threshold=80)

        merged = pd.merge(
            left,
            github_df,
            left_on="matched_name",
            right_on="Player",
            how="left",
        )

        # Drop merge helper cols
        merged = merged.drop(columns=["matched_name", "Player"], errors="ignore")

        final_rankings = pd.concat([final_rankings, merged], ignore_index=True)

    # Clean/rename columns to match your downstream logic
    final_rankings = final_rankings.drop(columns=["Team_x"], errors="ignore")
    final_rankings = final_rankings.rename(
        columns={
            "Team_y": "Team",
            "Rank": "FFA Rank",
            "Name": "Player Name",
            "1 QB": "Value",
        }
    )

    # Drop rows that didn’t successfully match
    final_rankings = final_rankings.dropna(subset=["FFA Rank"])

    # Numeric cleaning
    final_rankings["FFA Rank"] = pd.to_numeric(final_rankings["FFA Rank"], errors="coerce")
    final_rankings = final_rankings.dropna(subset=["FFA Rank"])
    final_rankings["FFA Rank"] = final_rankings["FFA Rank"].astype(int)

    # Keep columns you actually want
    keep_cols = ["Value", "Industry Rank", "FFA Rank", "Player Name", "Rookie", "Position", "Team"]
    final_rankings = final_rankings[keep_cols].copy()

    # Sort by your Value (desc)
    final_rankings = final_rankings.sort_values(by="Value", ascending=False).reset_index(drop=True)

    return final_rankings


def build_best_and_worst_json_tables(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns two DataFrames:
    - best_values_df  (positive diff, filtered like your original)
    - worst_values_df (negative diff, filtered like your original)

    Output columns are friendly for wpDataTables:
    Value, Industry Rank, FFA Rank, Diff, Player, Rookie, Pos, Team
    """
    df = merged.copy()

    # Ensure numeric ranks
    df["Industry Rank"] = pd.to_numeric(df["Industry Rank"], errors="coerce")
    df["FFA Rank"] = pd.to_numeric(df["FFA Rank"], errors="coerce")
    df = df.dropna(subset=["Industry Rank", "FFA Rank"])
    df["Industry Rank"] = df["Industry Rank"].astype(int)
    df["FFA Rank"] = df["FFA Rank"].astype(int)

    # Compute difference
    df["Diff"] = df["Industry Rank"] - df["FFA Rank"]

    # Rename for output
    df = df.rename(columns={"Player Name": "Player", "Position": "Pos"})

    # Reorder / select
    df = df[["Value", "Industry Rank", "FFA Rank", "Diff", "Player", "Rookie", "Pos", "Team"]].copy()

    # Optional: remove Joe Milton like you did
    df = df[df["Player"] != "Joe Milton"].copy()

    # Your original logic: top 150 by Value first, then chunking to find top value pockets.
    # We'll preserve the same idea but simplify into deterministic output:
    # - Best: Diff > 1, then take top 150 by Value
    # - Worst: Diff < -1, then take top 150 by Value
    #
    # If you *need* the chunk-by-chunk top-per-position behavior, tell me and I’ll match it exactly.
    best = df[df["Diff"] > 1].sort_values(by="Value", ascending=False).head(150).reset_index(drop=True)
    worst = df[df["Diff"] < -1].sort_values(by="Value", ascending=False).head(150).reset_index(drop=True)

    # Convert Value to float cleanly (wpDataTables likes consistent types)
    best["Value"] = pd.to_numeric(best["Value"], errors="coerce")
    worst["Value"] = pd.to_numeric(worst["Value"], errors="coerce")

    # Ensure Rookie is "" or "Rookie" (string)
    best["Rookie"] = best["Rookie"].fillna("").astype(str)
    worst["Rookie"] = worst["Rookie"].fillna("").astype(str)

    # Ensure Pos/Team/Player are strings
    for col in ["Player", "Pos", "Team"]:
        best[col] = best[col].fillna("").astype(str)
        worst[col] = worst[col].fillna("").astype(str)

    return best, worst


def write_outputs(best_df: pd.DataFrame, worst_df: pd.DataFrame) -> None:
    # JSON: array of objects (records) for wpDataTables
    best_df.to_json(OUTPUT_BEST_JSON, orient="records", indent=2)
    worst_df.to_json(OUTPUT_WORST_JSON, orient="records", indent=2)

    # Timestamp (ET label for humans; job runs in UTC)
    now_utc = datetime.now(timezone.utc)
    stamp = now_utc.strftime("%Y-%m-%d %H:%M UTC")

    with open(OUTPUT_LAST_UPDATED, "w", encoding="utf-8") as f:
        f.write(stamp + "\n")


def main():
    merged = build_merged_rankings()
    best_df, worst_df = build_best_and_worst_json_tables(merged)
    write_outputs(best_df, worst_df)


if __name__ == "__main__":
    main()
