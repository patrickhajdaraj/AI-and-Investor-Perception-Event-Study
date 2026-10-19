"""

PART 2: DATA PREPARATION

Purpose:
	Prepares cached Reuters AI news for content analysis and exclude candidates with 
	confounding headlines within the NYSE T-1 through T+5 event window

Contained steps:
    	- Excludes predefined Reuters news categories
    	- Removes duplicate Reuters stories and search results
    	- Verifies historical S&P 500 membership at the time of each announcement
    	- Extracts and cleans available Reuters full-story texts
   	- Identifies and excludes AI-related headlines from confounding event candidates
    	- Excludes AI events affected by non-AI confounding announcements within the NYSE trading window T-1 through T+5
    	- Creates a standardized dataset of eligible AI disclosure events

Input: 
	- SPX_AI_DISCLOSURES_CACHE.pkl and SPX_CONFOUNDING_EVENTS_CACHE.pkl created in part 1

Output: 
	- Cached cleaned AI disclosure events without confounding events within T-1 and T+5 (SPX_AI_EVENTS_PREPARED.pkl) 
	- Corresponding Excel export of cleaned AI disclosure events (SPX_AI_EVENTS_PREPARED.xlsx)

Important notes:
	- This query fully relies on previous created caches, therefore no LSEG session is required

"""

###############################################################################################################
# PREPARATIONS
###############################################################################################################

# IMPORT OF REQUIRED LIBRARIES

import lseg.data as ld
import pandas as pd
import pickle
import time
from pathlib import Path
from lseg.data.content import news
import pandas_market_calendars as mcal
import re
from bs4 import BeautifulSoup

# DEFINITION OF FILES

AI_EVENTS_CACHE = Path("SPX_AI_DISCLOSURES_CACHE.pkl")
CONFOUNDING_EVENTS_CACHE = Path("SPX_CONFOUNDING_EVENTS_CACHE.pkl")
CLEANED_AI_EVENTS = Path("SPX_AI_EVENTS_CLEANED.xlsx")
CLEANED_CONFOUNDING_EVENTS = Path("SPX_CONFOUNDING_EVENTS_CLEANED.xlsx")
OUTPUT_CACHE = Path("SPX_AI_EVENTS_PREPARED.pkl")
OUTPUT_EXCEL = Path("SPX_AI_EVENTS_PREPARED.xlsx")

###############################################################################################################
# CLEANING OF AI EVENTS AND CONFOUNDING EVENTS CACHE
###############################################################################################################

# DEFINITION OF REUTERS NEWS CATEGORIES TO EXCLUDE IN ANALYSIS 

# Reuters news categories, which suggest general news and economic outlook are excluded

EXCLUDED_REUTERS_PATTERNS = {
    "Newscasts": r"^NEWSCASTS?\s*[-:\u2013\u2014]",
    "LIVE MARKETS": r"^LIVE\s+MARKETS\s*[-:\u2013\u2014]",
    "GRAPHIC": r"^GRAPHIC(?:\s*[-:\u2013\u2014]|\s+)",
    "US STOCKS": r"^US\s+STOCKS\s*[-:\u2013\u2014]",
    "RPT": r"^RPT(?:\s*[-:\u2013\u2014]|\s+\d+\s*[-:\u2013\u2014])",
    "GLOBAL MARKETS": r"^GLOBAL\s+MARKETS\s*[-:\u2013\u2014]",
    "QUOTES": r"^QUOTES\s*[-:\u2013\u2014]",
    "REFILE": r"^REFILE\s*[-:\u2013\u2014]",
    "BUZZ": r"^BUZZ\s*[-:\u2013\u2014]",
    "MEDIA": r"^MEDIA\s*[-:\u2013\u2014]",
    "Wall St Week Ahead": r"^WALL\s+ST\s+WEEK\s+AHEAD\b",
    "EMERGING MARKETS": r"^EMERGING\s+MARKETS\s*[-:\u2013\u2014]",
    "PREVIEW": r"^PREVIEW\s*[-:\u2013\u2014]",
    "London stocks": r"^LONDON\s+STOCKS\s*[-:\u2013\u2014]",
    "European shares": r"^EUROPEAN\s+SHARES\s*[-:\u2013\u2014]",
    "Earnings Preview": r"^.*[-:\u2013\u2014]\s*EARNINGS\s+PREVIEW\s*$",
    "FACTBOX": r"^FACTBOX(?:\s*[-:\u2013\u2014]|\s+)",
    "BREAKINGVIEWS": r"^BREAKINGVIEWS\s*[-:\u2013\u2014]",
    "ANALYSIS": r"^ANALYSIS\s*[-:\u2013\u2014]",
    "COLUMN": r"^COLUMN\s*[-:\u2013\u2014]",
    "ROI": r"^ROI\s*[-:\u2013\u2014]",
    "EXPLAINER": r"^EXPLAINER\s*[-:\u2013\u2014]",}

# DEFINITION OF AI KEYWORD WHITELIST FOR CONFOUNDING EVENT FILTER

# Checks that actual AI events wont trigger confounding events (e.g. AI product launches or Acquisition of AI companies etc.)
# AI_WHITELIST consists of the AI Keywords used to retrieve AI disclosures in part 1

AI_WHITELIST = [
    # AI Dictionary based on Uberti-Bona Marin et al. (2025)
    "artificial intelligence",
    "machine learning",
    "generative",
    "voice assistant",
    "A.I",
    "artificial general intelligence",
    "deepfake",
    "deep learning",
    "image recognition",
    "chatbot",
    "NLP",
    "natural language processing",
    "computer vision",
    "speech recognition",
    "recommendation system",
    "recommender system",

    # Generative AI Dictionary based on Dogan & Kebede Teklie (2026)
    "generative AI",
    "large language model",
    "LLM",
    "foundation model",
    "ChatGPT",
    "GPT",
    "transformer model",
    "GenAI",
    "text generation",
    "image generation",

    # Own addition
    "AI"
]

def identify_excluded_reuters_category(headline):
    if not isinstance(headline, str):
        return None

    return next((
        name for name, pattern in EXCLUDED_REUTERS_PATTERNS.items()
        if re.match(pattern, headline.strip(), flags=re.IGNORECASE)
    ), None)


def contains_ai_keyword(headline):
    if not isinstance(headline, str):
        return False

    return any(re.search(
        r"(?<!\w)" + re.escape(keyword) + r"(?!\w)",
        headline, flags=re.IGNORECASE

    ) for keyword in AI_WHITELIST)


def extract_reuters_text(value):
    if isinstance(value, list):
        return " ".join(extract_reuters_text(item) for item in value).strip()

    if isinstance(value, dict):
        return extract_reuters_text(value.get("$", ""))

    if isinstance(value, str):
        return BeautifulSoup(value, "html.parser").get_text(" ", strip=True)

    return ""

# DEFINITION OF HEADLINE-BASED FIRM RELEVANCE FILTER

# Identifies explicit company references in Reuters headlines
# Uses full RICs, tickers and distinctive company name components
# Excludes generic legal suffixes and company descriptors
# No manual company alias dictionary is required

GENERIC_COMPANY_TERMS = {
    "inc", 
    "incorporated", 
    "corp", 
    "corporation",
    "company", 
    "co", 
    "ltd", 
    "limited", 
    "plc",
    "llc", 
    "holdings", 
    "holding", 
    "group",
    "the", 
    "and", 
    "of",
    "class",
    "technologies",
    "technology", 
    "systems",
    "services", 
    "international", 
    "global",
    "industries", 
    "enterprises", 
    "solutions",
    "financial", 
    "general", 
    "american",
    "united",
    "energy", 
    "power"
}


def identify_firm_relevance(headline, ric, company_name):

    if not isinstance(headline, str) or not headline.strip():
        return None

    # Check full company RIC

    if isinstance(ric, str) and ric.strip():
        ric = ric.strip()

        if re.search(
            r"(?<!\w)" + re.escape(ric) + r"(?!\w)",
            headline,
            flags=re.IGNORECASE
        ):
            return f"RIC: {ric}"

        # Check ticker without exchange suffix

        ticker = ric.split(".")[0]

        if len(ticker) >= 4 and re.search(
            r"(?<!\w)" + re.escape(ticker) + r"(?!\w)",
            headline
        ):
            return f"Ticker: {ticker}"

    # Check distinctive company name components

    if isinstance(company_name, str):

        name_parts = re.findall(
            r"[A-Za-z0-9]+",
            company_name
        )

        for part in name_parts:

            if (
                len(part) >= 4
                and part.lower() not in GENERIC_COMPANY_TERMS
            ):

                if re.search(
                    r"(?<!\w)" + re.escape(part) + r"(?!\w)",
                    headline,
                    flags=re.IGNORECASE
                ):
                    return f"Name: {part}"

    return None


# LOAD CACHES

# Load previouly created AI disclosure event and confounding event cache

for file in (AI_EVENTS_CACHE, CONFOUNDING_EVENTS_CACHE):
    if not file.exists():
        raise FileNotFoundError("Request cache not found: {}".format(file))

with AI_EVENTS_CACHE.open("rb") as f:
    ai_cache = pickle.load(f)

with CONFOUNDING_EVENTS_CACHE.open("rb") as f:
    confounding_events = pickle.load(f)["events"].copy()

spx_cache = ai_cache["spx"]
company_cache = ai_cache["companies"]
sector_cache = ai_cache["sectors"]
story_cache = ai_cache["stories"]

# FILTER CONFOUNDING NEWS

confounding_events["Excluded Reuters Category"] = (
    confounding_events["Headline / Document Title"]
    .apply(identify_excluded_reuters_category)
)

reuters_source = (
    confounding_events["Source Type"].fillna("")
    .astype(str).str.strip().str.lower().eq("reuters")
)

excluded_reuters_mask = (
    reuters_source & confounding_events["Excluded Reuters Category"].notna()
)

excluded_confounding_events = confounding_events.loc[excluded_reuters_mask].copy()

print("Confounding observations excluded by Reuters category:", len(excluded_confounding_events))

print(excluded_confounding_events["Excluded Reuters Category"].value_counts())

confounding_events = confounding_events.loc[~excluded_reuters_mask].copy()

print("Confounding observations after category filter:", len(confounding_events))

###############################################################################################################
# COMPARE AI EVENTS WITH CONFONFOUNDING EVENTS 
###############################################################################################################

# Remove ALL confounding candidates with an AI keyword in the Headline

ai_confounding_mask = (
    confounding_events["Headline / Document Title"].apply(contains_ai_keyword)
)

excluded_ai_confounding_events = confounding_events.loc[ai_confounding_mask].copy()

confounding_events = confounding_events.loc[~ai_confounding_mask].copy()

print("AI-related news excluded from confounding events:", len(excluded_ai_confounding_events))

print("Confounding observations remaining:", len(confounding_events))

# FILTER CONFOUNDING EVENTS BY FIRM RELEVANCE

# Retains only confounding news explicitly referencing
# the corresponding company in the Reuters headline

confounding_events["Firm Relevance Match"] = (
    confounding_events.apply(
        lambda row: identify_firm_relevance(
            row["Headline / Document Title"],
            row["RIC"],
            row["Company Name"]
        ),
        axis=1
    )
)

confounding_relevance_mask = (
    confounding_events["Firm Relevance Match"].notna()
)

print(
    "Confounding observations excluded by firm relevance:",
    (~confounding_relevance_mask).sum()
)

confounding_events = (
    confounding_events.loc[confounding_relevance_mask]
    .copy()
    .reset_index(drop=True)
)

print(
    "Confounding observations after firm relevance filter:",
    len(confounding_events)
)

# PREPARE CONFOUNDING EVENT TIMESTAMPS

if confounding_events[["RIC", "Date", "Time"]].isna().any().any():
    raise RuntimeError("Confounding events contain missing RICs, dates or times.")

confounding_events["Confounding Timestamp UTC"] = pd.to_datetime(
    confounding_events["Date"].astype(str).str.strip() + " "
    + confounding_events["Time"].astype(str).str.strip(),
    format="%Y-%m-%d %H:%M:%S", utc=True, errors="raise",
)

if confounding_events["Confounding Timestamp UTC"].isna().any():
    raise RuntimeError("Confounding events contain missing timestamps.")

print("Confounding events loaded:", len(confounding_events))

# EXPORT CLEANED CONFOUNDING EVENTS

# Preserves the exact column structure and sorting of the Part 1 export
# Only confounding observations passing the Part 2 filters are exported

CONFOUNDING_EXPORT_COLUMNS = [
    "RIC",
    "Company Name",
    "Date",
    "Time",
    "Confounding Event",
    "Source Type",
    "Headline / Document Title",
    "storyId"
]

confounding_export = (
    confounding_events[CONFOUNDING_EXPORT_COLUMNS]
    .sort_values(["RIC", "Date", "Time", "Source Type"])
    .reset_index(drop=True)
)

confounding_export.to_excel(
    CLEANED_CONFOUNDING_EVENTS,
    sheet_name="Confounding Events",
    index=False
)

print("Cleaned confounding events exported:", CLEANED_CONFOUNDING_EVENTS)

# COMBINE CACHED HEADLINES

frames = []

for cached in ai_cache["headlines"].values():
    if cached is None:
        continue
    for frame in ([cached] if isinstance(cached, pd.DataFrame) else cached):
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            frames.append(frame.copy())
if not frames:
    raise RuntimeError("No cached headline data found.")

headline_df = pd.concat(frames, ignore_index=True)
print("\nRaw headline observations:", len(headline_df))

# PREPARE DATES AND REMOVE DUPLICATE SEARCH HITS

headline_df["versionCreated"] = pd.to_datetime(
    headline_df["versionCreated"], utc=True, errors="raise"
)
headline_df["Date"] = headline_df["versionCreated"].dt.strftime("%Y-%m-%d")
headline_df["Time"] = headline_df["versionCreated"].dt.strftime("%H:%M:%S")
headline_df = headline_df.drop_duplicates(
    subset=["storyId", "versionCreated", "SearchRIC"]
).reset_index(drop=True)

print("After duplicate search hit removal:", len(headline_df))

# FILTER MONTHLY S&P 500 MEMBERSHIP

headline_df["MembershipDate"] = (
    headline_df["versionCreated"].dt.tz_convert("America/New_York")
    .dt.strftime("%Y-%m-01")
)

missing_snapshots = set(headline_df["MembershipDate"]) - set(spx_cache)
if missing_snapshots:

    raise RuntimeError("Missing SPX snapshot: {}".format(sorted(missing_snapshots)[0]))
spx_member = [
    ric in spx_cache[month]
    for ric, month in zip(headline_df["SearchRIC"], headline_df["MembershipDate"])
]

events = headline_df.loc[spx_member].copy()
print("After SPX membership filter:", len(events))

# FILTER EXCLUDED REUTERS AI NEWS CATEGORIES

events["Excluded Reuters Category"] = (
    events["headline"].apply(identify_excluded_reuters_category)
)

excluded_ai_events = events.loc[events["Excluded Reuters Category"].notna()].copy()

print("AI observations excluded by Reuters category:", len(excluded_ai_events))

print(excluded_ai_events["Excluded Reuters Category"].value_counts())

events = events.loc[events["Excluded Reuters Category"].isna()].copy()

print("AI observations after Reuters category filter:", len(events))

# REMOVE DUPLICATE REUTERS STORIES FOR EACH COMPANY

n_before = len(events)

valid_story_id = events["storyId"].notna() & events["storyId"].astype(str).str.strip().ne("")

events = pd.concat([
    events.loc[valid_story_id].sort_values("versionCreated").drop_duplicates(
        subset=["SearchRIC", "storyId"], keep="first"
    ),

    events.loc[~valid_story_id],

], ignore_index=True).sort_values("versionCreated").reset_index(drop=True)

print("Duplicate Reuters story observations removed:", n_before - len(events))

# ADD CACHED REUTERS STORY DATA AND FIRM METADATA

if not events["storyId"].isin(story_cache).all():
    raise RuntimeError("Some Reuters stories are missing from the cache.")

events["Associated RICs"] = events["storyId"].map(
    lambda sid: story_cache.get(sid, {}).get("associated_rics", [])
)

events["Story Text"] = events["storyId"].map(
    lambda sid: extract_reuters_text(story_cache.get(sid, {}).get("story_text", ""))
)

events["Full Story Available"] = events["Story Text"].str.strip().ne("")

events["Relevant RIC"] = events["SearchRIC"]

events["Company Name"] = events["Relevant RIC"].map(company_cache)

events["Sector"] = events["Relevant RIC"].map(sector_cache)

events["Source"] = "Reuters"

events = events.rename(columns={"headline": "Headline"})

# FILTER AI DISCLOSURE EVENTS BY FIRM RELEVANCE

# Retains only AI news explicitly referencing
# the corresponding company in the Reuters headline

events["Firm Relevance Match"] = (
    events.apply(
        lambda row: identify_firm_relevance(
            row["Headline"],
            row["Relevant RIC"],
            row["Company Name"]
        ),
        axis=1
    )
)

ai_relevance_mask = (
    events["Firm Relevance Match"].notna()
)

print(
    "AI observations excluded by firm relevance:",
    (~ai_relevance_mask).sum()
)

events = (
    events.loc[ai_relevance_mask]
    .copy()
    .reset_index(drop=True)
)

print(
    "AI observations after firm relevance filter:",
    len(events)
)

# EXPORT CLEANED AI EVENTS

# Preserves the exact column structure and sorting of the Part 1 export
# Only AI observations passing the Part 2 filters are exported

AI_EXPORT_COLUMNS = [
    "SearchRIC",
    "Company Name",
    "Sector",
    "versionCreated",
    "headline",
    "storyId",
    "sourceCode"
]

ai_export = (
    events.rename(columns={"Headline": "headline"})
    [AI_EXPORT_COLUMNS]
    .sort_values(["SearchRIC", "versionCreated"])
    .reset_index(drop=True)
    .copy()
)

# Convert UTC timestamps using the original Part 1 format

ai_export["versionCreated"] = (
    pd.to_datetime(ai_export["versionCreated"], utc=True)
    .dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
)

ai_export.to_excel(
    CLEANED_AI_EVENTS,
    sheet_name="AI Events",
    index=False
)

print("Cleaned AI events exported:", CLEANED_AI_EVENTS)

# PREPARE NYSE TRADING CALENDAR

ai_timestamps = pd.to_datetime(events["versionCreated"], utc=True, errors="raise")
calendar = mcal.get_calendar("NYSE")
schedule = calendar.schedule(
    start_date=(ai_timestamps.min() - pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
    end_date=(ai_timestamps.max() + pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
)

market_closes = pd.DatetimeIndex(pd.to_datetime(schedule["market_close"], utc=True))

# CHECK FOR CONFOUNDING EVENTS WITHIN EVENT WINDOW

# Preserve the original T-1 through T+5 window and same-story exemption

confounding_events_by_ric = {
    ric: group[["Confounding Timestamp UTC", "storyId"]]
    .drop_duplicates().sort_values("Confounding Timestamp UTC").reset_index(drop=True)
    for ric, group in confounding_events.groupby("RIC")
}


def has_confounding_event(row):
    event_timestamp = pd.to_datetime(row["versionCreated"], utc=True, errors="raise")
    event_position = market_closes.searchsorted(event_timestamp, side="right")

    if event_position < 2 or event_position + 5 >= len(market_closes):
        raise RuntimeError("Trading calendar does not cover the event window.")

    window_start = market_closes[event_position - 2]
    window_end = market_closes[event_position + 5]
    firm_confounds = confounding_events_by_ric.get(row["Relevant RIC"])

    if firm_confounds is None or firm_confounds.empty:
        return False
    return bool((
        (firm_confounds["Confounding Timestamp UTC"] >= window_start)
        & (firm_confounds["Confounding Timestamp UTC"] <= window_end)
        & (firm_confounds["storyId"] != row["storyId"])
    ).any())


# EXCLUDE AI EVENTS WITH CONFOUNDING EVENTS

confounding_mask = events.apply(has_confounding_event, axis=1)

events = events.loc[~confounding_mask].copy().reset_index(drop=True)

print("AI observations excluded due to confounding events:", confounding_mask.sum())

print("AI observations remaining after confounding filter:", len(events))

# EXCLUDE STOCK PRICE REPORTS AND ANALYST NEWS

# Excludes AI event candidates containing share or analyst references
# Applies only to the final prepared AI event dataset
# Cleaned AI and confounding event exports remain unchanged

EXCLUDED_NEWS_KEYWORDS = [
    "shares",
    "analyst",
    "analysts",
    "stock",
    "stocks",
    "valuation",
    "earnings",
    "market value"
]

excluded_news_pattern = (
    r"\b(?:"
    + "|".join(re.escape(keyword) for keyword in EXCLUDED_NEWS_KEYWORDS)
    + r")\b"
)

excluded_news_mask = events["Headline"].str.contains(
    excluded_news_pattern,
    case=False,
    regex=True,
    na=False
)

print(
    "AI observations excluded by share/analyst keywords:",
    excluded_news_mask.sum()
)

events = (
    events.loc[~excluded_news_mask]
    .copy()
    .reset_index(drop=True)
)

print(
    "AI observations after share/analyst filter:",
    len(events)
)

# CREATE PREPARED AI EVENTS DATASET

events["Event_ID"] = ["E-{}".format(i) for i in range(1, len(events) + 1)]

prepared_events = events[[
    "Event_ID", 
    "Date", 
    "Time", 
    "Source", 
    "Relevant RIC",
    "Company Name",
    "Sector", 
    "Associated RICs", 
    "Headline", 
    "storyId", 
    "Story Text", 
    "Full Story Available",
    "versionCreated"
]].copy()

# SAVE PICKLE CACHE

with OUTPUT_CACHE.open("wb") as f:
    pickle.dump(prepared_events, f, protocol=pickle.HIGHEST_PROTOCOL)

# EXPORT EXCEL

excel_events = prepared_events.copy()

# Convert UTC timestamps to strings for Excel compatibility
excel_events["versionCreated"] = (
    excel_events["versionCreated"]
    .dt.strftime("%Y-%m-%d %H:%M:%S.%f UTC")
)

excel_events["Associated RICs"] = excel_events["Associated RICs"].apply(
    lambda value: repr(list(value)) if isinstance(value, (list, tuple, set)) else value
)

long_text_count = excel_events["Story Text"].fillna("").str.len().gt(32767).sum()

if long_text_count:
    print("WARNING | {} full stories exceed Excel's 32,767-character cell limit.".format(long_text_count))
    excel_events["Story Text"] = excel_events["Story Text"].fillna("").str.slice(0, 32767)

with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
    excel_events.to_excel(writer, sheet_name="AI Event Candidates", index=False)
    worksheet = writer.sheets["AI Event Candidates"]
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions

# DISPLAY FINAL STATUS 

# Signals that query ran successfully

print("\nDATA PREPARATION COMPLETE")
print("Prepared firm-story observations:", len(prepared_events))
print("Unique Reuters stories:", prepared_events["storyId"].nunique())
print("Unique companies:", prepared_events["Relevant RIC"].nunique())
print("Observations with full story text:", prepared_events["Full Story Available"].sum())
print("Observations without full story text:", (~prepared_events["Full Story Available"]).sum())
print("Prepared dataset saved as:", OUTPUT_CACHE)
print("Prepared Excel dataset saved as:", OUTPUT_EXCEL)
