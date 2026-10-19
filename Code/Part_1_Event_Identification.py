"""

PART 1: EVENT IDENTIFICATION AND EVENT DATA RETRIEVAL

Purpose:
	Retrieves and caches the data required for the identification of AI-related events and potential confounding events

Contained steps:
	- Monthly S&P 500 constituent snapshots on the first day of each month during the observation period
	- The union of all identified RICs defines the study sample
	- Reuters news headlines and corresponding full stories identified through company RICs and predefined AI keywords
	- Reuters news indicating potential confounding events within the defined event Windows for each company, 
	  from the previous NYSE market close T-1 through the market close on T+5

Input: 
	- No inputs required

Output: 
	- Cached S&P 500 constituent and company metadata
	- Cached Reuters AI event candidate headlines and full stories (SPX_AI_DISCLOSURES_CACHE.pkl)
	- Cached Reuters confounding Event candidates (SPX_CONFOUNDING_EVENTS_CACHE.pkl)
	- Excel exports of raw AI and confounding event candidates (SPX_AI_DISCLOSURE_EVENTS.xlsx and SPX_CONFOUNDING_EVENTS.xlsx)

Important notes:
	- This query is designed for LSEG CodeBook, the Jupyter-based Python environment integrated into LSEG Workspace
	- This query can also be executed in other Python environments, such as Visual Studio Code or a local Jupyter Notebook, provided that
  	  the LSEG Data Library is installed and a compatible session is configured
	- Therefore, access to LSEG data is required
 
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

# OPENING OF LSEG SESSION

ld.open_session()

# DEFINITION OF OBSERVATION PERIOD FOR ANALYSIS

START_DAY = "2025-09-01"
END_DAY   = "2026-08-31"

START_DATE = f"{START_DAY}T00:00:00"
END_DATE   = f"{END_DAY}T23:59:59"

# CREATION OF CACHES

# Caches are used to store retrieved raw LSEG data, reducing redundant API requests and improving overall performance throughout the analysis
# Cache file SPX_AI_DISCLOSURES_CACHE saves S&P500 member RICs, company names, sectors, Reuters news headlines and corresponding full articles
# Cache file SPX_CONFOUNDING_EVENTS_CACHE saves results of the Reuters confounding searches

AI_EVENTS_CACHE = Path("SPX_AI_DISCLOSURES_CACHE.pkl")

CONFOUNDING_EVENTS_CACHE = Path("SPX_CONFOUNDING_EVENTS_CACHE.pkl")

def load_cache(path, default):
    if path.exists():
        with path.open("rb") as f:
            return pickle.load(f)
    return default

# Define AI events cache

cache = load_cache(
    AI_EVENTS_CACHE,
    {
        "spx": {},
        "companies": {},
        "sectors": {},
        "headlines": {},
        "stories": {}
    }
)

spx_cache = cache.setdefault("spx", {})
company_cache = cache.setdefault("companies", {})
sector_cache = cache.setdefault("sectors", {})
headline_cache = cache.setdefault("headlines", {})
story_cache = cache.setdefault("stories", {})

# Define disclosure events Cache

confounding_data = load_cache(
    CONFOUNDING_EVENTS_CACHE,
    {"searches": {}}
)

confounding_cache = confounding_data.setdefault("searches", {})

# SAVING OF BOTH CACHES 

# Combined save_cache function which saves both AI events and disclosure events cache when adressed
# Combined save function is incorporated in every following query for data retrieval to ensure no data gets lost

def save_cache():
    with AI_EVENTS_CACHE.open("wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    with CONFOUNDING_EVENTS_CACHE.open("wb") as f:
        pickle.dump(
            confounding_data,
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )

# SAFE REQUEST

# The safe_request function prevents the entire query from terminating when a single LSEG request fails due to rate limits or server errors
# The safe_request function iterates through defined waiting periods (5, 15, 30 and 60 seconds) for request and retries failed requests after each interval
# If all retry attempts fail, the current status is saved in corresponding caches before the final error is raised
# The safe_request function is embedded in every query for data retrieval to ensure data quality

def safe_request(
    request_query,
    label="LSEG request",
    waiting_time=None
):
    if waiting_time is None:
        waiting_time = [5, 15, 30, 60]
    for attempt in range(
        len(waiting_time) + 1
    ):
        try:
            result = request_query()
            time.sleep(0.4)
            return result

        except Exception as e:
            if attempt == len(waiting_time):
                save_cache()
                print(
                    f"FAILED | "
                    f"{label} | "
                    f"{type(e).__name__}: {e}"
                )
                raise
            wait = waiting_time[attempt]
            print(
                f"RETRY | "
                f"{label} | "
                f"waiting {wait}s"
            )
            time.sleep(wait)

# NORMALIZATION OF RICs

# Initial testing showed that RICs may be returned with historical or Nasdaq-specific suffixes 
# RICs are therefore normalized via function normalize_ric to ensure compatibility with Reuters news searches
# Normalize_ric function is embedded in corresponding queries

def normalize_ric(ric):
    if ric is None:
        return None
    
    ric = str(ric).strip()

    if not ric:
        return None

    ric = ric.split("^")[0]

    if ric.endswith(".OQ"):
        ric = ric[:-1]
    return ric

###############################################################################################################
# DETERMINATION OF STUDY SAMPLE AND RETRIEVAL OF CORRESPONDING COMPANY METADATA
###############################################################################################################

# CREATION OF MONTHLY S&P 500 SNAPSHOTS

# Creates a snapshot of all S&P 500 constituents on the first day of every month within observation period
# Identifies all companies included in the S&P 500 Index at the beginning of each month during the observation period
# Snapshots are based on predefined start and end days
# RICs are based on the field IndexConstituentRIC from lseg.data library

snapshot_dates = pd.date_range(
    start=START_DAY,
    end=END_DAY,
    freq="MS"
)

print(
    f"Loading "
    f"{len(snapshot_dates)} "
    f"monthly SPX snapshots"
)

for i, date in enumerate(
    snapshot_dates,
    start=1
):

# Conversion of dates to string to use it as parameter in subsequent query parts

    date_str = date.strftime(
        "%Y-%m-%d"
    )

# Checks, if RICs have already been identified and cached in previous runs

    if date_str in spx_cache:
        print(
            f"{i}/{len(snapshot_dates)} | "
            f"{date_str} | Cached"
        )

        continue

    print(
        f"{i}/{len(snapshot_dates)} | "
        f"{date_str} | Requesting"
    )

    df = safe_request(
        lambda date_str=date_str:
            ld.get_data(
                universe=".SPX",
                fields=[
                    "TR.IndexConstituentRIC"
                ],
                parameters={
                    "SDate": date_str,
                    "EDate": date_str
                }
            ),

        label=f"SPX {date_str}"
    )

# Search for column containing information regarding corresponding RICs

    ric_columns = [
        col
        for col in df.columns
        if (
            "RIC" in str(col).upper()
            and
            str(col).upper()
            != "INSTRUMENT"
        )
    ]

    if not ric_columns:
        raise ValueError(
            f"No RIC column found "
            f"for {date_str}"
        )

    ric_column = ric_columns[0]

# Normalization based on normalize_ric function and deduplication of retrieved RICs

    members = set(
        normalize_ric(x)
        for x in (
            df[
                ric_column
            ]
            .dropna()
        )
    )

    members.discard(None)

# Distinct list of retrieved RICs is saved in cache spx_cache

    spx_cache[
        date_str
    ] = members

    save_cache()

# UNION ALL IDENTIFIED S&P 500 RICs

# Union of all RICs retrieved in previous step
# Representing all distinct RICs of companies that were part of the S&P 500 index on the first day of each month within observation period
# Distinct list of identified RICs is also saved in variable rics_spx for dynamic use

rics_spx = sorted(
    set().union(
        *spx_cache.values()
    )
)

print(
    f"\nNumber of distinct SPX members: "
    f"{len(rics_spx)} RICs"
)

# ADD NAMES TO IDENTIFIED RICs

# Retrieves the companies names for the previously identified RICs of relevant S&P500 constituents
# Company names are based on the field TR.CommonName from lseg.data library

# Compare list of rics with rics associated with a company name to see, if there are already company names retrieved from previous runs
# In inital run, all RICs require to get a company name assigned and appear in variable rics_to_add_company_name

rics_to_add_company_name = [ric for ric in rics_spx if ric not in company_cache]

# Retrieval of company names

if rics_to_add_company_name:
    print(
        f"Loading company names for "
        f"{len(rics_to_add_company_name)} RICs"
    )
    company_df = safe_request(
        lambda:
            ld.get_data(
                universe=rics_to_add_company_name,
                fields=[
                    "TR.CommonName"
                ]
            ),
        label="Company names"
    )
    for _, row in company_df.iterrows():
        ric = normalize_ric(
            row.iloc[0]
        )
        company_name = row.iloc[1]
        if ric is not None:
            company_cache[ric] = company_name
    save_cache()

print(
    f"Company names cached: "
    f"{len(company_cache)}"
)

# ADD SECTORS TO IDENTIFIED RICs

# Retrieves the economical sector corresponding to the previously identified distinct RICs of S&P500 constituents
# Sectors are based on the field TR.TRBCEconomicSector from lseg.data library

# Compare list of rics with rics associated with a company name to see, if there are already sectors retrieved from previous runs
# In inital run, all RICs require to get a company name assigned and appear in variable rics_to_add_sector

rics_to_add_sector = [ric for ric in rics_spx if pd.isna(sector_cache.get(ric)) or not str(sector_cache.get(ric)).strip()]

if rics_to_add_sector:
    print(
        f"Loading company sectors for "
        f"{len(rics_to_add_sector)} RICs"
    )
    sector_df = safe_request(
        lambda:
            ld.get_data(
                universe=rics_to_add_sector,
                fields=[
                    "TR.TRBCEconomicSector"
                ]
            ),
        label="Company sectors"
    )

    for _, row in sector_df.iterrows():

        ric = normalize_ric(
            row.iloc[0]
        )
        sector = row.iloc[1]

        if (
            ric is not None
            and pd.notna(sector)
            and str(sector).strip()
        ):
            sector_cache[ric] = str(sector).strip()

    save_cache()

print(
    f"Company sectors cached: "
    f"{len(sector_cache)}"
)

###############################################################################################################
# RETRIEVAL OF AI DISCLOSURE EVENTS
###############################################################################################################

# DEFINITION OF AI KEYWORDS

# Definition of AI Keywords dictionary for event identification 
# Keywords are adapted from the AI Dictionary of Uberti-Bona Marin et al. (2026) and generative AI Dictionary of Dogan & Teklie (2026) 
# Keyword "AGI" left out intentionally from Uberti-Bona Marin et al. (2026) due to false positives
# Keyword "AI" was added intentionally
# Combines AI keywords via function build_ai_keyword_query for dynamic use in request query for Reuters news headlines

ai_keywords = [

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

# Combines all AI keywords into an OR-based Reuters search expression

def build_ai_keyword_query(ai_keywords):
    return "(" + " OR ".join(f'"{word}"' for word in ai_keywords) + ")"

# REUTERS HEADLINE REQUEST
 
# Defines how the LSEG Headline request is conducted
# Searches Reuters headlines and full stories using RICs and the AI keyword Dictionary defined in prior step
# Splits date ranges recursively when the 100-result limit is reached

def request_ai_headlines(ric, start_date, end_date):

    query = (
        f"R:{ric} "
        f"AND Language:LEN "
        f"AND Source:RTRS "
        f"AND {build_ai_keyword_query(ai_keywords)} "
        f"AND SearchIn:HeadlineOnly"
    )

    response = safe_request(
            lambda: ld.delivery.endpoint_request.Definition(
            url="/data/news/v1/headlines",
            method=ld.delivery.endpoint_request.RequestMethod.GET,
            query_parameters={
                "query": query,
                "dateFrom": start_date,
                "dateTo": end_date,
                "limit": 100,
                "sort": "oldToNew"
            }
        ).get_data(),
        label=f"Headlines {ric}"
    )

# Structuring raw LSEG data into one Pandas DataFrame

    raw = response.data.raw

    if not isinstance(raw, dict):
        raise RuntimeError(f"Unexpected Reuters response for {ric}")

    rows = []

    for item in raw.get("data") or []:

        news_item = item.get("newsItem") or {}
        meta = news_item.get("itemMeta") or {}
        content = news_item.get("contentMeta") or {}

        titles = meta.get("title") or []
        headline = titles[0] if titles else None

        if isinstance(headline, dict):
            headline = headline.get("$")

        timestamp = meta.get("versionCreated")

        if isinstance(timestamp, dict):
            timestamp = timestamp.get("$")

        source = None

        for entry in (content.get("creator") or []) + (content.get("infoSource") or []):
            if isinstance(entry, dict) and entry.get("_qcode"):
                source = entry["_qcode"]
                break

        rows.append({
            "versionCreated": timestamp,
            "headline": headline,
            "storyId": item.get("storyId"),
            "sourceCode": source
        })

    df = pd.DataFrame(
        rows,
        columns=["versionCreated", "headline", "storyId", "sourceCode"]
    )

    if not df.empty:
        df["versionCreated"] = pd.to_datetime(
            df["versionCreated"], utc=True, errors="coerce"
        )
        df = df.dropna(subset=["versionCreated"])

    return df


# HANDLE 100-RESULT LIMIT

# Initial testing showed that all relevant Reuters headlines may not be retrieved entirely
# This part splits the observation period when the 100-result limit is reached to ensure complete retrieval of relevant Reuters news headlines

def get_ai_headlines(ric, start_date, end_date, depth=0):

    df = request_ai_headlines(ric, start_date, end_date)

    if df.empty:
        return []

    if len(df) < 100:
        df["SearchRIC"] = ric
        return [df]

    if depth >= 20:
        raise RuntimeError(
            f"Too many headline results "
            f"for {ric} between "
            f"{start_date} and {end_date}."
        )

    start = pd.Timestamp(
        start_date
    )

    end = pd.Timestamp(
        end_date
    )

    if (
        end - start
        <= pd.Timedelta(
            seconds=10
        )
    ):
        raise RuntimeError(
            f"AI headline interval too dense "
            f"for {ric}: "
            f"{start_date} to {end_date}"
        )

    midpoint = (
        start
        +
        (end - start) / 2
    )

    print(
        f"SPLIT AFTER 100 RESULTS | "
        f"{ric} | "
        f"{start_date} to {end_date}"
    )

    left = get_ai_headlines(
        ric,
        start.strftime("%Y-%m-%dT%H:%M:%S"),
        midpoint.strftime("%Y-%m-%dT%H:%M:%S"),
        depth + 1
    )

    right = get_ai_headlines(
        ric,
        (midpoint + pd.Timedelta(seconds=1)).strftime(
            "%Y-%m-%dT%H:%M:%S"
        ),
        end.strftime("%Y-%m-%dT%H:%M:%S"),
        depth + 1
    )

    return left + right


# DOWNLOAD OF NEWS HEADLINES

# Executes the previously defined Reuters news search under consideration of all previously defined parameters
# Saves identified headlines in headline_cache as part of the AI_EVENTS_CACHE

all_ai_headlines = []

print(
    f"\nSearching Reuters headlines "
    f"for {len(rics_spx)} RICs"
)

for i, ric in enumerate(rics_spx, start=1):

    print(f"{i}/{len(rics_spx)} | {ric}")

    cache_key = (ric, tuple(ai_keywords), START_DATE, END_DATE)

    if cache_key in headline_cache:
        frames = headline_cache[cache_key]
    else:
        frames = get_ai_headlines(ric, START_DATE, END_DATE)

        headline_cache[cache_key] = frames
        save_cache()

    all_ai_headlines.extend(frames)


# IDENTIFY FULL STORIES TO RETRIEVE

# Combines Reuters AI headline results
# Determines S&P 500 membership of corresponding companies using the event month
# Preserves original UTC timestamps for subsequent event-time analysis

if not all_ai_headlines:
    raise RuntimeError("No Reuters headlines found.")

headline_df = pd.concat(all_ai_headlines, ignore_index=True)

headline_df["versionCreated"] = pd.to_datetime(
    headline_df["versionCreated"], utc=True, errors="raise"
)

headline_df["MembershipDate"] = (
    headline_df["versionCreated"]
    .dt.tz_convert("America/New_York")
    .dt.strftime("%Y-%m-01")
)

headline_df["SPXMember"] = headline_df.apply(
    lambda row: row["SearchRIC"] in spx_cache[row["MembershipDate"]],
    axis=1
)


# DOWNLOAD OF IDENTIFIED FULL STORIES

# Downloads full Reuters stories for eligible S&P 500 observations
# Retrieves associated company RICs and full story text
# Reuses previously cached stories

# Identification of all relevant Story IDs

story_ids = (
    headline_df.loc[headline_df["SPXMember"], "storyId"]
    .dropna()
    .unique()
)

# Check, if there are Story IDs which have previously not been chached

missing_story_ids = [
    story_id for story_id in story_ids
    if story_id not in story_cache
    or not str(story_cache[story_id].get("story_text") or "").strip()
]

print(f"\nUnique stories: {len(story_ids)}")
print(f"Stories to download: {len(missing_story_ids)}")

# Download loop for all Story IDs to download

for i, story_id in enumerate(missing_story_ids, start=1):

    print(f"{i}/{len(missing_story_ids)} | Downloading story")

    response = safe_request(
        lambda story_id=story_id: ld.delivery.endpoint_request.Definition(
            url="/data/news/v1/stories/" + story_id,
            method=ld.delivery.endpoint_request.RequestMethod.GET
        ).get_data(),
        label=f"Story {story_id}"
    )

    raw = response.data.raw

    if not isinstance(raw, dict):
        raise RuntimeError(f"Unexpected story response for {story_id}")

    news_item = raw.get("newsItem") or {}
    content_meta = news_item.get("contentMeta") or {}

    # Extract associated Reuters RICs

    associated_rics = []

    for item in (news_item.get("assert") or []) + (content_meta.get("subject") or []):
        qcode = item.get("_qcode", "")

        if qcode.startswith("R:"):
            associated_rics.append(qcode.split(":", 1)[1])

    associated_rics = list(dict.fromkeys(associated_rics))

    # Extract full story text

    content_set = news_item.get("contentSet") or {}

    story_text = ""

    for value in [
        content_set.get("inlineXML"),
        news_item.get("body"),
        news_item.get("content")
    ]:
        if value:
            story_text = value.get("$", "") if isinstance(value, dict) else value
            if story_text:
                break

    # Cache retrieved stories

    story_cache[story_id] = {
        "associated_rics": associated_rics,
        "story_text": story_text
    }

    # Save cache after every 10 downloaded stories

    if i % 10 == 0:
        save_cache()

# Final cache save

save_cache()

stories_with_text = sum(
    bool(str(story_cache.get(sid, {}).get("story_text") or "").strip())
    for sid in story_ids
)

print(
    f"Full story download complete: "
    f"{stories_with_text}/{len(story_ids)} stories with text"
)


###############################################################################################################
# RETRIEVAL OF CONFOUNDING EVENTS
###############################################################################################################

# CREATION OF CONFOUNDING EVENT SEARCH WINDOWS

# Determines the confounding search window [-1, +5] using NYSE trading days and the first market close following each AI event announcement
# The search begins at the close of T-2 to cover the full return period of T-1

# Selection of relevant AI disclosure events

ai_event_dates = (
    headline_df.loc[
        headline_df["SPXMember"],
        ["SearchRIC", "versionCreated"]
    ]
    .drop_duplicates()
    .copy()
)

# Check, wether AI events are contained

if ai_event_dates.empty:
    raise RuntimeError("No eligible AI observations for confounding searches")

# Harmonization of publication dates to UTC for NYSE trading calendar

ai_event_dates["Event Timestamp UTC"] = pd.to_datetime(
    ai_event_dates["versionCreated"], utc=True
)

calendar = mcal.get_calendar("NYSE")

# Determine trading hours for the relevant search periods

schedule = calendar.schedule(
    start_date=(
        ai_event_dates["Event Timestamp UTC"].min()
        - pd.Timedelta(days=30)
    ).strftime("%Y-%m-%d"),
    end_date=(
        ai_event_dates["Event Timestamp UTC"].max()
        + pd.Timedelta(days=30)
    ).strftime("%Y-%m-%d")
)

# Extract market closings

market_closes = pd.DatetimeIndex(schedule["market_close"])

# Definition of get_confounding_search_window for calculation of search windows

def get_confounding_search_window(timestamp):
    position = market_closes.searchsorted(timestamp, side="right")

    if position < 2 or position + 5 >= len(market_closes):
        raise RuntimeError("Trading calendar does not cover the event window")

    return market_closes[position - 2], market_closes[position + 5]

# Calculate individual confounding search windows for each AI event

windows = [
    get_confounding_search_window(timestamp)
    for timestamp in ai_event_dates["Event Timestamp UTC"]
]

ai_event_dates["Window Start"] = [start for start, end in windows]
ai_event_dates["Window End"] = [end for start, end in windows]

CONFOUNDING_START_DATE = (
    ai_event_dates["Window Start"].min()
    .strftime("%Y-%m-%dT00:00:00Z")
)

CONFOUNDING_END_DATE = (
    ai_event_dates["Window End"].max()
    .strftime("%Y-%m-%dT23:59:59Z")
)

# MERGE OVERLAPPING WINDOWS BY RIC

# Combines overlapping confounding search windows for each company to reduce redundant Reuters news requests

confounding_event_windows = {}

for ric, group in ai_event_dates.groupby("SearchRIC"):

    windows = group[
        ["Window Start", "Window End"]
    ].sort_values("Window Start").itertuples(index=False, name=None)

    merged_windows = []

    for start, end in windows:

        if merged_windows and start <= merged_windows[-1][1]:
            merged_windows[-1][1] = max(merged_windows[-1][1], end)
        else:
            merged_windows.append([start, end])

    confounding_event_windows[ric] = merged_windows

rics_with_ai_events = sorted(confounding_event_windows)

print(f"\nRICs requiring confounding event searches: {len(rics_with_ai_events)}")
print(f"Confounding event search windows: {sum(map(len, confounding_event_windows.values()))}")


# DEFINITION OF KEYWORDS FOR CONFOUNDING EVENTS

# Definition of keywords for confounding event search
# Definition of potential confounding event categories are based on Bhagwat et al. (2020) building on Borah and Tellis (2014)
# Empirical validation of confounding Event categories based on Jeon et al. (2022) and Neuhierl et al. (2013)

CONFOUNDING_EVENT_KEYWORDS = [

    # Earnings and financial results

    "earnings",
    "financial results",
    "quarterly results",
    "annual results",
    "profit warning",

    # Dividend announcements

    "dividend",

    # Executive Adjustments

    "CEO resigns",
    "CEO resignation",
    "CEO departure",
    "CEO appointment",
    "appoints CEO",
    "CEO steps down",
    "CEO retires",

    #  Stock splits or structural stock adjustments

    "stock split",
    "share split",
    "stock consolidation",
    "share consolidation",

    # Legal events

    "lawsuit",
    "litigation",
    "class action",
    "settlement",

    #  Merger and acquisition activities

    "acquisition",
    "acquires",
    "merger",
    "takeover",

    # Product launches

    "product launch",
    "new product",
    "launches new",
    "unveils new",
    "introduces new product",

    # Product failures or recalls

    "product recall",
    "recalls product",
    "product defect",
    "product safety issue",

]

def confounding_event_indicators():
    return "(" + " OR ".join(
        f'"{keyword}"' for keyword in CONFOUNDING_EVENT_KEYWORDS
    ) + ")"

# REUTERS CONFOUNDING EVENT HEADLINES REQUEST

# Retrieves Reuters headlines matching the defined confounding keywords within the specified company-specific search window

def request_confounding_event_headlines(ric, start_date, end_date):

    query = (
        f"R:{ric} "
        f"AND Language:LEN "
        f"AND Source:RTRS "
        f"AND {confounding_event_indicators()} "
        f"AND SearchIn:HeadlineOnly"
    )

    response = safe_request(
        lambda: ld.delivery.endpoint_request.Definition(
            url="/data/news/v1/headlines",
            method=ld.delivery.endpoint_request.RequestMethod.GET,
            query_parameters={
                "query": query,
                "dateFrom": start_date,
                "dateTo": end_date,
                "limit": 100,
                "sort": "oldToNew"
            }
        ).get_data(),
        label=f"Confounding headlines {ric}",
        waiting_time=[3, 10]
    )

    raw = response.data.raw

    if not isinstance(raw, dict):
        raise RuntimeError(f"Unexpected Reuters response for {ric}")

    rows = []

    for item in raw.get("data") or []:

        news_item = item.get("newsItem") or {}
        meta = news_item.get("itemMeta") or {}

        titles = meta.get("title") or []
        headline = titles[0] if titles else None

        if isinstance(headline, dict):
            headline = headline.get("$")

        timestamp = meta.get("versionCreated")

        if isinstance(timestamp, dict):
            timestamp = timestamp.get("$")

        rows.append({
            "versionCreated": timestamp,
            "headline": headline,
            "storyId": item.get("storyId")
        })

    df = pd.DataFrame(
        rows,
        columns=["versionCreated", "headline", "storyId"]
    )

    if not df.empty:
        df["versionCreated"] = pd.to_datetime(
            df["versionCreated"], utc=True, errors="coerce"
        )
        df = df.dropna(subset=["versionCreated"])

    return df

# RETRIEVE HEADLINES OF CONFOUNDING EVENTS

# Recursively splits search intervals when Reuters returns 100 results or when request errors prevent successful retrieval
# Raises an error if complete retrieval cannot be ensured

def get_confounding_event_headlines(ric, start_date, end_date, depth=0):

    start = pd.to_datetime(start_date, utc=True)
    end = pd.to_datetime(end_date, utc=True)

    try:
        df = request_confounding_event_headlines(
            ric, start_date, end_date
        )

    except Exception as e:

        if type(e).__name__ not in [
            "RemoteProtocolError",
            "ReadTimeout",
            "ConnectTimeout",
            "ConnectError",
            "LDError"
        ]:
            raise

        if depth >= 10 or end - start <= pd.Timedelta(minutes=1):
            raise RuntimeError(
                f"Confounding event request failed for {ric}: "
                f"{start_date} to {end_date}"
            ) from e

        print(f"SPLIT AFTER ERROR | {ric} | {start_date} to {end_date}")

    else:

        if df.empty:
            return []

        if len(df) < 100:
            df = df.reset_index()
            df["RIC"] = ric
            return [df]

        if depth >= 20 or end - start <= pd.Timedelta(seconds=10):
            raise RuntimeError(
                f"Confounding event retrieval incomplete for {ric}: "
                f"{start_date} to {end_date}"
            )

        print(f"SPLIT AFTER 100 RESULTS | {ric} | {start_date} to {end_date}")

    # Split the interval and retrieve both halves

    midpoint = (start + (end - start) / 2).floor("s")

    left = get_confounding_event_headlines(
        ric,
        start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        midpoint.strftime("%Y-%m-%dT%H:%M:%SZ"),
        depth + 1
    )

    right = get_confounding_event_headlines(
        ric,
        (midpoint + pd.Timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        depth + 1
    )

    return left + right

# DOWNLOAD REUTERS CONFOUNDING EVENTS

# Retrieves and caches Reuters confounding event headlines for all relevant companies and merged search windows
# Previously completed searches are skipped

print(
    f"\nSearching Reuters confounding events for "
    f"{len(confounding_event_windows)} RICs with identified AI events"
)

# Iterate through all relevant RICs and their merged search windows

for i, (ric, windows) in enumerate(confounding_event_windows.items(), start=1):

    print(
        f"{i}/{len(confounding_event_windows)} | "
        f"{ric} | {len(windows)} window(s)"
    )

    for window_start, window_end in windows:

        # Convert pandas timestamps to UTC-formatted strings

        search_start = window_start.strftime("%Y-%m-%dT%H:%M:%SZ")
        search_end = window_end.strftime("%Y-%m-%dT%H:%M:%SZ")

        cache_key = (
            "confounding_keywords",
            ric,
            search_start,
            search_end,
            tuple(CONFOUNDING_EVENT_KEYWORDS)
        )

# Reuse cached searches only if they contain valid headline results

        if cache_key in confounding_cache:
            continue

        frames = get_confounding_event_headlines(
            ric, search_start, search_end
        )

        confounding_cache[cache_key] = frames

        save_cache()

# CLASSIFICATION OF REUTERS CONFOUNDING EVENTS

# Classifies identified Reuters news into the predefined
# confounding event categories based on matching keywords

def classify_confounding_events(text):

    if pd.isna(text):
        return "Potential Confounding Event"

    text = str(text).lower()

    # Earnings and financial results

    if any(term in text for term in [
        "earnings",
        "financial results",
        "quarterly results",
        "annual results",
        "profit warning"
    ]):
        return "Earnings and Financial Results"

    # Dividend announcements

    if any(term in text for term in [
        "dividend"
    ]):
        return "Dividend Announcement"

    # Executive adjustments (CEO-level)

    if any(term in text for term in [
        "ceo resigns",
        "ceo resignation",
        "ceo departure",
        "ceo appointment",
        "appoints ceo",
        "ceo steps down",
        "ceo retires"
    ]):
        return "Executive Adjustment"

    # Stock splits or structural stock adjustments

    if any(term in text for term in [
        "stock split",
        "share split",
        "stock consolidation",
        "share consolidation"
    ]):
        return "Stock Split / Structural Adjustment"

    # Legal events

    if any(term in text for term in [
        "lawsuit",
        "litigation",
        "class action",
        "settlement"
    ]):
        return "Legal Event"

    # Mergers and Acquisitions

    if any(term in text for term in [
        "acquisition",
        "acquires",
        "merger",
        "takeover"
    ]):
        return "Merger / Acquisition"

    # Product Launches

    if any(term in text for term in [
	"product launch",
	"new product",
	"launches new",
	"unveils new",
	"introduces new product",
    ]):
        return "Product Launch"

    # Product Failures / Recalls

    if any(term in text for term in [
        "product recall",
	"recalls product",
	"product defect",
	"product safety issue",
    ]):
        return "Product Failure / Recalls"

    return "Potential Confounding Event"

# CREATION OF REUTERS CONFOUNDING EVENT TABLE

# Creates a standardized table of retrieved Reuters confounding event
# candidates and assigns predefined categories based on their headlines

confounding_event_frames = []

valid_confounding_keys = {
    (
        "confounding_keywords",
        ric,
        start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        tuple(CONFOUNDING_EVENT_KEYWORDS)
    )
    for ric, windows in confounding_event_windows.items()
    for start, end in windows
}

for cache_key, frames in confounding_cache.items():

    if cache_key not in valid_confounding_keys:
        continue

    ric = cache_key[1]

    for frame in frames:

        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue

        frame = frame.copy()

        frame["versionCreated"] = pd.to_datetime(
            frame["versionCreated"], utc=True
        )

        frame["Company Name"] = company_cache.get(ric)

        frame["Date"] = frame["versionCreated"].dt.strftime("%Y-%m-%d")
        frame["Time"] = frame["versionCreated"].dt.strftime("%H:%M:%S")

        frame["Confounding Event"] = frame["headline"].apply(
            classify_confounding_events
        )

        frame["Source Type"] = "Reuters"

        frame = frame.rename(
            columns={"headline": "Headline / Document Title"}
        )

        confounding_event_frames.append(
            frame[
                [
                    "RIC",
                    "Company Name",
                    "Date",
                    "Time",
                    "Confounding Event",
                    "Source Type",
                    "Headline / Document Title",
                    "storyId"
                ]
            ]
        )

# COMBINE REUTERS CONFOUNDING EVENTS

# Combines all identified Reuters confounding event candidates into one Pandas DataFrame

CONFOUNDING_COLUMNS = [
    "RIC",
    "Company Name",
    "Date",
    "Time",
    "Confounding Event",
    "Source Type",
    "Headline / Document Title",
    "storyId"
]

if confounding_event_frames:

    confounding_events = (
        pd.concat(
            confounding_event_frames,
            ignore_index=True
        )
        .drop_duplicates(subset=["RIC", "storyId"])
        .sort_values(["Date", "Time", "RIC"])
        .reset_index(drop=True)
    )

else:

    confounding_events = pd.DataFrame(
        columns=CONFOUNDING_COLUMNS
    )

print(
    f"Reuters confounding events identified: "
    f"{len(confounding_events)}"
)

# SAVE FINAL CONFOUNDING EVENT CACHE

# Saves the consolidated confounding event dataset alongside
# the previously retrieved search results

confounding_data["events"] = confounding_events

save_cache()

# EXPORT AI AND CONFOUNDING EVENTS TO EXCEL

# Prepares and exports the identified AI event candidates and potential confounding events into separate excel files

AI_EVENTS_EXCEL = Path("SPX_AI_DISCLOSURE_EVENTS.xlsx")
CONFOUNDING_EVENTS_EXCEL = Path("SPX_CONFOUNDING_EVENTS.xlsx")

# Prepare AI events for export

ai_events_export = headline_df.loc[headline_df["SPXMember"]].copy()

ai_events_export["Company Name"] = ai_events_export["SearchRIC"].map(company_cache)
ai_events_export["Sector"] = ai_events_export["SearchRIC"].map(sector_cache)

# Convert UTC timestamps to strings to ensure Excel compatibility

ai_events_export["versionCreated"] = (
    pd.to_datetime(ai_events_export["versionCreated"], utc=True)
    .dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
)

ai_events_export = (
    ai_events_export[
        [
            "SearchRIC",
            "Company Name",
            "Sector",
            "versionCreated",
            "headline",
            "storyId",
            "sourceCode"
        ]
    ]
    .drop_duplicates()
    .sort_values(["SearchRIC", "versionCreated"])
    .reset_index(drop=True)
)

# Prepare confounding events for export

confounding_events_export = (
    confounding_events
    .sort_values(["RIC", "Date", "Time", "Source Type"])
    .reset_index(drop=True)
)

# Export AI events

with pd.ExcelWriter(AI_EVENTS_EXCEL, engine="openpyxl") as writer:
    ai_events_export.to_excel(writer, sheet_name="AI Events", index=False)

# Export confounding events

with pd.ExcelWriter(CONFOUNDING_EVENTS_EXCEL, engine="openpyxl") as writer:
    confounding_events_export.to_excel(
        writer, sheet_name="Confounding Events", index=False
    )

print(f"AI Events Exported: {len(ai_events_export)}")
print(f"Confounding Events Exported: {len(confounding_events_export)}")


# DISPLAY FINAL DATA RETRIEVAL STATUS

# Summarizes retrieved data and confirms the location of saved excel exports

print("\nDATA RETRIEVAL COMPLETE")
print(f"S&P 500 Snapshots: {len(spx_cache)}")
print(f"Headline Searches: {len(headline_cache)}")
print(f"Reuters Stories: {len(story_cache)}")
print(f"AI Event Candidates: {len(ai_events_export)}")
print(f"Potential Confounding Events: {len(confounding_events_export)}")
print(f"\nAI Event Cache: {AI_EVENTS_CACHE}")
print(f"Confounding Event Cache: {CONFOUNDING_EVENTS_CACHE}")
print(f"\nAI Event Excel Export: {AI_EVENTS_EXCEL}")
print(f"Confounding Event Excel Export: {CONFOUNDING_EVENTS_EXCEL}")
