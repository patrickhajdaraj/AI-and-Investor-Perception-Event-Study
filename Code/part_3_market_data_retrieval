"""

PART 3: MARKET DATA RETRIEVAL

Purpose:
	Retrieves historical stock and market index prices for the prepared AI disclosure events 
	to prepare the dataset for subsequent event study calculations


Contained steps:
	- Assigns event trading days based on Reuters announcement timestamps and the NYSE trading calendar
	- Accounts for announcements outside regular trading hours, weekends and market Holidays
	- Retrieves historical daily opening and closing prices for the corresponding S&P 500 companies using LSEG
	- Retrieves historical daily closing prices for the S&P 500 Index as the market benchmark
	- Matches company and market index prices to the corresponding event trading days
	- Checks the completeness of retrieved market data and identifies missing price observations
	- Creates a consolidated dataset combining prepared AI disclosure events with historical market prices

Input: 
	- SPX_AI_EVENTS_PREPARED.xlsx created in Part 2

Output: 
	- Cached historical company and S&P 500 price data (SPX_AI_EVENT_PRICES_CACHE.pkl)
	- Final AI disclosure dataset with corresponding market prices (SPX_AI_EVENTS_WITH_PRICES.pkl)
	- Excel export of AI disclosure events and market prices (SPX_AI_EVENTS_WITH_PRICES.xlsx)
	
Important notes:
	- The query is designed for LSEG CodeBook, the Jupyter-based Python environment integrated into LSEG Workspace
	- It can also be executed in other Python environments, such as Visual Studio Code or a local Jupyter Notebook, provided that
  	  the LSEG Data Library is installed and a compatible session is configured
	- Access to LSEG data is required

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
import pandas_market_calendars as mcal


# OPEN LSEG SESSION

ld.open_session()


# DEFINE INPUT AND OUTPUT FILES

PREPARED_EVENTS_FILE = Path(
    "SPX_AI_EVENTS_PREPARED.xlsx"
)

PRICE_CACHE_FILE = Path(
    "SPX_AI_EVENT_PRICES_CACHE.pkl"
)

OUTPUT_PKL = Path(
    "SPX_AI_EVENTS_WITH_PRICES.pkl"
)

OUTPUT_XLSX = Path(
    "SPX_AI_EVENTS_WITH_PRICES.xlsx"
)


# LOAD PREPARED EVENTS

if not PREPARED_EVENTS_FILE.exists():
    raise FileNotFoundError(
        f"Prepared event table not found: "
        f"{PREPARED_EVENTS_FILE}"
    )

events = pd.read_excel(
    PREPARED_EVENTS_FILE
)

print(
    f"Prepared events loaded: "
    f"{len(events)}"
)


# CHECK REQUIRED COLUMNS

required_columns = [
    "Event_ID",
    "Date",
    "Time",
    "Source",
    "Relevant RIC",
    "Company Name",
    "Headline"
]

missing_columns = [
    column
    for column in required_columns
    if column not in events.columns
]

if missing_columns:
    raise RuntimeError(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )


# CREATE EVENT TIMESTAMP IN UTC

events[
    "Event Timestamp UTC"
] = pd.to_datetime(
    events["Date"].astype(str).str.strip()
    + " "
    + events["Time"].astype(str).str.strip(),
    format="%Y-%m-%d %H:%M:%S",
    utc=True,
    errors="raise"
)

if events[
    "Event Timestamp UTC"
].isna().any():
    raise RuntimeError(
        "Some AI events contain missing timestamps."
    )


# CREATE NYSE TRADING CALENDAR

calendar = mcal.get_calendar(
    "NYSE"
)

schedule = calendar.schedule(
    start_date=(
        events[
            "Event Timestamp UTC"
        ].min()
        - pd.Timedelta(days=30)
    ).strftime(
        "%Y-%m-%d"
    ),
    end_date=(
        events[
            "Event Timestamp UTC"
        ].max()
        + pd.Timedelta(days=30)
    ).strftime(
        "%Y-%m-%d"
    )
)

market_closes = pd.DatetimeIndex(
    pd.to_datetime(
        schedule[
            "market_close"
        ],
        utc=True
    )
)

trading_dates = pd.DatetimeIndex(
    pd.to_datetime(
        schedule.index
    )
).normalize()

###############################################################################################################
# DEFINITION OF TRADIBG DAYS BASED ON NYSE TRADING CALENDAR
###############################################################################################################

# ASSIGN EVENT TRADING DAYS

def assign_trading_days(
    event_timestamp
):

    position = market_closes.searchsorted(
        event_timestamp,
        side="right"
    )

    if (
        position < 1
        or position + 5 >= len(
            market_closes
        )
    ):
        raise RuntimeError(
            "Trading calendar does not "
            "cover required event window."
        )

    return {
        "T-1":
            trading_dates[
                position - 1
            ],

        "T0":
            trading_dates[
                position
            ],

        "T+1":
            trading_dates[
                position + 1
            ],

        "T+2":
            trading_dates[
                position + 2
            ],

        "T+3":
            trading_dates[
                position + 3
            ],

        "T+4":
            trading_dates[
                position + 4
            ],

        "T+5":
            trading_dates[
                position + 5
            ]
            }

trading_day_assignments = (
    events[
        "Event Timestamp UTC"
    ]
    .apply(
        assign_trading_days
    )
)


events[
    "Trading Date T-1"
] = trading_day_assignments.apply(
    lambda x: x["T-1"]
)

events[
    "Trading Date T0"
] = trading_day_assignments.apply(
    lambda x: x["T0"]
)

events[
    "Trading Date T+1"
] = trading_day_assignments.apply(
    lambda x: x["T+1"]
)

events[
    "Trading Date T+2"
] = trading_day_assignments.apply(
    lambda x: x["T+2"]
)

events[
    "Trading Date T+3"
] = trading_day_assignments.apply(
    lambda x: x["T+3"]
)

events[
    "Trading Date T+4"
] = trading_day_assignments.apply(
    lambda x: x["T+4"]
)

events[
    "Trading Date T+5"
] = trading_day_assignments.apply(
    lambda x: x["T+5"]
)


# DEFINE PRICE RETRIEVAL PERIOD

PRICE_START_DATE = (
    events[
        "Trading Date T-1"
    ]
    .min()
    - pd.Timedelta(days=5)
).strftime(
    "%Y-%m-%d"
)

PRICE_END_DATE = (
    events[
        "Trading Date T+5"
    ]
    .max()
    + pd.Timedelta(days=5)
).strftime(
    "%Y-%m-%d"
)

print(
    f"Price retrieval period: "
    f"{PRICE_START_DATE} to "
    f"{PRICE_END_DATE}"
)


# LOAD PRICE CACHE

if PRICE_CACHE_FILE.exists():

    try:

        with PRICE_CACHE_FILE.open(
            "rb"
        ) as f:

            price_cache = pickle.load(
                f
            )

    except Exception as e:

        print(
            f"Could not load price cache: "
            f"{e}"
        )

        price_cache = {}

else:

    price_cache = {}


def save_price_cache():

    with PRICE_CACHE_FILE.open(
        "wb"
    ) as f:

        pickle.dump(
            price_cache,
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )


# SAFE REQUEST

def safe_request(
    request_function,
    label="LSEG request",
    waits=None
):

    if waits is None:

        waits = [
            5,
            15,
            30,
            60
        ]

    for attempt in range(
        len(waits) + 1
    ):

        try:

            result = (
                request_function()
            )

            time.sleep(
                0.4
            )

            return result

        except Exception as e:

            error_text = str(e)

            if (
                "The universe is not found"
                in error_text
                or
                "TS.Interday.UserRequestError.70005"
                in error_text
            ):
                raise

            if attempt == len(
                waits
            ):

                save_price_cache()

                print(
                    f"FAILED | "
                    f"{label} | "
                    f"{type(e).__name__}: "
                    f"{e}"
                )

                raise

            wait = waits[
                attempt
            ]

            print(
                f"RETRY | "
                f"{label} | "
                f"waiting {wait}s"
            )

            time.sleep(
                wait
            )


# NORMALIZE HISTORICAL PRICE RESPONSE

def normalize_price_history(
    df,
    ric
):

    if (
        df is None
        or df.empty
    ):

        raise RuntimeError(
            f"No historical prices "
            f"returned for {ric}"
        )

    df = df.copy()

    if isinstance(
        df.columns,
        pd.MultiIndex
    ):

        if ric in (
            df.columns
            .get_level_values(0)
        ):

            df = df.xs(
                ric,
                axis=1,
                level=0
            )

        elif ric in (
            df.columns
            .get_level_values(-1)
        ):

            df = df.xs(
                ric,
                axis=1,
                level=-1
            )

    df.index = pd.to_datetime(
        df.index
    )

    if (
        getattr(
            df.index,
            "tz",
            None
        )
        is not None
    ):

        df.index = (
            df.index
            .tz_localize(None)
        )

    df.index = (
        df.index
        .normalize()
    )

    df = (
        df[
            ~df.index.duplicated(
                keep="last"
            )
        ]
        .sort_index()
    )

    return df

###############################################################################################################
# RETRIEVAL OF STOCK AND SPX-INDEX PRICES
###############################################################################################################

# RETRIEVE COMPANY PRICE HISTORY

unique_rics = sorted(
    events[
        "Relevant RIC"
    ]
    .dropna()
    .astype(str)
    .unique()
)

print(
    f"Unique company RICs requiring "
    f"price data: {len(unique_rics)}"
)


failed_price_rics = []

for i, ric in enumerate(
    unique_rics,
    start=1
):

    cache_key = (
        ric,
        PRICE_START_DATE,
        PRICE_END_DATE
    )

    if cache_key in price_cache:

        print(
            f"{i}/{len(unique_rics)} | "
            f"{ric} | cached"
        )

        continue

    print(
        f"{i}/{len(unique_rics)} | "
        f"{ric} | requesting"
    )

    try:

        price_df = safe_request(

            lambda ric=ric:

                ld.get_history(
                    universe=ric,
                    fields=[
                        "OPEN_PRC",
                        "TRDPRC_1"
                    ],
                    interval="daily",
                    start=PRICE_START_DATE,
                    end=PRICE_END_DATE
                ),

            label=f"Historical prices {ric}"
        )

        price_cache[
            cache_key
        ] = normalize_price_history(
            price_df,
            ric
        )

        save_price_cache()

    except Exception as e:

        error_text = str(e)

        if (
            "The universe is not found"
            in error_text
            or
            "TS.Interday.UserRequestError.70005"
            in error_text
        ):

            print(
                f"WARNING | Historical price universe "
                f"not found | {ric}"
            )

            failed_price_rics.append(
                ric
            )

            continue

        raise

# RETRIEVE S&P 500 PRICE HISTORY

SPX_RIC = ".SPX"

spx_cache_key = (
    SPX_RIC,
    PRICE_START_DATE,
    PRICE_END_DATE
)

if spx_cache_key not in price_cache:

    print(
        "Requesting S&P 500 "
        "historical prices"
    )

    spx_df = safe_request(

        lambda:

            ld.get_history(
                universe=SPX_RIC,
                fields=[
                    "TR.PriceClose"
                ],
                interval="daily",
                start=PRICE_START_DATE,
                end=PRICE_END_DATE
            ),

        label="Historical prices .SPX"
    )

    price_cache[
        spx_cache_key
    ] = normalize_price_history(
        spx_df,
        SPX_RIC
    )

    save_price_cache()

else:

    print(
        "S&P 500 historical prices | cached"
    )


# HELPER TO RETRIEVE PRICE FROM CACHE

def get_cached_price(
    ric,
    trading_date,
    field
):

    cache_key = (
        ric,
        PRICE_START_DATE,
        PRICE_END_DATE
    )

    if cache_key not in price_cache:

        return None

    df = price_cache[
        cache_key
    ]

    trading_date = (
        pd.Timestamp(
            trading_date
        )
        .normalize()
    )

    if trading_date not in df.index:

        return None

    if field not in df.columns:

        return None

    value = df.loc[
        trading_date,
        field
    ]

    if isinstance(
        value,
        pd.Series
    ):

        value = value.iloc[
            -1
        ]

    if pd.isna(
        value
    ):

        return None

    return float(
        value
    )


# ADD COMPANY PRICES TO EACH EVENT

events[
    "Closing Price T-1"
] = events.apply(

    lambda row:

        get_cached_price(
            row["Relevant RIC"],
            row["Trading Date T-1"],
            "TRDPRC_1"
        ),

    axis=1
)


events[
    "Opening Price Disclosure Day"
] = events.apply(

    lambda row:

        get_cached_price(
            row["Relevant RIC"],
            row["Trading Date T0"],
            "OPEN_PRC"
        ),

    axis=1
)


events[
    "Closing Price Disclosure Day"
] = events.apply(

    lambda row:

        get_cached_price(
            row["Relevant RIC"],
            row["Trading Date T0"],
            "TRDPRC_1"
        ),

    axis=1
)


events[
    "Opening Price T+1"
] = events.apply(

    lambda row:

        get_cached_price(
            row["Relevant RIC"],
            row["Trading Date T+1"],
            "OPEN_PRC"
        ),

    axis=1
)


events[
    "Closing Price T+1"
] = events.apply(

    lambda row:

        get_cached_price(
            row["Relevant RIC"],
            row["Trading Date T+1"],
            "TRDPRC_1"
        ),

    axis=1
)


events[
    "Opening Price T+2"
] = events.apply(

    lambda row:

        get_cached_price(
            row["Relevant RIC"],
            row["Trading Date T+2"],
            "OPEN_PRC"
        ),

    axis=1
)


events[
    "Closing Price T+2"
] = events.apply(

    lambda row:

        get_cached_price(
            row["Relevant RIC"],
            row["Trading Date T+2"],
            "TRDPRC_1"
        ),

    axis=1
)

events[
    "Opening Price T+3"
] = events.apply(
    lambda row: get_cached_price(
        row["Relevant RIC"],
        row["Trading Date T+3"],
        "OPEN_PRC"
    ),
    axis=1
)

events[
    "Closing Price T+3"
] = events.apply(
    lambda row: get_cached_price(
        row["Relevant RIC"],
        row["Trading Date T+3"],
        "TRDPRC_1"
    ),
    axis=1
)

events[
    "Opening Price T+4"
] = events.apply(
    lambda row: get_cached_price(
        row["Relevant RIC"],
        row["Trading Date T+4"],
        "OPEN_PRC"
    ),
    axis=1
)

events[
    "Closing Price T+4"
] = events.apply(
    lambda row: get_cached_price(
        row["Relevant RIC"],
        row["Trading Date T+4"],
        "TRDPRC_1"
    ),
    axis=1
)

events[
    "Opening Price T+5"
] = events.apply(

    lambda row:

        get_cached_price(
            row["Relevant RIC"],
            row["Trading Date T+5"],
            "OPEN_PRC"
        ),

    axis=1
)


events[
    "Closing Price T+5"
] = events.apply(

    lambda row:

        get_cached_price(
            row["Relevant RIC"],
            row["Trading Date T+5"],
            "TRDPRC_1"
        ),

    axis=1
)


# ADD S&P 500 CLOSING PRICES

events[
    "Closing Price SPX T-1"
] = events[
    "Trading Date T-1"
].apply(

    lambda date:

        get_cached_price(
            SPX_RIC,
            date,
            "Price Close"
        )
)


events[
    "Closing Price SPX T=0"
] = events[
    "Trading Date T0"
].apply(

    lambda date:

        get_cached_price(
            SPX_RIC,
            date,
            "Price Close"
        )
)


events[
    "Closing Price SPX T+1"
] = events[
    "Trading Date T+1"
].apply(

    lambda date:

        get_cached_price(
            SPX_RIC,
            date,
            "Price Close"
        )
)


events[
    "Closing Price SPX T+2"
] = events[
    "Trading Date T+2"
].apply(

    lambda date:

        get_cached_price(
            SPX_RIC,
            date,
            "Price Close"
        )
)

events[
    "Closing Price SPX T+3"
] = events[
    "Trading Date T+3"
].apply(
    lambda date: get_cached_price(
        SPX_RIC,
        date,
        "Price Close"
    )
)

events[
    "Closing Price SPX T+4"
] = events[
    "Trading Date T+4"
].apply(
    lambda date: get_cached_price(
        SPX_RIC,
        date,
        "Price Close"
    )
)

events[
    "Closing Price SPX T+5"
] = events[
    "Trading Date T+5"
].apply(

    lambda date:

        get_cached_price(
            SPX_RIC,
            date,
            "Price Close"
        )
)


# CREATE FINAL MARKET DATA TABLE

market_data = (
    events[
        [
            "Event_ID",
            "Date",
            "Time",
            "Trading Date T-1",
            "Trading Date T0",
            "Trading Date T+1",
	    "Trading Date T+2",
	    "Trading Date T+3",
	    "Trading Date T+4",
	    "Trading Date T+5",
            "Source",
            "Relevant RIC",
            "Company Name",
            "Sector",
            "Headline",
	    "Story Text",
            "Closing Price T-1",
            "Opening Price Disclosure Day",
            "Closing Price Disclosure Day",
            "Opening Price T+1",
            "Closing Price T+1",
	    "Opening Price T+2",
	    "Closing Price T+2",
	    "Opening Price T+3",
	    "Closing Price T+3",
	    "Opening Price T+4",
	    "Closing Price T+4",
	    "Opening Price T+5",
	    "Closing Price T+5",
            "Closing Price SPX T-1",
            "Closing Price SPX T=0",
            "Closing Price SPX T+1",
	    "Closing Price SPX T+2",
	    "Closing Price SPX T+3",
	    "Closing Price SPX T+4",
	    "Closing Price SPX T+5"
        ]
    ]
    .rename(
        columns={
            "Relevant RIC": "RIC"
        }
    )
    .copy()
)

# QUALITY CHECK FOR MISSING PRICES

price_columns = [
    "Closing Price T-1",
    "Opening Price Disclosure Day",
    "Closing Price Disclosure Day",
    "Opening Price T+1",
    "Closing Price T+1",
    "Opening Price T+2",
    "Closing Price T+2",
    "Opening Price T+3",
    "Closing Price T+3",
    "Opening Price T+4",
    "Closing Price T+4",
    "Opening Price T+5",
    "Closing Price T+5",
    "Closing Price SPX T-1",
    "Closing Price SPX T=0",
    "Closing Price SPX T+1",
    "Closing Price SPX T+2",
    "Closing Price SPX T+3",
    "Closing Price SPX T+4",
    "Closing Price SPX T+5"
]

missing_prices = (
    market_data[
        price_columns
    ]
    .isna()
    .sum()
)

print(
    "\nMissing price observations:"
)

print(
    missing_prices[
        missing_prices > 0
    ]
)

# CHECK COMPLETENESS OF MARKET DATA

complete_price_mask = market_data[price_columns].notna().all(axis=1)

print(
    "Events with complete price data:",
    complete_price_mask.sum()
)

print(
    "Events with missing price data:",
    (~complete_price_mask).sum()
)

print(
    "Share of events with complete price data:",
    f"{complete_price_mask.mean():.2%}"
)


# SAVE FINAL DATASET

with OUTPUT_PKL.open(
    "wb"
) as f:

    pickle.dump(
        market_data,
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )


market_data.to_excel(
    OUTPUT_XLSX,
    index=False
)


# DISPLAY FINAL STATUS

# Confirms successfull run

print()

print(
    "MARKET DATA RETRIEVAL COMPLETE"
)

print(
    f"Events with market data: "
    f"{len(market_data)}"
)

print(
    f"Unique company RICs: "
    f"{market_data['RIC'].nunique()}"
)

print(
    f"Price cache saved as: "
    f"{PRICE_CACHE_FILE}"
)

print(
    f"Final market data cache saved as: "
    f"{OUTPUT_PKL}"
)

print(
    f"Final Excel saved as: "
    f"{OUTPUT_XLSX}"
)
