"""Parse a WhatsApp 'Export chat' .txt file (Android and iPhone formats) into a DataFrame."""

import re

import pandas as pd

# Android: 31/12/23, 10:15 pm - Ravi: message      iPhone: [31/12/23, 10:15:30 PM] Ravi: message
LINE_RE = re.compile(
    r"^‎?\[?(?P<date>\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}),?\s+"
    r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?(?:\s?[apAP]\.?\s?[mM]\.?)?)\]?\s*(?:-|–)?\s*"
    r"(?P<rest>.*)$"
)
SKIP = ("<media omitted>", "image omitted", "video omitted", "sticker omitted", "audio omitted",
        "gif omitted", "document omitted", "this message was deleted", "you deleted this message",
        "null", "<this message was edited>")


def parse_chat(text, dayfirst=True):
    rows = []
    for line in text.splitlines():
        line = line.replace(" ", " ").strip()
        m = LINE_RE.match(line)
        if m:
            rest = m.group("rest")
            if ": " not in rest:            # system line: "Ravi added Priya", "Messages are encrypted"
                continue
            user, msg = rest.split(": ", 1)
            rows.append({"date": m.group("date"), "time": m.group("time"), "user": user.strip(), "message": msg})
        elif rows and line:                 # multi-line message continues
            rows[-1]["message"] += " " + line

    df = pd.DataFrame(rows, columns=["date", "time", "user", "message"])
    if df.empty:
        return df
    stamp = (df.date + " " + df.time.str.replace(".", "", regex=False)).str.upper()
    df["datetime"] = pd.to_datetime(stamp, dayfirst=dayfirst, errors="coerce", format="mixed")
    df = df.dropna(subset=["datetime"])
    df = df[~df.message.str.strip().str.lower().isin(SKIP)]
    df = df[df.message.str.strip().str.len() > 0]
    df["month"] = df.datetime.dt.to_period("M").astype(str)
    df["day"] = df.datetime.dt.date
    df["weekday"] = df.datetime.dt.day_name()
    return df.reset_index(drop=True)
