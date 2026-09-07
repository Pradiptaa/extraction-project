"""Indonesian-aware normalizers: currency, dates, number-words, rates.

These exist because English-convention parsers silently corrupt Indonesian-
formatted numbers (`.` = thousands, `,` = decimal) and dates (day-month_name-year,
month names in Indonesian). See analisis_pipeline_kontrak.md section G.4 / 4.5.
"""
from __future__ import annotations

import re
from typing import Optional

MONTHS_ID = {
    "januari": 1, "februari": 2, "maret": 3, "april": 4, "mei": 5, "juni": 6,
    "juli": 7, "agustus": 8, "september": 9, "oktober": 10, "november": 11,
    "desember": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "agt": 8, "ags": 8, "sep": 9, "sept": 9, "okt": 10, "nov": 11, "des": 12,
}

_NUM_WORDS_ONES = {
    "nol": 0, "satu": 1, "dua": 2, "tiga": 3, "empat": 4, "lima": 5,
    "enam": 6, "tujuh": 7, "delapan": 8, "sembilan": 9,
}
_NUM_WORDS_TEENS = {
    "sepuluh": 10, "sebelas": 11, "duabelas": 12,
}
_NUM_WORDS_MAGNITUDE = {
    "puluh": 10, "ratus": 100, "ribu": 1_000, "juta": 1_000_000,
    "miliar": 1_000_000_000, "milyar": 1_000_000_000, "triliun": 1_000_000_000_000,
}


def parse_currency_id(raw: str) -> Optional[float]:
    """`Rp. 1.500.000.000,00` -> 1500000000.00. Returns None if unparseable."""
    if not raw:
        return None
    m = re.search(r"[\d.,]+", raw)
    if not m:
        return None
    token = m.group(0).strip(".,")
    if not token:
        return None
    if "," in token:
        integer_part, _, frac_part = token.rpartition(",")
        integer_part = integer_part.replace(".", "")
        try:
            return float(f"{integer_part}.{frac_part}")
        except ValueError:
            return None
    integer_part = token.replace(".", "")
    try:
        return float(integer_part)
    except ValueError:
        return None


def parse_number_words_id(text: str) -> Optional[int]:
    """`Seratus Dua Puluh` -> 120. Best-effort, used only for words_check cross-validation."""
    if not text:
        return None
    words = re.findall(r"[A-Za-zÀ-ÿ]+", text.lower())
    if not words:
        return None

    total = 0
    current = 0
    matched_any = False
    for w in words:
        if w == "belas":
            current = (current or 1) + 10
            matched_any = True
            continue
        # "se-" prefix means "one of" (seratus=100, seribu=1000, sepuluh=10 is
        # already listed explicitly): se+MAGNITUDE collapses to 1*MAGNITUDE.
        if w.startswith("se") and w[2:] in _NUM_WORDS_MAGNITUDE:
            mult = _NUM_WORDS_MAGNITUDE[w[2:]]
            current = (current or 1) * mult
            if mult >= 100:
                # flush hundreds/thousands/millions immediately: the words
                # that follow (e.g. "dua puluh" after "seratus") start a new,
                # additive lower-magnitude segment rather than multiplying
                # into this one.
                total += current
                current = 0
            matched_any = True
            continue
        if w in _NUM_WORDS_MAGNITUDE:
            mult = _NUM_WORDS_MAGNITUDE[w]
            current = (current or 1) * mult
            if mult >= 100:
                total += current
                current = 0
            matched_any = True
            continue
        if w in _NUM_WORDS_TEENS:
            current += _NUM_WORDS_TEENS[w]
            matched_any = True
            continue
        if w in _NUM_WORDS_ONES:
            current += _NUM_WORDS_ONES[w]
            matched_any = True
            continue
        # words like "dan", "yang" etc are ignored silently
    total += current
    return total if matched_any else None


_DATE_NUMERIC_RE = re.compile(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})\b")
_DATE_TEXTUAL_RE = re.compile(
    r"\b(\d{1,2})\s+([A-Za-zé]+)\s+(\d{4})\b", re.IGNORECASE
)


def parse_date_id(raw: str) -> tuple[Optional[str], str]:
    """Returns (iso8601_date_or_None, precision). Handles numeric and Indonesian-textual forms."""
    if not raw:
        return None, "none"

    m = _DATE_TEXTUAL_RE.search(raw)
    if m:
        day, month_name, year = m.groups()
        month = MONTHS_ID.get(month_name.lower())
        if month:
            try:
                return f"{int(year):04d}-{month:02d}-{int(day):02d}", "day"
            except ValueError:
                pass

    m = _DATE_NUMERIC_RE.search(raw)
    if m:
        a, b, c = m.groups()
        year = c if len(c) == 4 else f"20{c}"
        try:
            day, month = int(a), int(b)
            if month > 12 and day <= 12:
                day, month = month, day
            return f"{int(year):04d}-{month:02d}-{day:02d}", "day"
        except ValueError:
            pass

    m = re.search(r"\bTAHUN\s+ANGGARAN\s+(\d{4})\b", raw, re.IGNORECASE) or re.search(
        r"\b(\d{4})\b", raw
    )
    if m:
        return m.group(1), "year"

    return None, "none"


_RATE_PERMILLE_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*‰")
_RATE_PERCENT_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*%")
_RATE_FRACTION_RE = re.compile(r"(\d+)\s*/\s*(\d+)")


def parse_rate(raw: str) -> Optional[float]:
    """Unifies %, ‰, x/y forms to a decimal ratio."""
    if not raw:
        return None
    m = _RATE_PERMILLE_RE.search(raw)
    if m:
        return float(m.group(1).replace(",", ".")) / 1000.0
    m = _RATE_PERCENT_RE.search(raw)
    if m:
        return float(m.group(1).replace(",", ".")) / 100.0
    m = _RATE_FRACTION_RE.search(raw)
    if m:
        num, den = m.groups()
        try:
            return int(num) / int(den)
        except ZeroDivisionError:
            return None
    return None
