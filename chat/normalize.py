# normalize.py
from rapidfuzz import process, fuzz
import pycountry

DEPT_SYNONYMS = {
    "it": "Information Technology",
    "information technology": "Information Technology",
    "tech": "Information Technology",
    "technology": "Information Technology",
    "hr": "Human Resources",
    "people": "Human Resources",
    "cust success": "Customer Success",
    "cs": "Customer Success",
    "ops": "Operations",
    "finance": "Finance",
    "marketing": "Marketing",
    "sales": "Sales",
    "product": "Product",
    "legal": "Legal"
}


ROLE_GROUPS = {
    "decision makers": ["C-Suite", "EVP/VP", "General Manager", "Director", "Manager"],
    "leadership": ["C-Suite", "President", "EVP/VP", "General Manager", "Director"],
    "executives": ["C-Suite", "President", "EVP/VP"],
    "students": ["Students and Interns"],
    "founders": ["Owner/Founder"],
    "board": ["Board Member"]
}

SENIORITY_LIST = [
    "Board Member", "Owner/Founder", "C-Suite", "President", "EVP/VP",
    "General Manager", "Director", "Manager", "IC Leader",
    "Individual Contributor", "Retired / Disabled", "Self-Employed",
    "Students and Interns", "Unemployed", "Volunteer"
]

def expand_seniority(value: str):
    if not value:
        return []

    val = value.lower().strip()

    # Expand using predefined groups
    if val in ROLE_GROUPS:
        return ROLE_GROUPS[val]

    # Fuzzy match if it’s a single known value
    match, score, _ = process.extractOne(value, SENIORITY_LIST, scorer=fuzz.WRatio)
    return [match] if score >= 80 else [value]


def normalize_filters(filters: dict):
    """
    Expand seniority groups like 'decision makers' into multiple levels,
    and normalize values.
    """
    if filters.get("Seniority"):
        filters["Seniority"] = expand_seniority(filters["Seniority"])
    return filters

def normalize_department(text: str) -> str | None:
    if not text: return None
    t = text.strip().lower()
    if t in DEPT_SYNONYMS: return DEPT_SYNONYMS[t]
    # fuzzy to known departments
    departments = list({v for v in DEPT_SYNONYMS.values()})
    match, score, _ = process.extractOne(t, departments, scorer=fuzz.WRatio)
    return match if score >= 80 else text

def normalize_seniority(text: str) -> str | None:
    if not text: return None
    match, score, _ = process.extractOne(text, SENIORITY_LIST, scorer=fuzz.WRatio)
    return match if score >= 80 else text

def normalize_country(name: str) -> str | None:
    if not name: return None
    t = name.strip()
    # common shorthands
    aliases = {
        "us":"United States","usa":"United States","u.s.":"United States","u.s.a":"United States",
        "uk":"United Kingdom","uae":"United Arab Emirates","korea":"Korea",
        "south korea":"Korea", "north korea":"Korea", "china":"China", "india":"India",
        "japan":"Japan", "germany":"Germany", "france":"France",
        "canada":"Canada", "brazil":"Brazil", "mexico":"Mexico", "australia":"Australia",
        "russia":"Russia", "south africa":"South Africa", "spain":"Spain",
        "italy":"Italy", "netherlands":"Netherlands", "sweden":"Sweden", "norway":"Norway",
        "finland":"Finland", "denmark":"Denmark", "belgium":"Belgium",
        "switzerland":"Switzerland", "austria":"Austria", "poland":"Poland",
    }
    low = t.lower().replace(".", "")
    if low in aliases: return aliases[low]
    # pycountry lookup
    try:
        c = pycountry.countries.lookup(t)
        return c.name
    except Exception:
        return t

def build_location_string(country: str|None, state: str|None, city: str|None) -> str|None:
    parts = [p for p in [city, state, country] if p]
    return ", ".join(parts) if parts else None
