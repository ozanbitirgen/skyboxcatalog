import requests
import streamlit as st
import urllib3
import pandas as pd
import os
import json
import datetime

# API Headers
HEADERS = {
    'X-Account': '3521',
    'X-Api-Token': 'f1b1f800-5483-4a9a-9fc0-7419dbeb0018',
    'X-Application-Token': 'ddbc62c3-000c-4741-82c7-c8743a55d5ea',
    'accept': 'application/json'
}

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(page_title="Skybox Event Catalog Search", layout="wide")

url = "https://skybox.vividseats.com/services/event-catalog/search"

HISTORY_DIR = os.path.join(os.path.dirname(__file__), "search_history")
HISTORY_INDEX = os.path.join(HISTORY_DIR, "index.json")

def _ensure_history_dir():
    os.makedirs(HISTORY_DIR, exist_ok=True)
    if not os.path.exists(HISTORY_INDEX):
        with open(HISTORY_INDEX, "w", encoding="utf-8") as f:
            json.dump([], f)

def _load_history():
    _ensure_history_dir()
    try:
        with open(HISTORY_INDEX, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
    except Exception:
        return []

def _persist_history(items):
    _ensure_history_dir()
    with open(HISTORY_INDEX, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

def _save_search(params, df, raw):
    _ensure_history_dir()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    entry_id = f"{int(datetime.datetime.now().timestamp())}_{ts}"
    # Build filename as: eventDate - event (fallback to keywords or 'search')
    date_label = params.get('eventDateFrom') or params.get('eventDateTo') or datetime.datetime.now().strftime("%Y-%m-%d")
    evt = params.get('event')
    kw = params.get('keywords')
    if isinstance(evt, str) and evt.strip():
        event_label = evt.strip()
    elif isinstance(kw, list) and kw:
        event_label = " ".join(str(x) for x in kw[:5])  # limit to first 5 keywords
    else:
        event_label = "search"
    # Sanitize components for filesystem safety, allow letters, digits, space, dash, underscore
    def _clean(s: str) -> str:
        s = s.replace("/", "-").replace("\\", "-")
        return "".join(c for c in s if c.isalnum() or c in (" ", "-", "_"))
    date_label = _clean(str(date_label)).strip() or datetime.datetime.now().strftime("%Y-%m-%d")
    event_label = _clean(str(event_label)).strip() or "search"
    # Trim overly long event labels
    event_label = event_label[:60]
    # Build keywords label from params['keywords'] if present, else literal 'keywords'
    kw_list = kw if isinstance(kw, list) else []
    if kw_list:
        kw_label = _clean(" ".join(str(x) for x in kw_list[:5])).strip()
        kw_label = kw_label[:60] or "keywords"
    else:
        kw_label = "keywords"
    csv_name = f"{date_label} - {event_label} - {kw_label}.csv"
    json_name = f"{date_label} - {event_label} - {kw_label}.json"
    csv_path = os.path.join(HISTORY_DIR, csv_name)
    json_path = os.path.join(HISTORY_DIR, json_name)
    try:
        df.to_csv(csv_path, index=False)
    except Exception:
        df.reset_index(drop=True).to_csv(csv_path, index=False)
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False)
    except Exception:
        pass
    items = _load_history()
    entry = {
        "id": entry_id,
        "timestamp": ts,
        "row_count": int(len(df)),
        "params": params,
        "csv_path": csv_path,
        "json_path": json_path,
    }
    items.append(entry)
    _persist_history(items)
    return entry

def _delete_search(entry_id):
    items = _load_history()
    keep = []
    for it in items:
        if str(it.get("id")) == str(entry_id):
            for p in [it.get("csv_path"), it.get("json_path")]:
                try:
                    if p and os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass
        else:
            keep.append(it)
    _persist_history(keep)

def _load_saved_entry(entry_id):
    items = _load_history()
    for it in items:
        if str(it.get("id")) == str(entry_id):
            csv_path = it.get("csv_path")
            json_path = it.get("json_path")
            if csv_path and os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                if 'select' not in df.columns:
                    df.insert(0, 'select', False)
                key_col = None
                for cand in ['eventId', 'id', 'event_id', 'eventID']:
                    if cand in df.columns:
                        key_col = cand
                        break
                if key_col is None:
                    if 'row_id' not in df.columns:
                        df.insert(1, 'row_id', range(1, len(df) + 1))
                    key_col = 'row_id'
                df.set_index(key_col, drop=False, inplace=True)
                df.index.name = 'row_idx'
                st.session_state['rows_df'] = df
                st.session_state['key_col_name'] = key_col
            if json_path and os.path.exists(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        st.session_state['raw_data'] = json.load(f)
                except Exception:
                    st.session_state['raw_data'] = None
            return True
    return False

def _queue_load(entry_id):
    st.session_state['pending_load_entry'] = entry_id

def _queue_delete(entry_id):
    st.session_state['pending_delete_entry'] = entry_id

with st.sidebar:
    st.header("Search Parameters")
    limit = st.number_input("limit (max 1000)", min_value=1, max_value=1000, value=100, step=50)
    sortDir = st.selectbox("sortDir", options=["asc", "desc"], index=0)
    pageNumber = st.number_input("pageNumber", min_value=1, value=1, step=1)
    event = st.text_input("event", value="")
    eventType = st.selectbox("eventType", options=["", "Concert", "Theater", "Sports", "Other"], index=0)
    venue = st.text_input("venue", value="")
    city = st.text_input("city", value="")
    eventDateFrom = st.date_input("eventDateFrom", value=None, format="YYYY-MM-DD")
    eventDateTo = st.date_input("eventDateTo", value=None, format="YYYY-MM-DD")
    keywords_text = st.text_area("keywords (comma-separated)", value="")
    excludeParking = st.checkbox("excludeParking", value=False)
    run = st.button("Search")

params = {}
params["limit"] = min(int(limit), 1000)
params["sortDir"] = sortDir
params["pageNumber"] = int(pageNumber)
if event:
    params["event"] = event
if eventType:
    params["eventType"] = eventType
if venue.strip():
    params["venue"] = venue
if city.strip():
    params["city"] = city
if eventDateFrom:
    params["eventDateFrom"] = eventDateFrom.strftime("%Y-%m-%d")
if eventDateTo:
    params["eventDateTo"] = eventDateTo.strftime("%Y-%m-%d")
keywords = [x.strip() for x in keywords_text.split(",") if x.strip()]
if keywords:
    params["keywords"] = keywords
params["excludeParking"] = str(bool(excludeParking)).lower()

st.title("Skybox Event Catalog Search")

# Initialize persistent state for results
if 'rows_df' not in st.session_state:
    st.session_state['rows_df'] = None
if 'raw_data' not in st.session_state:
    st.session_state['raw_data'] = None

# Handle queued actions before rendering anything else to avoid multi-clicks
_pending_load = st.session_state.get('pending_load_entry')
if _pending_load:
    st.session_state.pop('pending_load_entry', None)
    if _load_saved_entry(_pending_load):
        st.rerun()
_pending_delete = st.session_state.get('pending_delete_entry')
if _pending_delete:
    st.session_state.pop('pending_delete_entry', None)
    _delete_search(_pending_delete)
    st.rerun()

# Always render current results if present (even when Search button isn't pressed)
if st.session_state['rows_df'] is not None:
    df_full = st.session_state['rows_df']

    # Controls for performance
    col_a, col_b, col_c = st.columns([1,1,2])
    with col_a:
        page_size = st.selectbox('Rows per page', options=[25, 50, 100, 200], index=1, key='page_size')
    with col_b:
        total_pages = max(1, (len(df_full) + page_size - 1) // page_size)
        page = st.number_input('Page', min_value=1, max_value=total_pages, value=1, step=1, key='page_num')
    with col_c:
        show_selected_only = st.checkbox('Show only selected', value=False, key='show_selected_only')

    if show_selected_only and 'select' in df_full.columns:
        view_df = df_full[df_full['select']]
        start = 0
        end = len(view_df)
    else:
        start = (page - 1) * page_size
        end = min(start + page_size, len(df_full))
        view_df = df_full.iloc[start:end]

    editor_key = f"rows_editor_{page}_{int(show_selected_only)}"
    edited_page = st.data_editor(
        view_df.copy(),
        column_config={
            'select': st.column_config.CheckboxColumn('select', help='Select rows to export', default=False)
        } if 'select' in view_df.columns else {},
        disabled=[c for c in view_df.columns if c != 'select'],
        column_order=['select'] + [c for c in view_df.columns if c != 'select'],
        num_rows="fixed",
        width='stretch',
        hide_index=True,
        key=editor_key
    )

    # Persist only the edited 'select' values back to the full DataFrame
    df_full = st.session_state['rows_df']
    key_col_name = st.session_state.get('key_col_name')

    if isinstance(edited_page, pd.DataFrame) and 'select' in edited_page.columns:
        if key_col_name and key_col_name in edited_page.columns and key_col_name in df_full.columns:
            left = edited_page[[key_col_name, 'select']].copy()
            sel_map = dict(zip(left[key_col_name].astype(str), left['select']))
            df_full_keys = df_full[key_col_name].astype(str)
            updated = df_full_keys.map(sel_map)
            df_full.loc[updated.notna(), 'select'] = updated[updated.notna()].astype(bool)
        else:
            df_full.loc[edited_page.index, 'select'] = edited_page['select']
        st.session_state['rows_df'] = df_full
    elif isinstance(edited_page, dict):
        # Fallback for Streamlit versions where session value is a dict with deltas
        for r in edited_page.get('edited_rows', []):
            idx = r.get('index')
            val = r.get('value', {}).get('select')
            if idx is not None and val is not None:
                try:
                    df_full.loc[idx, 'select'] = bool(val)
                except Exception:
                    # If idx is not in index, try aligning by key column
                    if key_col_name and key_col_name in df_full.columns:
                        mask = df_full[key_col_name].astype(str) == str(idx)
                        df_full.loc[mask, 'select'] = bool(val)
        st.session_state['rows_df'] = df_full

    # Download selected (from the full DataFrame)
    if 'select' in st.session_state['rows_df'].columns:
        selected_full = st.session_state['rows_df'][st.session_state['rows_df']['select']]
        st.caption(f"Selected rows: {len(selected_full)} of {len(st.session_state['rows_df'])}")
        if not selected_full.empty:
            csv_bytes = selected_full.drop(columns=['select']).to_csv(index=False).encode('utf-8')
            st.download_button(
                label='Download selected as CSV',
                data=csv_bytes,
                file_name='selected_rows.csv',
                mime='text/csv'
            )


st.subheader("Search history")
_hist = _load_history()
if not _hist:
    st.caption("No history yet")
else:
    for idx, entry in enumerate(reversed(_hist)):
        name = os.path.splitext(os.path.basename(entry.get('csv_path', '')))[0] or entry.get('timestamp', '')
        title = f"{name} - rows: {entry.get('row_count', 0)}"
        with st.expander(title, expanded=False):
            st.text("Parameters:")
            st.code(json.dumps(entry.get('params', {}), ensure_ascii=False, indent=2))
            c1, c2, c3 = st.columns([1,1,2])
            with c1:
                st.button("Load", key=f"load_{entry.get('id')}", on_click=_queue_load, args=(entry.get('id'),))
            with c2:
                st.button("Delete", key=f"del_{entry.get('id')}", on_click=_queue_delete, args=(entry.get('id'),))
            with c3:
                st.write(name)
                if os.path.exists(entry.get('csv_path', '')):
                    with open(entry.get('csv_path'), 'rb') as f:
                        st.download_button(
                            label='Download CSV',
                            data=f.read(),
                            file_name=os.path.basename(entry.get('csv_path')),
                            mime='text/csv',
                            key=f"dl_{entry.get('id')}"
                        )

if run:
    try:
        resp = requests.get(url, headers=HEADERS, params=params, verify=False, timeout=30)
        st.caption(resp.url)
        if resp.status_code != 200:
            st.error(f"Request failed: {resp.status_code}")
        else:
            if "application/json" in resp.headers.get("Content-Type", ""):
                data = resp.json()
                rows = data.get("rows") if isinstance(data, dict) else None
                if rows:
                    df = pd.DataFrame(rows)
                    if 'select' not in df.columns:
                        df.insert(0, 'select', False)
                    # Set a stable unique key for reliable row editing
                    key_col = None
                    for cand in ['eventId', 'id', 'event_id', 'eventID']:
                        if cand in df.columns:
                            key_col = cand
                            break
                    if key_col is None:
                        if 'row_id' not in df.columns:
                            df.insert(1, 'row_id', range(1, len(df) + 1))
                        key_col = 'row_id'
                    # Preserve key both as a column and as index for readability; use column for alignment
                    df.set_index(key_col, drop=False, inplace=True)
                    df.index.name = 'row_idx'
                    # Update session state so data persists across reruns
                    st.session_state['rows_df'] = df
                    st.session_state['raw_data'] = data
                    st.session_state['key_col_name'] = key_col
                    try:
                        _save_search(params, df.drop(columns=['select']), data)
                    except Exception:
                        pass
                    st.rerun()
                else:
                    st.session_state['rows_df'] = None
                    st.session_state['raw_data'] = data
                    st.rerun()
            else:
                st.warning("Non-JSON response received")
    except requests.exceptions.RequestException as e:
        st.error(f"Error: {e}")