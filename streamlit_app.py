import requests
import streamlit as st
import urllib3
import pandas as pd
import os
import json
import datetime

# Add this near the top of your file, after the imports
def _merge_selected_exports(selected_ids):
    """Merge multiple selected exports into a single DataFrame"""
    all_dfs = []
    _hist = _load_history()
    
    for entry_id in selected_ids:
        entry = next((e for e in _hist if e.get('id') == entry_id), None)
        if entry and os.path.exists(entry.get('export_csv_path', '')):
            try:
                df = pd.read_csv(entry['export_csv_path'])
                df['_source_search_id'] = entry_id  # Track which search it came from
                all_dfs.append(df)
            except Exception as e:
                st.error(f"Error loading {entry_id}: {e}")
    
    if not all_dfs:
        st.warning("No valid exports selected or found")
        return None
    
    return pd.concat(all_dfs, ignore_index=True)

# Update the Search History section to include checkboxes and merge button
st.subheader("Search History")
_hist = _load_history()

if not _hist:
    st.caption("No search history yet. Perform a search to save it here.")
else:
    # Add a container for the merge button (will be shown if any items are selected)
    merge_container = st.container()
    
    # Initialize selected_ids in session state if not exists
    if 'selected_history_ids' not in st.session_state:
        st.session_state.selected_history_ids = []
    
    selected_ids = []
    
    for entry in reversed(_hist):
        event_name = entry.get('params', {}).get('event', 'Unnamed Event')
        date_str = entry.get('timestamp', 'Unknown date')
        row_count = entry.get('row_count', 0)
        
        # Create a unique key for the checkbox
        checkbox_key = f"select_{entry.get('id')}"
        
        # Create columns for layout
        col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
        
        with col1:
            # Add a checkbox for selection
            is_selected = st.checkbox(
                "Select",
                key=checkbox_key,
                value=entry.get('id') in st.session_state.selected_history_ids
            )
            if is_selected:
                selected_ids.append(entry.get('id'))
        
        with col2:
            if st.button("🔍", key=f"view_{entry.get('id')}"):
                _queue_load(entry.get('id'))
        
        with col3:
            if st.button("🗑️", key=f"del_{entry.get('id')}"):
                _queue_delete(entry.get('id'))
        
        with col4:
            # Show entry details in an expander
            with st.expander(f"{date_str} - {event_name} ({row_count} rows)", expanded=False):
                st.write(f"**Search Parameters:**")
                st.json(entry.get('params', {}))
                
                # Download button for individual export
                export_path = entry.get('export_csv_path')
                if export_path and os.path.exists(export_path):
                    with open(export_path, 'rb') as f:
                        st.download_button(
                            "💾 Download CSV",
                            f,
                            file_name=os.path.basename(export_path),
                            mime='text/csv',
                            key=f"dl_{entry.get('id')}",
                            use_container_width=True
                        )
    
    # Update the selected IDs in session state
    st.session_state.selected_history_ids = selected_ids
    
    # Show merge button if any items are selected
    if selected_ids:
        with merge_container:
            st.write("---")
            if st.button("📊 Merge Selected Exports", use_container_width=True):
                merged_df = _merge_selected_exports(selected_ids)
                if merged_df is not None:
                    # Generate a timestamp for the merged export
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"merged_export_{timestamp}.csv"
                    
                    # Create download button for merged export
                    csv = merged_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="💾 Download Merged CSV",
                        data=csv,
                        file_name=filename,
                        mime='text/csv',
                        use_container_width=True
                    )
                    
                    # Display a preview of the merged data
                    st.write("### Merged Data Preview")
                    st.dataframe(merged_df.head())

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

# Columns to display/save from results
ALLOWED_COLUMNS = [
    "id",
    "name",
    "date",
    "venue",
    "performer",
    "keywords",
    "stubhubEventId",
]

# Hidden defaults for API pagination
DEFAULT_LIMIT = 1000

# Helper: find a DataFrame column by case-insensitive name
_DEF_STUBHUB_COL = "stubhubEventId"

def _find_col_case_insensitive(df: pd.DataFrame, name: str):
    low = name.lower()
    for c in df.columns:
        if str(c).lower() == low:
            return c
    return None

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
    try:
        # Create timestamp and unique ID
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry_id = f"{int(datetime.datetime.now().timestamp())}"
        
        # Create filenames
        csv_filename = f"{entry_id}.csv"
        export_csv_filename = f"{entry_id}_export.csv"
        json_filename = f"{entry_id}.json"
        
        csv_path = os.path.join(HISTORY_DIR, csv_filename)
        export_csv_path = os.path.join(HISTORY_DIR, export_csv_filename)
        json_path = os.path.join(HISTORY_DIR, json_filename)
        
        # Save the main CSV
        df.to_csv(csv_path, index=False)
        
        # Prepare InHandAt dates (1 day before event date)
        date_col = None
        for col in df.columns:
            if str(col).lower() == 'date':
                date_col = col
                break

        if date_col and date_col in df.columns:
            try:
                in_hand_dates = (pd.to_datetime(df[date_col]) - pd.Timedelta(days=1)).dt.strftime('%Y-%m-%d')
                in_hand_at = in_hand_dates.tolist()
            except:
                in_hand_at = [''] * len(df)
        else:
            in_hand_at = [''] * len(df)

        # Update the export CSV with the specified columns
        export_df = pd.DataFrame({
            'DeliveryType': ['pdf'] * len(df),
            'TicketCount': ['4'] * len(df),
            'InHandAt': in_hand_at,  # Updated to use the calculated dates
            'Section': [params.get('_exportSection', 'RESERVED')] * len(df),
            'ROW': ['GA'] * len(df),
            'StubhubEventId': df[_DEF_STUBHUB_COL] if _DEF_STUBHUB_COL in df.columns else [0] * len(df),
            'UnitCost': [float(st.session_state.get('unit_cost', 800.0))] * len(df),
            'FaceValue': [''] * len(df),
            'AutoBroadcast': [True] * len(df),
            'SellerOwn': [False] * len(df),
            'ListingNotes': [''] * len(df),
        })
        export_df.to_csv(export_csv_path, index=False)
        
        # Save the raw data JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)
        
        # Get row count from raw if available, otherwise use DataFrame length
        row_count = raw.get("rows_count", len(df) if df is not None else 0)
        # Create history entry
        entry = {
            "id": entry_id,
            "timestamp": ts,
            "row_count": row_count,  # Use the more accurate row count
            "params": params,
            "csv_path": csv_path,
            "export_csv_path": export_csv_path,
            "json_path": json_path,
        }
        
        # Load existing history and add new entry
        items = _load_history()
        items.append(entry)
        
        # Save updated history
        with open(HISTORY_INDEX, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        
        return entry
        
    except Exception as e:
        print(f"Error saving search: {e}")
        st.error(f"Error saving search: {e}")
        return None

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
            _sec = it.get("params", {}).get("_exportSection")
            if isinstance(_sec, str) and _sec in ("RESERVED", "GA"):
                st.session_state['export_section'] = _sec
            if csv_path and os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                # Normalize nested fields to names for display/save
                for _c in ['venue', 'performer']:
                    if _c in df.columns:
                        def _extract_name(x):
                            try:
                                if isinstance(x, str) and (x.strip().startswith('{') or x.strip().startswith('[')):
                                    obj = json.loads(x)
                                else:
                                    obj = x
                                if isinstance(obj, dict) and 'name' in obj:
                                    return obj.get('name')
                                if isinstance(obj, list) and obj and isinstance(obj[0], dict) and 'name' in obj[0]:
                                    return obj[0].get('name')
                            except Exception:
                                return x
                            return x
                        df[_c] = df[_c].apply(_extract_name)
                # Keep only numeric and non-zero stubhubEventId; exclude nulls and 0s.
                _col = _find_col_case_insensitive(df, _DEF_STUBHUB_COL)
                if _col is not None:
                    _s = pd.to_numeric(df[_col], errors='coerce')
                    df = df[_s.notna() & (_s != 0)].copy()
                # Now reduce to allowed columns for display/save
                allowed_cols = [c for c in ALLOWED_COLUMNS if c in df.columns]
                if allowed_cols:
                    df = df[allowed_cols].copy()
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
                st.session_state['raw_data'] = None
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

def _update_entry_section(entry_id, section):
    """Update the export section for a history entry"""
    items = _load_history()
    for it in items:
        if str(it.get("id")) == str(entry_id):
            if 'params' not in it:
                it['params'] = {}
            it['params']['_exportSection'] = section
            
            # Update the export file with new settings
            export_csv_path = it.get('export_csv_path')
            if export_csv_path and os.path.exists(export_csv_path):
                try:
                    export_df = pd.read_csv(export_csv_path)
                    export_df['Section'] = section
                    export_df['UnitCost'] = float(st.session_state.get('unit_cost', 800.0))
                    export_df.to_csv(export_csv_path, index=False)
                except Exception as e:
                    print(f"Error updating export file {export_csv_path}: {e}")
            break
    _persist_history(items)

with st.sidebar:
    st.header("Search Parameters")
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
            _col = _find_col_case_insensitive(selected_full, _DEF_STUBHUB_COL)
            if _col and _col in selected_full.columns:
                sh_series = selected_full[_col]
                export_df = pd.DataFrame({
                    'DeliveryType': ['pdf'] * len(selected_full),
                    'TicketCount': [''] * len(selected_full),
                    'InHandAt': [''] * len(selected_full),
                    'Section': [st.session_state.get('export_section', 'RESERVED')] * len(selected_full),
                    'ROW': ['GA'] * len(selected_full),
                    'StubhubEventId': sh_series,
                    'UnitCost': [st.session_state.get('unit_cost', 800)] * len(selected_full),
                    'FaceValue': [''] * len(selected_full),
                    'AutoBroadcast': [True] * len(selected_full),
                    'SellerOwn': [False] * len(selected_full),
                    'ListingNotes': [''] * len(selected_full),
                })
                csv_bytes = export_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label='Download export CSV',
                    data=csv_bytes,
                    file_name='export.csv',
                    mime='text/csv'
                )

st.subheader("Export Settings")
with st.form(key="export_settings_form"):
    col1, col2 = st.columns(2)
    
    # Section selection
    with col1:
        saved_section = st.session_state.get('export_section', 'RESERVED')
        new_section = st.radio(
            "Section",
            options=["RESERVED", "GA"],
            index=(0 if saved_section == 'RESERVED' else 1),
            horizontal=True,
            key="section_global"
        )

    # Unit cost input
    with col2:
        unit_cost = st.number_input(
            "Unit Cost (USD)",
            min_value=0.0,
            step=10.0,
            value=float(st.session_state.get('unit_cost', 800.0)),
            key="unit_cost_global"
        )
        st.session_state['unit_cost'] = float(unit_cost)
    
    # Update button
    if st.form_submit_button("Update All Export Settings"):
        st.session_state['export_section'] = new_section
        
        # Update all history entries with the new settings
        items = _load_history()
        for entry in items:
            # Update the entry's section in the history
            if 'params' not in entry:
                entry['params'] = {}
            entry['params']['_exportSection'] = new_section
            
            # Update the export file
            export_csv_path = entry.get('export_csv_path')
            if export_csv_path and os.path.exists(export_csv_path):
                try:
                    export_df = pd.read_csv(export_csv_path)
                    export_df['Section'] = new_section
                    export_df['UnitCost'] = float(unit_cost)
                    export_df.to_csv(export_csv_path, index=False)
                except Exception as e:
                    print(f"Error updating export file {export_csv_path}: {e}")
        
        _persist_history(items)
        st.success("Export settings updated for all entries!")

st.subheader("Search History")
_hist = _load_history()
if not _hist:
    st.caption("No search history yet. Perform a search to save it here.")
else:
    for entry in reversed(_hist):
        # Create a more descriptive title
        event_name = entry.get('params', {}).get('event', 'Unnamed Event')
        date_str = entry.get('timestamp', 'Unknown date')
        row_count = entry.get('row_count', 0)
        
        with st.expander(f"{date_str} - {event_name} ({row_count} rows)", expanded=False):
            st.write(f"**Search Parameters:**")
            st.json(entry.get('params', {}))
            
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                if st.button("🔍 Load", key=f"load_{entry.get('id')}"):
                    _queue_load(entry.get('id'))
            
            with col2:
                if st.button("🗑️ Delete", key=f"del_{entry.get('id')}"):
                    _queue_delete(entry.get('id'))
            
            # Download button
            export_path = entry.get('export_csv_path')
            if export_path and os.path.exists(export_path):
                with open(export_path, 'rb') as f:
                    st.download_button(
                        "💾 Download CSV",
                        f,
                        file_name=os.path.basename(export_path),
                        mime='text/csv',
                        key=f"dl_{entry.get('id')}",
                        use_container_width=True
                    )

if run:
    try:
        all_rows = []
        with st.spinner("Loading results..."):
            status = st.empty()
            page = 1
            last_url = None
            while True:
                status.write(f"Fetching page {page}...")
                q = dict(params)
                q['pageNumber'] = page
                q['limit'] = DEFAULT_LIMIT
                resp = requests.get(url, headers=HEADERS, params=q, verify=False, timeout=30)
                last_url = resp.url
                if resp.status_code != 200:
                    st.error(f"Request failed on page {page}: {resp.status_code}")
                    break
                if "application/json" not in resp.headers.get("Content-Type", ""):
                    st.warning("Non-JSON response received")
                    break
                data = resp.json()
                rows = data.get("rows") if isinstance(data, dict) else None
                if not rows:
                    break
                if isinstance(rows, list):
                    all_rows.extend(rows)
                page += 1
            status.empty()

        if last_url:
            st.caption(last_url)
        if all_rows:
            df = pd.DataFrame(all_rows)
            
            # Filter out past events (only keep events from today forward)
            date_col = _find_col_case_insensitive(df, 'date')
            if date_col and date_col in df.columns:
                try:
                    # Convert date strings to datetime for comparison
                    today = pd.Timestamp.now().normalize()  # Get today's date at midnight
                    df['_date_parsed'] = pd.to_datetime(df[date_col], errors='coerce')
                    # Keep only rows where date is today or in the future
                    df = df[df['_date_parsed'].notna() & (df['_date_parsed'] >= today)].copy()
                    df = df.drop(columns=['_date_parsed'])  # Clean up temporary column
                except Exception as e:
                    st.warning(f"Could not filter by date: {str(e)}")
            
            # Normalize nested fields to names for display/save
            for _c in ['venue', 'performer']:
                if _c in df.columns:
                    def _extract_name(x):
                        try:
                            if isinstance(x, str) and (x.strip().startswith('{') or x.strip().startswith('[')):
                                obj = json.loads(x)
                            else:
                                obj = x
                            if isinstance(obj, dict) and 'name' in obj:
                                return obj.get('name')
                            if isinstance(obj, list) and obj and isinstance(obj[0], dict) and 'name' in obj[0]:
                                return obj[0].get('name')
                        except Exception:
                            return x
                        return x
                    df[_c] = df[_c].apply(_extract_name)
            # Keep only numeric and non-zero stubhubEventId; exclude nulls and 0s.
            _col = _find_col_case_insensitive(df, _DEF_STUBHUB_COL)
            if _col is not None:
                _s = pd.to_numeric(df[_col], errors='coerce')
                df = df[_s.notna() & (_s != 0)].copy()
            # Now reduce to allowed columns for display/save
            allowed_cols = [c for c in ALLOWED_COLUMNS if c in df.columns]
            if allowed_cols:
                df = df[allowed_cols].copy()
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
            st.session_state['raw_data'] = {"rows_count": len(all_rows)}
            st.session_state['key_col_name'] = key_col
            try:
                params_to_save = dict(params)
                params_to_save["_exportSection"] = st.session_state.get('export_section', 'RESERVED')
                _save_search(params_to_save, df.drop(columns=['select']), {"rows_count": len(df)})
            except Exception:
                pass
            st.rerun()
        else:
            st.session_state['rows_df'] = None
            st.session_state['raw_data'] = None
            st.rerun()
    except requests.exceptions.RequestException as e:
        st.error(f"Error: {e}")