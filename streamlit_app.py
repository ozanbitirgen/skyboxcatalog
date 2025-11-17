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
       # In the _save_search function, after creating the entry:
        entry = {
            "id": entry_id,
            "timestamp": ts,
            "row_count": row_count,
            "params": params,
            "csv_path": csv_path,
            "export_csv_path": export_csv_path,
            "json_path": json_path,
        }

        # Store the current search ID in session state
        st.session_state['current_search_id'] = entry_id
        
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

def _update_export_settings(entry_id, section, unit_cost):
    """Update export settings for a specific history entry"""
    items = _load_history()
    for item in items:
        if str(item.get("id")) == str(entry_id):
            if "params" not in item:
                item["params"] = {}
            item["params"]["_exportSection"] = section
            item["params"]["_unitCost"] = unit_cost
            break
    _persist_history(items)

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

def _clear_search_history():
    """Clear all search history and related files"""
    try:
        history = _load_history()
        # Delete all history files
        for entry in history:
            for path_key in ['csv_path', 'export_csv_path', 'json_path']:
                path = entry.get(path_key)
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass
        # Clear the history index
        _persist_history([])
        # Clear any selected entries
        if 'selected_entries' in st.session_state:
            st.session_state.selected_entries = set()
        return True
    except Exception as e:
        st.error(f"Error clearing search history: {e}")
        return False

def _load_saved_entry(entry_id):
    items = _load_history()
    # In the _load_saved_entry function, after loading the entry:
    st.session_state['current_search_id'] = entry_id
    for it in items:
        if str(it.get("id")) == str(entry_id):
            try:
                # Restore export settings
                params = it.get('params', {})
                st.session_state['export_section'] = params.get('_exportSection', 'RESERVED')
                st.session_state['unit_cost'] = float(params.get('_unitCost', 800))
                
                csv_path = os.path.join(HISTORY_DIR, f"{entry_id}.csv")
                json_path = os.path.join(HISTORY_DIR, f"{entry_id}.json")
                
                if os.path.exists(csv_path):
                    # Load and process the CSV
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
                    
                    # Add selection columns
                    if 'select' not in df.columns:
                        df.insert(0, 'select', False)
                    if 'selected' not in df.columns:
                        df.insert(0, 'selected', False)
                    
                    # Set up the index
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
                    
                    # Update session state
                    st.session_state['rows_df'] = df
                    st.session_state['raw_data'] = None
                    st.session_state['key_col_name'] = key_col
                    st.session_state['search_performed'] = True
                    
                    # Load JSON data if it exists
                    if os.path.exists(json_path):
                        try:
                            with open(json_path, "r", encoding="utf-8") as f:
                                st.session_state['raw_data'] = json.load(f)
                        except Exception:
                            st.session_state['raw_data'] = None
                    
                    return True  # Success
                    
            except Exception as e:
                st.error(f"Error loading saved search: {e}")
                return False
    return False

def _queue_load(entry_id):
    """Load a saved search entry"""
    if _load_saved_entry(entry_id):
        st.rerun()

def _queue_delete(entry_id):
    st.session_state['pending_delete_entry'] = entry_id

with st.sidebar:
    st.header("Search Parameters")
    with st.form(key='search_form'):
        event = st.text_input("event", value="", key="event_input")
        eventType = st.selectbox("eventType", options=["", "Concert", "Theater", "Sports", "Other"], index=0)
        venue = st.text_input("venue", value="")
        city = st.text_input("city", value="")
        eventDateFrom = st.date_input("eventDateFrom", value=None, format="YYYY-MM-DD")
        eventDateTo = st.date_input("eventDateTo", value=None, format="YYYY-MM-DD")
        keywords_text = st.text_area("keywords (comma-separated)", value="")
        excludeParking = st.checkbox("excludeParking", value=False)

        run = st.form_submit_button("Search")

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
# Create a container at the top for the loading indicator
loading_container = st.empty()
# Initialize persistent state for results
if 'rows_df' not in st.session_state:
    st.session_state['rows_df'] = None
if 'raw_data' not in st.session_state:
    st.session_state['raw_data'] = None

_pending_delete = st.session_state.get('pending_delete_entry')
if _pending_delete:
    st.session_state.pop('pending_delete_entry', None)
    _delete_search(_pending_delete)
    st.rerun()

# Always render current results if present (even when Search button isn't pressed)
# In the "Always render current results if present" section, replace with:
if st.session_state['rows_df'] is not None:
    df_full = st.session_state['rows_df'].copy()
    
    # Add 'selected' column if it doesn't exist
    if 'selected' not in df_full.columns:
        df_full['selected'] = False
    
    # Controls for pagination and filtering
    col_a, col_b, col_c = st.columns([1,1,2])
    with col_a:
        page_size = st.selectbox('Rows per page', options=[25, 50, 100, 200], index=1, key='page_size')
    with col_b:
        total_pages = max(1, (len(df_full) + page_size - 1) // page_size)
        page = st.number_input('Page', min_value=1, max_value=total_pages, value=1, step=1, key='page_num')
    with col_c:
        show_selected_only = st.checkbox('Show only selected', value=False, key='show_selected_only')
    
    # Filter data based on selection
    if show_selected_only and 'select' in df_full.columns:
        view_df = df_full[df_full['select']].copy()
    else:
        start = (page - 1) * page_size
        end = min(start + page_size, len(df_full))
        view_df = df_full.iloc[start:end].copy()
    
    # Move this before the data_editor
    column_order = ['selected'] + [c for c in view_df.columns if c != 'selected']
    view_df = view_df[column_order]

    # Then create the data_editor
    with st.form('data_editor_form'):
        # In the data_editor configuration, update the column_config to hide the 'select' column:
        edited_df = st.data_editor(
            view_df,
            column_config={
                'selected': st.column_config.CheckboxColumn('Select for Deletion', default=False),
                'select': None  # This hides the 'select' column
            },
            disabled=[c for c in view_df.columns if c != 'selected'],
            hide_index=True,
            use_container_width=True,
            height=400,
            key=f'data_editor_{page}_{int(show_selected_only)}'
        )
        
        # Add buttons for actions
        col1, col2 = st.columns([1, 1])
        with col1:
            # In the 'Delete Selected Rows' button click handler, update it to:
            if st.form_submit_button('🗑️ Delete Selected Rows'):
                # Get the indices of selected rows in the full dataframe
                selected_indices = edited_df[edited_df['selected']].index
                if len(selected_indices) > 0:
                    # Remove the selected rows from the full dataframe
                    df_full = df_full.drop(selected_indices).reset_index(drop=True)
                    
                    # Remove the 'selected' column if no more rows left
                    if len(df_full) == 0:
                        df_full = df_full.drop(columns=['selected'], errors='ignore')
                    
                    # Update the session state
                    st.session_state['rows_df'] = df_full
                    
                    # Update the history file with the remaining rows
                    try:
                        history = _load_history()
                        current_entry_id = str(st.session_state.get('current_search_id'))
                        
                        for entry in history:
                            if str(entry.get('id')) == current_entry_id:
                                # Update the CSV file with remaining rows
                                csv_path = entry.get('csv_path')
                                if csv_path and os.path.exists(csv_path):
                                    df_full.to_csv(csv_path, index=False)
                                
                                # Update the row count in the history entry
                                entry['row_count'] = len(df_full)
                                break
                        
                        _persist_history(history)
                    except Exception as e:
                        st.error(f"Error updating history: {e}")
                    
                    st.rerun()
        
        with col2:
            # Keep the export functionality
            pass
    
    # Update the full dataframe with any changes from the editor
    if 'select' in view_df.columns:
        # Update the select column in the full dataframe
        for idx, row in edited_df.iterrows():
            if idx in df_full.index:
                df_full.at[idx, 'select'] = row['select']
                df_full.at[idx, 'selected'] = row['selected']
        st.session_state['rows_df'] = df_full

    # Download selected (from the full DataFrame)
    if 'select' in st.session_state['rows_df'].columns:
        selected_full = st.session_state['rows_df'][st.session_state['rows_df']['select']]
        st.caption(f"Selected rows: {len(selected_full)} of {len(st.session_state['rows_df'])}")
        if not selected_full.empty:
            _col = _find_col_case_insensitive(selected_full, _DEF_STUBHUB_COL)
            if _col and _col in selected_full.columns:
                sh_series = selected_full[_col]
                # Replace the export_df creation block with this:
                export_df = pd.DataFrame({
                    'DeliveryType': ['pdf'] * len(selected_full),
                    'TicketCount': [''] * len(selected_full),
                    'InHandAt': [''] * len(selected_full),
                    'Section': [st.session_state.get('export_section', 'RESERVED')] * len(selected_full),
                    'ROW': ['GA'] * len(selected_full),
                    'StubhubEventId': sh_series,
                    'UnitCost': [float(st.session_state.get('unit_cost', 800))] * len(selected_full),
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
    # Initialize session state for confirmation
    if 'confirm_clear' not in st.session_state:
        st.session_state.confirm_clear = False
# Initialize selected_entries in session state if it doesn't exist
if 'selected_entries' not in st.session_state:
    st.session_state.selected_entries = set()

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
        entry_id = entry.get('id')
        
        # Create a checkbox for selection
        is_checked = entry_id in st.session_state.selected_entries
        if st.checkbox(f"{date_str} - {event_name} ({row_count} rows)",
                      value=is_checked,
                      key=f"select_{entry_id}",
                      on_change=lambda eid=entry_id, checked=is_checked: (
                          st.session_state.selected_entries.add(eid) 
                          if checked and eid not in st.session_state.selected_entries 
                          else st.session_state.selected_entries.discard(eid)
                      ) if checked else None):
            st.session_state.selected_entries.add(entry_id)
        else:
            st.session_state.selected_entries.discard(entry_id) if entry_id in st.session_state.selected_entries else None
        
        with st.expander("Details", expanded=False):
            st.write(f"**Search Parameters:**")
            st.json(entry.get('params', {}))
            
            # Add export settings editor
            st.write("**Export Settings**")
            current_section = entry.get('params', {}).get('_exportSection', 'RESERVED')
            current_unit_cost = entry.get('params', {}).get('_unitCost', 800.0)
            
            with st.form(key=f"export_settings_{entry_id}"):
                new_section = st.selectbox(
                    "Section",
                    options=["RESERVED", "GA"],
                    index=0 if current_section == "RESERVED" else 1,
                    key=f"section_{entry_id}"
                )
                new_unit_cost = st.number_input(
                    "Unit Cost",
                    value=float(current_unit_cost),
                    step=1.0,
                    key=f"unit_cost_{entry_id}"
                )
                
                if st.form_submit_button("💾 Update Export Settings", use_container_width=True):
                    _update_export_settings(entry_id, new_section, new_unit_cost)
                    st.session_state.success_message = "✅ Export settings updated!"
                    st.session_state.success_message_id = entry_id
                    st.rerun()
                
                # Show success message if it exists for this entry
                if st.session_state.get('success_message') and st.session_state.get('success_message_id') == entry_id:
                    st.success(st.session_state.success_message)
                    # Clear the message after showing it
                    st.session_state.success_message = None
                    st.session_state.success_message_id = None
            
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                if st.button("🔍 Load", key=f"load_{entry_id}", on_click=_queue_load, args=(entry_id,)):
                    pass  # The on_click handler will handle the loading
            
            with col2:
                if st.button("🗑️ Delete", key=f"del_{entry_id}"):
                    _delete_search(entry_id)
                    st.rerun()
            
            # Download button
            export_path = entry.get('export_csv_path')
            if export_path and os.path.exists(export_path):
                with open(export_path, 'rb') as f:
                    st.download_button(
                        "💾 Download CSV",
                        f,
                        file_name=os.path.basename(export_path),
                        mime='text/csv',
                        key=f"dl_{entry_id}",
                        use_container_width=True
                    )
    
    # Add export selected button at the bottom
    if st.session_state.selected_entries:
        if st.button("📤 Export Selected", use_container_width=True, type="primary"):
            selected_exports = [e for e in _hist if e.get('id') in st.session_state.selected_entries]
            if selected_exports:
                all_dfs = []
                for entry in selected_exports:
                    export_path = entry.get('export_csv_path')
                    if export_path and os.path.exists(export_path):
                        try:
                            df = pd.read_csv(export_path)
                            # Use the export settings from the entry
                            section = entry.get('params', {}).get('_exportSection', 'RESERVED')
                            unit_cost = float(entry.get('params', {}).get('_unitCost', 800))
                            

                            # Add or update the export settings in the dataframe
                            df['Section'] = section
                            df['UnitCost'] = unit_cost
                                

                            df['_source_export'] = entry.get('timestamp', '') + ' - ' + entry.get('params', {}).get('event', 'Unnamed Event')
                            all_dfs.append(df)
                        except Exception as e:
                            st.error(f"Error reading {export_path}: {e}")
                
                if all_dfs:
                    combined_df = pd.concat(all_dfs, ignore_index=True)
                    # Rest of the export logic...
                    csv = combined_df.to_csv(index=False).encode('utf-8')
                    
                    # Create a timestamp for the filename
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"combined_export_{timestamp}.csv"
                    
                    st.download_button(
                        label="💾 Download Combined CSV",
                        data=csv,
                        file_name=filename,
                        mime='text/csv',
                        key="download_combined_csv",
                        use_container_width=True
                    )
                else:
                    st.warning("No valid export files found in selected entries.")
            else:
                st.warning("No selected entries with valid export files found.")

# Add clear history button at the bottom of the search history section
if _hist:  # Only show if there's history
    st.markdown("---")  # Add a divider
    if st.session_state.get('confirm_clear', False):
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("✅ Confirm Clear All", type="primary", use_container_width=True):
                if _clear_search_history():
                    st.success("Search history cleared successfully!")
                    st.rerun()
        with col2:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state.confirm_clear = False
                st.rerun()
    else:
        if st.button("🗑️ Clear All Search History", type="primary", use_container_width=True):
            st.session_state.confirm_clear = True
            st.rerun()

if run:
    try:
        all_rows = []
        with loading_container:
            with st.spinner("Searching for events..."):
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
                    # Keep only rows where date is today or in the future, and exclude year 2099
                    df = df[df['_date_parsed'].notna() & 
                          (df['_date_parsed'] >= today) & 
                          (df['_date_parsed'].dt.year != 2099)].copy()
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