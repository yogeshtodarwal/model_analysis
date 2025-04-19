import os
import pickle as pkl
import zlib
import traceback
import json # For AJAX responses
import io # For df.info() capture
from flask import Flask, request, render_template_string, redirect, url_for, flash, session, jsonify, abort
import pandas as pd
import numpy as np
from pathlib import Path
import time # For basic caching expiry

# --- Custom Utils Import ---
try:
    # Assuming utils_tbox is accessible in the Python path
    from utils_tbox.utils_tbox import read_pklz
except ImportError:
    print("WARNING: Could not import read_pklz from utils_tbox.utils_tbox.")
    print("Falling back to a basic implementation within the app.")
    # Define a basic fallback read_pklz if the import fails
    def read_pklz(f_path):
        try:
            with open(f_path, 'rb') as file:
                compressed_data = file.read()
                try:
                    # Try decompressing first
                    data = pkl.loads(zlib.decompress(compressed_data))
                except zlib.error:
                    # If zlib fails, try direct unpickle (might be uncompressed)
                    data = pkl.loads(compressed_data)
                except pkl.UnpicklingError:
                     # If direct unpickle fails after zlib error, maybe it was just pickled bytes?
                     # This case is ambiguous, re-raise for now.
                     raise
                # --- Potential Double Decompression Logic (if needed) ---
                # Check if the result is bytes and looks like zlib data
                if isinstance(data, bytes) and data.startswith(b'x\x9c'):
                    print(f"DEBUG: Data loaded as bytes starting with zlib header. Attempting second decompression for {f_path}")
                    try:
                        data = pkl.loads(zlib.decompress(data))
                    except Exception as e_inner:
                        print(f"ERROR: Second decompression/unpickle failed for {f_path}: {e_inner}")
                        # Return the bytes object in this case, as it's the best we have
                        return data # Or raise an error? Returning bytes for now.
                return data
        except FileNotFoundError:
            print(f"ERROR (fallback read_pklz): File not found: {f_path}")
            raise
        except Exception as e:
            print(f"ERROR (fallback read_pklz): Failed to read/process file {f_path}: {e}")
            traceback.print_exc()
            raise

# --- Configuration ---
BASE_DIR = os.path.expanduser("~")
# BASE_DIR = "/path/to/your/results"

# --- Constants ---
AJAX_MAX_LIST_PREVIEW = 20  # Max list items to return directly in AJAX
AJAX_MAX_DICT_PREVIEW = 20  # Max dict items to return directly in AJAX
AJAX_MAX_DF_ROWS = 10       # Max DataFrame rows for AJAX preview
AJAX_MAX_STR_PREVIEW = 500   # Max string length in AJAX preview
# CACHE_TIMEOUT_SECONDS removed

# --- Helper Function: Get Data at Path ---
def get_value_at_path(data, path_list):
    """Navigates data using a list of keys/indices."""
    current = data
    for key_or_index in path_list:
        try:
            if isinstance(current, dict):
                current = current[key_or_index]
            elif isinstance(current, (list, tuple)):
                current = current[int(key_or_index)] # Indices must be integers
            elif isinstance(current, pd.DataFrame) and key_or_index == '__head__':
                 # Special case for requesting DataFrame head preview
                 return current.head(AJAX_MAX_DF_ROWS)
            elif isinstance(current, pd.DataFrame) and key_or_index == '__info__':
                 # Special case for requesting DataFrame info
                 buffer = io.StringIO()
                 current.info(buf=buffer)
                 return buffer.getvalue()
            else:
                # Cannot navigate further into this type
                raise TypeError(f"Cannot index/access item within type {type(current).__name__}")
        except (KeyError, IndexError, ValueError, TypeError) as e:
            print(f"ERROR: Path navigation failed at '{key_or_index}' in path {path_list}: {e}")
            return None # Indicate navigation failure
    return current

# --- Helper Function: Prepare Data for AJAX ---
def prep_for_ajax(data_part, current_path_str):
    """Prepares a piece of data for JSON serialization, creating placeholders for large items."""
    data_type = type(data_part).__name__

    if data_part is None:
        return {'type': 'NoneType', 'value': None}

    elif isinstance(data_part, (int, float, bool, str, bytes)):
        if isinstance(data_part, bytes):
             val = repr(data_part[:AJAX_MAX_STR_PREVIEW]) + ('...' if len(data_part) > AJAX_MAX_STR_PREVIEW else '')
        else:
             val = str(data_part)
             val = val[:AJAX_MAX_STR_PREVIEW] + ('...' if len(val) > AJAX_MAX_STR_PREVIEW else '')
        return {'type': data_type, 'value': val}

    elif isinstance(data_part, pd.DataFrame):
        # Return metadata, prompt user to load head/info via specific paths
        # Ensure column names are strings for JSON serialization
        cols = [str(c) for c in data_part.columns]
        return {
            'type': 'DataFrame',
            'shape': data_part.shape,
            'columns': cols[:50] + (['...'] if len(cols) > 50 else []), # Preview columns
            'memory_usage': f"{data_part.memory_usage(deep=True).sum() / 1e6:.2f} MB",
            'load_head_path': f"{current_path_str}/__head__", # Special path component
            'load_info_path': f"{current_path_str}/__info__", # Special path component
            'is_lazy': True # Indicate more can be loaded
        }

    elif isinstance(data_part, np.ndarray):
        # Return metadata and maybe a tiny preview if 1D and small
        preview = None
        if data_part.ndim == 1 and data_part.size < 20:
             preview = [prep_for_ajax(item, f"{current_path_str}[{i}]") for i, item in enumerate(data_part)]

        return {
            'type': 'ndarray',
            'shape': data_part.shape,
            'dtype': str(data_part.dtype),
            'size': data_part.size,
            'preview': preview, # Will be null if not generated
            'is_lazy': preview is None and data_part.size > 0 # Indicate more can be loaded if no preview
        }

    elif isinstance(data_part, dict):
        keys = list(data_part.keys())
        item_count = len(keys)
        preview_items = {}
        is_lazy = item_count > AJAX_MAX_DICT_PREVIEW

        keys_to_show = keys[:AJAX_MAX_DICT_PREVIEW]
        for k in keys_to_show:
             # Represent key safely for JSON/JS
             key_str = str(k)
             key_str = key_str[:AJAX_MAX_STR_PREVIEW] + ('...' if len(key_str) > AJAX_MAX_STR_PREVIEW else '')
             # IMPORTANT: Only return metadata/placeholder for the value, not the value itself
             preview_items[key_str] = {
                 'type': type(data_part[k]).__name__,
                 'load_path': f"{current_path_str}/{k}", # Path to load this specific item
                 'is_placeholder': True
             }

        return {
            'type': 'dict',
            'item_count': item_count,
            'preview': preview_items,
            'is_lazy': is_lazy
        }

    elif isinstance(data_part, (list, tuple)):
        item_count = len(data_part)
        preview_items = []
        is_lazy = item_count > AJAX_MAX_LIST_PREVIEW

        items_to_show = data_part[:AJAX_MAX_LIST_PREVIEW]
        for i, item in enumerate(items_to_show):
             # IMPORTANT: Only return metadata/placeholder for the item
             preview_items.append({
                 'type': type(item).__name__,
                 'load_path': f"{current_path_str}/{i}", # Path to load this specific item
                 'index': i,
                 'is_placeholder': True
             })

        return {
            'type': data_type,
            'item_count': item_count,
            'preview': preview_items,
            'is_lazy': is_lazy
        }

    else:
        # Fallback for other types: return repr preview
        val = repr(data_part)
        val = val[:AJAX_MAX_STR_PREVIEW] + ('...' if len(val) > AJAX_MAX_STR_PREVIEW else '')
        return {'type': data_type, 'value': val, 'is_lazy': False}


# --- Flask App Initialization ---
app = Flask(__name__)
app.secret_key = os.urandom(24)

# --- HTML Templates ---

# BROWSE_TEMPLATE (Lists files, links to view_file)
BROWSE_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>PKLZ Browser</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.8.1/font/bootstrap-icons.css">
    <style>
        body { font-size: 0.9rem; }
        .file-browser { max-height: 600px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; margin-bottom: 20px; background-color: #f8f9fa; }
        .file-browser ul { list-style-type: none; padding-left: 0; }
        .file-browser li { padding: 3px 0; }
        .file-browser a { text-decoration: none; color: #0d6efd; }
        .file-browser a:hover { text-decoration: underline; }
        .file-browser .bi { margin-right: 5px; }
        .alert { margin-top: 15px; }
    </style>
</head>
<body>
<div class="container-fluid mt-3">
    <h1>PKLZ File Browser</h1>

    {% with messages = get_flashed_messages(with_categories=true) %}
      {% if messages %}
        {% for category, message in messages %}
        <div class="alert alert-{{ category }} alert-dismissible fade show" role="alert">
          {{ message }}
          <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        </div>
        {% endfor %}
      {% endif %}
    {% endwith %}

    <div class="card">
        <div class="card-header">File Browser</div>
        <div class="card-body file-browser">
            <p>Current Directory: <code>{{ current_path_display }}</code></p>
            <ul>
                {% if show_parent_link %}
                <li><a href="{{ url_for('browse', path=parent_path) }}"><i class="bi bi-arrow-up-left-circle"></i> Parent Directory</a></li>
                {% endif %}
                {% for item in items.dirs %}
                <li><a href="{{ url_for('browse', path=item.path) }}"><i class="bi bi-folder"></i> {{ item.name }}</a></li>
                {% endfor %}
                {% for item in items.files %}
                <li> <!-- Link to the new view_file route -->
                    <a href="{{ url_for('view_file', file_path=item.path) }}" title="View {{ item.path }}">
                        <i class="bi bi-file-earmark-zip"></i> {{ item.name }}
                    </a>
                </li>
                {% endfor %}
                {% if not items.files and not items.dirs and not show_parent_link %}
                 <p><em>Empty directory.</em></p>
                {% elif not items.files %}
                 <p class="mt-2"><em>No .pklz or .pkl files found here.</em></p>
                {% endif %}
            </ul>
        </div>
    </div>
</div>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

# VIEW_FILE_TEMPLATE (Initial structure + JS for AJAX)
VIEW_FILE_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>View: {{ file_name }}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.8.1/font/bootstrap-icons.css">
    <style>
        body { font-size: 0.9rem; }
        .explorer { border: 1px solid #ddd; padding: 15px; background-color: #fff; margin-top: 20px; }
        .explorer-item { margin-left: 25px; padding-left: 15px; border-left: 1px dotted #ccc; margin-top: 5px; }
        .explorer-item .key { font-weight: bold; color: #0056b3; cursor: pointer; }
        .explorer-item .index { font-weight: bold; color: #198754; cursor: pointer; }
        .explorer-item .placeholder { color: #6c757d; font-style: italic; cursor: pointer; display: inline-block; padding: 2px 5px; border: 1px dashed #ccc; border-radius: 3px; }
        .explorer-item .placeholder:hover { background-color: #e9ecef; }
        .explorer-item .load-indicator { display: none; color: #adb5bd; margin-left: 10px; font-size: 0.8em; }
        .explorer-item .data-type { font-style: italic; color: #6c757d; margin-left: 5px; font-size: 0.85em; }
        .explorer-item .value { margin-left: 5px; font-family: monospace; color: #d63384; white-space: pre-wrap; word-break: break-all; }
        .explorer-item .error { color: #dc3545; font-weight: bold; }
        .df-preview table { font-size: 0.8em; margin-top: 5px; max-height: 300px; overflow: auto; display: block; } /* Ensure table scrolls */
        .df-info { white-space: pre; font-family: monospace; font-size: 0.8em; background-color: #f8f8f8; border: 1px solid #eee; padding: 5px; margin-top: 5px; max-height: 200px; overflow-y: auto; }
        .breadcrumb-item a { text-decoration: none; }
        .breadcrumb-item.active { color: #6c757d; }
    </style>
</head>
<body>
<div class="container-fluid mt-3">
    <nav aria-label="breadcrumb">
      <ol class="breadcrumb">
        <li class="breadcrumb-item"><a href="{{ url_for('browse', path=dir_path) }}"><i class="bi bi-folder"></i> Browser</a></li>
        <li class="breadcrumb-item active" aria-current="page"><i class="bi bi-file-earmark-zip"></i> {{ file_name }}</li>
      </ol>
    </nav>

    <h1>Explore: {{ file_name }}</h1>

    {% with messages = get_flashed_messages(with_categories=true) %}
      {% if messages %}
        {% for category, message in messages %}
        <div class="alert alert-{{ category }} alert-dismissible fade show" role="alert">
          {{ message }}
          <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        </div>
        {% endfor %}
      {% endif %}
    {% endwith %}

    <div class="explorer">
        <div id="data-container">
            <!-- Initial structure rendered by prep_for_ajax will go here -->
            {{ initial_html | safe }}
        </div>
    </div>

</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
<script>
    // Simple unique ID generator for elements
    let elementIdCounter = 0;
    function generateUniqueId() {
        return 'elem-' + elementIdCounter++;
    }

    function renderData(data, targetElementId) {
        const target = document.getElementById(targetElementId);
        if (!target) {
            console.error('Target element not found:', targetElementId);
            return;
        }

        let html = '';
        const dataType = data.type || 'Unknown';

        if (dataType === 'NoneType') {
            html = '<span class="value text-muted">None</span>';
        } else if (dataType === 'DataFrame') {
            html = `<div><span class="data-type">(DataFrame)</span> Shape: ${data.shape}, Memory: ${data.memory_usage}`;
            html += `<br>Columns: ${data.columns.join(', ')}`;
            html += `<div class="mt-1">
                        <button class="btn btn-sm btn-outline-secondary placeholder" data-path="${data.load_head_path}" data-file="{{ file_path_js }}">Load Head (${AJAX_MAX_DF_ROWS} rows)</button>
                        <button class="btn btn-sm btn-outline-secondary placeholder" data-path="${data.load_info_path}" data-file="{{ file_path_js }}">Load Info</button>
                        <span class="load-indicator bi bi-hourglass-split"> Loading...</span>
                     </div>
                     <div class="df-content mt-2"></div></div>`; // Container for head/info
        } else if (dataType === 'ndarray') {
            html = `<div><span class="data-type">(ndarray)</span> Shape: ${data.shape}, Dtype: ${data.dtype}, Size: ${data.size}`;
            if (data.preview) { // Small 1D array preview
                 html += '<div class="explorer-item">';
                 data.preview.forEach((item, i) => {
                    html += `<div id="${generateUniqueId()}">`; // Give each item a container
                    // Assuming preview items are already rendered simple values
                    html += `<span class="index">[${i}]</span>: ${renderData(item, null)}`; // Recursive call (limited depth)
                    html += `</div>`;
                 });
                 html += '</div>';
            } else if (data.is_lazy) {
                 html += '<span class="data-type"> (Content too large for direct preview)</span>';
            }
            html += '</div>';
        } else if (dataType === 'dict') {
            html = `<div><span class="data-type">(dict)</span> ${data.item_count} items`;
            if (data.is_lazy) {
                 html += ` (Showing first ${Object.keys(data.preview).length})`;
            }
             html += `<div class="explorer-item">`;
             for (const key in data.preview) {
                 const item = data.preview[key];
                 const itemId = generateUniqueId();
                 html += `<div id="${itemId}"><span class="key placeholder" data-path="${item.load_path}" data-file="{{ file_path_js }}">${escapeHtml(key)}</span>: <span class="data-type">(${item.type})</span> <span class="load-indicator bi bi-hourglass-split"></span></div>`;
             }
             if (data.is_lazy && data.item_count > Object.keys(data.preview).length) {
                 html += `<div class="text-muted fst-italic ms-2">... ${data.item_count - Object.keys(data.preview).length} more items</div>`;
             }
             html += `</div></div>`;

        } else if (dataType === 'list' || dataType === 'tuple') {
             html = `<div><span class="data-type">(${dataType})</span> ${data.item_count} items`;
             if (data.is_lazy) {
                 html += ` (Showing first ${data.preview.length})`;
             }
             html += `<div class="explorer-item">`;
             data.preview.forEach(item => {
                 const itemId = generateUniqueId();
                 // Fix: Added closing quote to data-file attribute
                 html += `<div id="${itemId}"><span class="index placeholder" data-path="${item.load_path}" data-file="{{ file_path_js }}"> [${item.index}]</span>: <span class="data-type">(${item.type})</span> <span class="load-indicator bi bi-hourglass-split"></span></div>`;
             });
              if (data.is_lazy && data.item_count > data.preview.length) {
                 html += `<div class="text-muted fst-italic ms-2">... ${data.item_count - data.preview.length} more items</div>`;
             }
             html += `</div></div>`;
        } else if (dataType === 'DataFrame_Head') { // Special type for head result
             html = `<div class="df-preview table-responsive">${data.html}</div>`;
        } else if (dataType === 'DataFrame_Info') { // Special type for info result
             html = `<div class="df-info">${escapeHtml(data.text)}</div>`;
        }
         else { // Primitive types or repr fallback
            html = `<span class="value">${escapeHtml(data.value !== undefined ? String(data.value) : 'Error: Undefined Value')}</span> <span class="data-type">(${dataType})</span>`;
        }

        if (targetElementId) {
            target.innerHTML = html; // Replace placeholder content
            // Re-attach listeners to newly added elements within this container
            attachListeners(target);
        }
        return html; // Return HTML if called recursively/initially
    }

    function escapeHtml(unsafe) {
        if (unsafe === null || unsafe === undefined) return '';
        return String(unsafe)
             .replace(/&/g, "&amp;")
             .replace(/</g, "&lt;")
             .replace(/>/g, "&gt;")
             .replace(/"/g, "&quot;") // Corrected escaping and fixed Python syntax error
             .replace(/'/g, "&#39;"); // Corrected escaping
    }

    function fetchData(filePath, dataPath, targetElement) {
        const indicator = targetElement.querySelector('.load-indicator');
        if (indicator) indicator.style.display = 'inline-block';
        // Store the original placeholder text/style if needed, or just replace
        const placeholderSpan = targetElement.querySelector('.placeholder');


        fetch("{{ url_for('get_data_part') }}", {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ file_path: filePath, data_path: dataPath }),
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => { throw new Error(err.error || `HTTP error! status: ${response.status}`) });
            }
            return response.json();
        })
        .then(data => {
            if (indicator) indicator.style.display = 'none';
             // Pass the ID of the container DIV holding the placeholder
            renderData(data, targetElement.id);
        })
        .catch(error => {
            console.error('Fetch error:', error);
            if (indicator) indicator.style.display = 'none';
            targetElement.innerHTML = `<span class="error">Error loading: ${error.message}</span>`;
        });
    }

    function attachListeners(parentElement) {
         // Use event delegation on a container if possible, but for simplicity:
         parentElement.querySelectorAll('.placeholder').forEach(element => {
            // Prevent adding listener multiple times
            if (element.dataset.listenerAttached) return;
            element.dataset.listenerAttached = 'true';

            element.addEventListener('click', function(event) {
                event.preventDefault();
                const filePath = this.dataset.file;
                const dataPath = this.dataset.path;
                // Find the parent container div to replace its content
                const targetContainer = this.closest('div[id^="elem-"]'); // Find closest parent with generated ID
                 if (filePath && dataPath && targetContainer) {
                    fetchData(filePath, dataPath, targetContainer);
                } else {
                    console.error("Missing data attributes or target container", this.dataset, targetContainer);
                     if(targetContainer) targetContainer.innerHTML = "<span class='error'>Error: Cannot load data (missing attributes).</span>";
                }
            });
        });
    }

    // Initial attachment of listeners
    document.addEventListener('DOMContentLoaded', function() {
        const initialContainer = document.getElementById('data-container');
        attachListeners(initialContainer);

         // Assign unique IDs to initial placeholders for replacement
        initialContainer.querySelectorAll('.placeholder').forEach(element => {
             const parentDiv = element.closest('div');
             if(parentDiv && !parentDiv.id) {
                 parentDiv.id = generateUniqueId();
             }
        });

    });

</script>
</body>
</html>
"""

# --- Flask Routes ---

@app.route('/')
@app.route('/browse')
def browse():
    """Displays files and directories, linking files to view_file."""
    req_path = request.args.get('path', BASE_DIR)
    try:
        current_path = os.path.abspath(req_path)
        base_path = os.path.abspath(BASE_DIR)
        if not current_path.startswith(base_path):
            flash(f"Access denied: Cannot browse outside of '{base_path}'", "danger")
            current_path = base_path
    except Exception as e:
        flash(f"Error processing path: {e}", "danger")
        current_path = os.path.abspath(BASE_DIR)

    items = {'dirs': [], 'files': []}
    current_path_display = os.path.relpath(current_path, os.path.dirname(base_path)) # More user-friendly display path

    try:
        if not os.path.isdir(current_path):
             flash(f"Path not found or is not a directory: {current_path}", "warning")
             # Try going to parent if it's valid
             parent = Path(current_path).parent
             if os.path.isdir(parent) and str(parent).startswith(os.path.abspath(BASE_DIR)):
                 return redirect(url_for('browse', path=str(parent)))
             else:
                 return redirect(url_for('browse', path=BASE_DIR))

        for item_name in sorted(os.listdir(current_path), key=str.lower):
            item_path = os.path.join(current_path, item_name)
            try:
                if os.path.isdir(item_path):
                    items['dirs'].append({'name': item_name, 'path': item_path})
                elif os.path.isfile(item_path) and item_name.lower().endswith((".pklz", ".pkl")):
                    items['files'].append({'name': item_name, 'path': item_path})
            except OSError: pass # Ignore permission denied on specific items
    except OSError as e:
        flash(f"Error accessing directory {current_path}: {e}", "danger")
        return redirect(url_for('browse', path=BASE_DIR)) # Redirect to base on error

    # Parent directory link logic (same as before)
    show_parent_link = False
    parent_path = None
    abs_base_path = os.path.abspath(BASE_DIR)
    if os.path.abspath(current_path) != abs_base_path:
        parent_path = str(Path(current_path).parent)
        # Ensure parent path doesn't go above BASE_DIR
        if os.path.abspath(parent_path).startswith(abs_base_path) or os.path.abspath(parent_path) == abs_base_path:
            show_parent_link = True
        else:
             parent_path = abs_base_path # Correctly set parent to base if it would go outside
             show_parent_link = (os.path.abspath(current_path) != abs_base_path) # Show parent link only if not already at base


    return render_template_string(BROWSE_TEMPLATE,
                                 current_path=current_path,
                                 current_path_display=current_path_display,
                                 items=items,
                                 show_parent_link=show_parent_link,
                                 parent_path=parent_path)

@app.route('/view')
def view_file():
    """Loads the initial top-level structure of a PKLZ file."""
    from markupsafe import escape # Moved import here
    file_path = request.args.get('file_path')
    if not file_path:
        flash("No file path provided.", "danger")
        return redirect(url_for('browse'))

    # --- Security Check ---
    abs_file_path = os.path.abspath(file_path)
    abs_base_dir = os.path.abspath(BASE_DIR)
    if not abs_file_path.startswith(abs_base_dir) or not os.path.isfile(abs_file_path):
        flash("Invalid or forbidden file path.", "danger")
        return redirect(url_for('browse'))

    top_level_data = None # Initialize to None
    try:
        # Load the top-level object using the imported (or fallback) read_pklz
        top_level_data = read_pklz(abs_file_path) # Replaced read_pklz_cached

        # Prepare the initial structure using the AJAX helper
        initial_data_prepared = prep_for_ajax(top_level_data, "") # Start with empty path string

        # Need to escape file path for JS string literal *before* generating initial_html
        file_path_js = abs_file_path.replace('\\', '\\\\').replace("'", "\\'").replace('"', '&quot;')

        # Hacky initial render - just show top level keys/items as placeholders
        initial_html = ""
        # --- This initial rendering logic might need adjustment based on what read_pklz returns ---
        if isinstance(top_level_data, dict):
             initial_html += f'<div><span class="data-type">(dict)</span> {len(top_level_data)} items <div class="explorer-item">'
             for k in top_level_data:
                  key_str = str(k)
                  key_str = key_str[:AJAX_MAX_STR_PREVIEW] + ('...' if len(key_str) > AJAX_MAX_STR_PREVIEW else '')
                  item_type = type(top_level_data[k]).__name__
                  data_path = f"/{k}"
                  initial_html += f'<div id="elem-{hash(data_path)}"><span class="key placeholder" data-path="{escape(data_path)}" data-file="{file_path_js}">{escape(key_str)}</span>: <span class="data-type">({item_type})</span><span class="load-indicator bi bi-hourglass-split"></span></div>'
             initial_html += '</div></div>'
        elif isinstance(top_level_data, list):
             initial_html += f'<div><span class="data-type">(list)</span> {len(top_level_data)} items <div class="explorer-item">'
             for i, item in enumerate(top_level_data[:AJAX_MAX_LIST_PREVIEW]):
                  item_type = type(item).__name__
                  data_path = f"/{i}"
                  initial_html += f'<div id="elem-{hash(data_path)}"><span class="index placeholder" data-path="{escape(data_path)}" data-file="{file_path_js}">[{i}]</span>: <span class="data-type">({item_type})</span><span class="load-indicator bi bi-hourglass-split"></span></div>'
             if len(top_level_data) > AJAX_MAX_LIST_PREVIEW:
                  initial_html += f'<div class="text-muted fst-italic ms-2">... {len(top_level_data) - AJAX_MAX_LIST_PREVIEW} more items</div>'
             initial_html += '</div></div>'
        elif isinstance(top_level_data, bytes): # Explicitly handle if bytes are returned
             initial_html = f'<span class="value">Loaded as bytes: {escape(repr(top_level_data)[:500])}...</span> <span class="data-type">(bytes)</span>'
             initial_html += '<br><span class="error">Could not fully decode the structure. The pickled object might be compressed bytes.</span>'
        else:
             # Also escape the value here for safety
             initial_html = f'<span class="value">{escape(str(top_level_data)[:500])}</span> <span class="data-type">({type(top_level_data).__name__})</span>'


        return render_template_string(VIEW_FILE_TEMPLATE,
                                     file_path=abs_file_path,
                                     file_path_js=file_path_js, # Pass escaped path to JS main script block
                                     file_name=os.path.basename(abs_file_path),
                                     dir_path=os.path.dirname(abs_file_path),
                                     initial_html=initial_html, # Render initial placeholders
                                     AJAX_MAX_DF_ROWS=AJAX_MAX_DF_ROWS # Pass constant to JS
                                     )

    except MemoryError:
         flash(f"Out of Memory loading {os.path.basename(file_path)}. The file structure might be too large.", "danger")
         return redirect(url_for('browse', path=os.path.dirname(abs_file_path)))
    except Exception as e:
        # Check if the error message indicates the bytes issue and top_level_data was bytes
        if "unexpected '}'" in str(e) and isinstance(top_level_data, bytes):
             flash(f"Error loading initial data for {os.path.basename(file_path)}: Encountered template error, possibly due to data being loaded as raw bytes.", "warning")
        else:
             flash(f"Error loading initial data for {os.path.basename(file_path)}: {e}", "danger")
        traceback.print_exc()
        return redirect(url_for('browse', path=os.path.dirname(abs_file_path)))


@app.route('/get_data_part', methods=['POST'])
def get_data_part():
    """AJAX endpoint to fetch a specific part of the data structure."""
    try:
        payload = request.get_json()
        if not payload:
            abort(400, description="Missing JSON payload.")

        file_path = payload.get('file_path')
        data_path_str = payload.get('data_path') # e.g., "/results/0/val" or "/cfg" or "/results/0/val/__head__"

        if not file_path or data_path_str is None:
             abort(400, description="Missing 'file_path' or 'data_path' in payload.")

        # --- Security Check ---
        abs_file_path = os.path.abspath(file_path)
        abs_base_dir = os.path.abspath(BASE_DIR)
        if not abs_file_path.startswith(abs_base_dir) or not os.path.isfile(abs_file_path):
            abort(403, description="Invalid or forbidden file path.")

        # --- Parse Path ---
        path_parts = [p for p in data_path_str.split('/') if p]
        if not path_parts and data_path_str != '/':
             abort(400, description="Invalid data path format.")

        parsed_path = []
        for part in path_parts:
             if part.isdigit():
                 parsed_path.append(int(part))
             elif part in ['__head__', '__info__']:
                  parsed_path.append(part)
             else:
                 parsed_path.append(part)

        # --- Load Data ---
        # Use the imported (or fallback) read_pklz - no caching
        top_level_data = read_pklz(abs_file_path) # Replaced read_pklz_cached

        # --- Navigate ---
        # Add a check here: if top_level_data is bytes, navigation will fail.
        if isinstance(top_level_data, bytes):
             abort(400, description=f"Cannot navigate data path '{data_path_str}'. The base file loaded as raw bytes, not a dictionary or list.")

        target_data = get_value_at_path(top_level_data, parsed_path)

        # --- Handle Potential Nested Compressed/Pickled Bytes ---
        nested_decode_error = None
        if isinstance(target_data, bytes) and target_data.startswith(b'x\x9c'):
            print(f"DEBUG: Target data at '{data_path_str}' is bytes starting with zlib header. Attempting nested decompression.")
            try:
                decompressed_bytes = zlib.decompress(target_data)
                unpickled_object = pkl.loads(decompressed_bytes)
                print(f"DEBUG: Successfully decompressed/unpickled nested bytes at '{data_path_str}'. Type: {type(unpickled_object).__name__}")
                target_data = unpickled_object # Replace bytes with the actual object
            except zlib.error as zde:
                nested_decode_error = f"Nested zlib decompression failed: {zde}"
                print(f"ERROR: {nested_decode_error}")
            except pkl.UnpicklingError as pue:
                nested_decode_error = f"Nested pickle loading failed: {pue}"
                print(f"ERROR: {nested_decode_error}")
            except Exception as e_nested:
                nested_decode_error = f"Unexpected error during nested decoding: {e_nested}"
                print(f"ERROR: {nested_decode_error}")
                traceback.print_exc()
        # --- End Nested Decode Handling ---

        if target_data is None and parsed_path:
             parent_path = parsed_path[:-1]
             last_key = parsed_path[-1]
             parent_data = get_value_at_path(top_level_data, parent_path)
             is_legit_none = False
             try:
                  if isinstance(parent_data, dict) and parent_data.get(last_key) is None and last_key in parent_data:
                      is_legit_none = True
                  elif isinstance(parent_data, (list, tuple)) and int(last_key) < len(parent_data) and parent_data[int(last_key)] is None:
                      is_legit_none = True
             except: pass

             if not is_legit_none:
                  abort(404, description=f"Data not found at path: {data_path_str}")

        # --- Prepare Response ---
        if isinstance(target_data, pd.DataFrame) and parsed_path[-1] == '__head__':
             html_preview = target_data.to_html(classes='table table-sm table-bordered table-striped', border=0, max_rows=AJAX_MAX_DF_ROWS, escape=True)
             response_data = {'type': 'DataFrame_Head', 'html': html_preview}
        elif isinstance(target_data, str) and data_path_str.endswith('/__info__'):
              response_data = {'type': 'DataFrame_Info', 'text': target_data}
        else:
              response_data = prep_for_ajax(target_data, data_path_str)
              # If nested decoding failed, add the error to the response
              if nested_decode_error:
                  if 'value' in response_data: # Add to existing primitive display
                      response_data['value'] += f'\n<span class="error">({nested_decode_error})</span>'
                  else: # Add as a separate error field
                      response_data['nested_error'] = nested_decode_error

        return jsonify(response_data)

    except FileNotFoundError:
         abort(404, description="PKLZ file not found.")
    except MemoryError:
          abort(500, description="Server ran out of memory processing the request. Data part might be too large.")
    except Exception as e:
        print(f"ERROR in /get_data_part: {e}")
        traceback.print_exc()
        abort(500, description=f"Internal server error: {e}")


# --- Run the App ---
if __name__ == '__main__':
    print(f"--- PKLZ Explorer (Lazy Loading) ---")
    print(f"Serving files starting from base directory: {os.path.abspath(BASE_DIR)}")
    print(f"Access the browser at http://127.0.0.1:5002")
    app.run(host='0.0.0.0', port=5002, debug=True)