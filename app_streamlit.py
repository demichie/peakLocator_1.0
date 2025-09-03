import streamlit as st
import numpy as np
import io
from compute import compute, plot_single_grd

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Peak Finder Application")
st.title("Peak Finder Application")

# --- Constants & State Initialization ---
MIN_FILES, MAX_FILES = 2, 10
if 'num_inputs' not in st.session_state: st.session_state.num_inputs = MIN_FILES
if 'uploaded_files_bytes' not in st.session_state: st.session_state.uploaded_files_bytes = {}
if 'results' not in st.session_state: st.session_state.results = None

# --- Callback for Live Viewer ---
def update_viewer_files():
    """Reads all currently uploaded files into session state for the viewer."""
    st.session_state.uploaded_files_bytes = {}
    for i in range(st.session_state.num_inputs):
        file_key = f"filename{i+1}"
        if st.session_state[file_key] is not None:
            uploaded_file_obj = st.session_state[file_key]
            uploaded_file_obj.seek(0)
            st.session_state.uploaded_files_bytes[f"Input {i+1}: {uploaded_file_obj.name}"] = uploaded_file_obj.read()

# --- Main Layout with Tabs ---
tab1, tab2, tab3 = st.tabs(["Input Configuration", "Input Data Viewer", "Results"])

# ==============================================================================
# TAB 1: INPUT CONFIGURATION
# ==============================================================================
with tab1:
    st.header("1. Configure Inputs")

    # --- Advanced Options ---
    with st.expander("Advanced Options"):
        interp_method = st.selectbox("Interpolation Method", ['nearest', 'linear', 'cubic'], help="Method for `scipy.griddata`.")
        colorscale = st.selectbox("Result Colorscale", ['Jet', 'Viridis', 'Plasma', 'Inferno', 'Cividis'], help="Plotly colorscale for the result map.")

    # --- Dynamic Input Slots ---
    input_cols = st.columns(2)
    for i in range(st.session_state.num_inputs):
        col = input_cols[i % 2]
        with col:
            st.subheader(f"Input Slot {i+1}")
            st.file_uploader(f"Upload file {i+1}", type=['grd', 'dat', 'npy'], key=f"filename{i+1}", on_change=update_viewer_files)
            opts_col1, opts_col2 = st.columns(2)
            with opts_col1:
                st.selectbox("Min/Max", ['Max', 'Min'], key=f"MinMax{i+1}")
            with opts_col2:
                st.number_input("Sigma", min_value=0.1, value=1.0, step=0.1, key=f"sigma{i+1}", help="Standard deviation multiplier.")
            st.markdown("---")

    # --- Manage Input Slots Buttons ---
    st.subheader("Manage Input Slots")
    manage_cols = st.columns(8)
    def add_input_slot():
        if st.session_state.num_inputs < MAX_FILES: st.session_state.num_inputs += 1
    def remove_input_slot():
        if st.session_state.num_inputs > MIN_FILES: st.session_state.num_inputs -= 1
    with manage_cols[0]:
        st.button("Add Input Slot", on_click=add_input_slot, disabled=(st.session_state.num_inputs >= MAX_FILES))
    with manage_cols[1]:
        st.button("Remove Last Slot", on_click=remove_input_slot, disabled=(st.session_state.num_inputs <= MIN_FILES))

    st.markdown("---")
    st.header("2. Run Computation")
    # --- Computation Trigger ---
    if st.button("Run Computation", type="primary"):
        uploaded_files_data, minmax_choices, sigma_choices = [], [], []
        for i in range(st.session_state.num_inputs):
            file_key = f"filename{i+1}"
            if st.session_state[file_key] is not None:
                uploaded_file_obj = st.session_state[file_key]
                uploaded_file_obj.seek(0)
                uploaded_files_data.append((uploaded_file_obj.name, uploaded_file_obj))
                minmax_choices.append(st.session_state[f"MinMax{i+1}"])
                sigma_choices.append(st.session_state[f"sigma{i+1}"])

        if len(uploaded_files_data) < MIN_FILES:
            st.error(f"Please upload at least {MIN_FILES} files to run the computation.")
        else:
            with st.spinner("Computation in progress... (results are cached after first run)"):
                try:
                    # Pass the advanced options to the compute function
                    results = compute(uploaded_files_data, minmax_choices, sigma_choices, interp_method, colorscale)
                    st.session_state.results = {'fig': results[0], 'data_masked': results[1], 'header': results[2], 'areaOpt': results[3], 'fit_idx': results[4], 'mean_data_str': results[5], 'std_data_str': results[6]}
                    st.success("Computation complete! Check the 'Results' tab.")
                except ValueError as e:
                    st.session_state.results = None
                    st.error(f"Error in data processing: {e}")
                    st.info("This can happen if the input grids do not have a valid overlapping area or another data issue occurs.")
                except Exception as e:
                    st.session_state.results = None
                    st.error(f"An unexpected error occurred: {e}")

# ==============================================================================
# TAB 2: INPUT DATA VIEWER
# ==============================================================================
with tab2:
    st.header("Preview Uploaded Input Files")
    if not st.session_state.uploaded_files_bytes:
        st.warning("No files uploaded yet. Upload files in the 'Input Configuration' tab.")
    else:
        file_options = list(st.session_state.uploaded_files_bytes.keys())
        selected_file_key = st.selectbox("Select a file to display:", file_options)
        if selected_file_key:
            file_bytes_to_plot = st.session_state.uploaded_files_bytes[selected_file_key]
            st.plotly_chart(plot_single_grd(io.BytesIO(file_bytes_to_plot)), use_container_width=True)

# ==============================================================================
# TAB 3: RESULTS
# ==============================================================================
with tab3:
    st.header("Analysis Results")
    if st.session_state.results is None:
        st.info("Results will be displayed here after a successful computation.")
    else:
        results = st.session_state.results
        st.plotly_chart(results['fig'], use_container_width=True)
        st.subheader("Calculated Metrics")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric("Common Peak Region Area (AreaOpt)", f"{results['areaOpt']:.4f}")
        with res_col2:
            st.metric("Fitting Index", f"{results['fit_idx']:.4f}")
        st.subheader("Input Data Statistics")
        stats_col1, stats_col2 = st.columns(2)
        with stats_col1:
            st.write("Means:"); st.code(results['mean_data_str'])
        with stats_col2:
            st.write("Standard Deviations:"); st.code(results['std_data_str'])
        st.subheader("Download Results")
        data_masked_for_save = np.copy(results['data_masked'])
        no_data_value = 1.70141E+38
        data_masked_for_save[np.isnan(data_masked_for_save)] = no_data_value
        output = io.StringIO()
        np.savetxt(output, data_masked_for_save, fmt='%1.7f', comments='')
        grd_file_content = results['header'] + "\n" + output.getvalue()
        st.download_button(label="Download Result (out.grd)", data=grd_file_content.encode('utf-8'), file_name="out.grd", mime="application/octet-stream")