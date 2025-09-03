import streamlit as st
import numpy as np
import io
from scipy.interpolate import griddata
import plotly.graph_objects as go

# La funzione load_surfer rimane invariata
def load_surfer(fname_or_stream, fmt='ascii'):
    """Reads a Surfer grid file from a filename or a file-like object."""
    assert fmt in ['ascii', 'binary'], f"Invalid grid format '{fmt}'. Should be 'ascii' or 'binary'."
    if isinstance(fname_or_stream, str):
        ftext = open(fname_or_stream, 'r')
    else:
        fname_or_stream.seek(0)
        ftext = io.TextIOWrapper(fname_or_stream, encoding='utf-8')
    try:
        if fmt == 'ascii':
            id_line = ftext.readline()
            if not id_line.strip() == 'DSAA':
                raise ValueError("Not a valid Surfer ASCII GRD file (missing DSAA header).")
            nx, ny = [int(s) for s in ftext.readline().split()]
            xmin, xmax = [float(s) for s in ftext.readline().split()]
            ymin, ymax = [float(s) for s in ftext.readline().split()]
            zmin, zmax = [float(s) for s in ftext.readline().split()]
            data = np.fromiter((float(i) for line in ftext for i in line.split()), dtype='float64')
            x_coords = np.linspace(xmin, xmax, nx)
            y_coords = np.linspace(ymin, ymax, ny)
            x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)
            x, y = x_mesh.ravel(), y_mesh.ravel()
        else: # fmt == 'binary'
            raise NotImplementedError("Binary file support is not implemented yet.")
    finally:
        if isinstance(fname_or_stream, str):
            ftext.close()
    return x, y, data, (ny, nx), zmin, zmax

# Modifichiamo computePeak per accettare il metodo di interpolazione
def computePeak(filenames_and_data, MinMax, sigma, interp_method='nearest'):
    """Core computation logic, now with selectable interpolation method."""
    x_coords, y_coords, grds, nx_list, ny_list, xmin_list, xmax_list, ymin_list, ymax_list, dx_list, dy_list = [], [], [], [], [], [], [], [], [], [], []
    for i, (fname, file_data) in enumerate(filenames_and_data):
        x_tmp, y_tmp, grd_tmp, (ny_tmp, nx_tmp), zmin_tmp, zmax_tmp = load_surfer(file_data)
        x_coords.append(x_tmp); y_coords.append(y_tmp)
        no_data_value = 1.70141E+38
        grd_tmp[grd_tmp >= no_data_value] = np.nan
        grds.append(grd_tmp)
        nx_list.append(nx_tmp); ny_list.append(ny_tmp); xmin_list.append(x_tmp.min()); xmax_list.append(x_tmp.max()); ymin_list.append(y_tmp.min()); ymax_list.append(y_tmp.max())
        dx_val = (xmax_list[-1] - xmin_list[-1]) / (nx_tmp - 1) if nx_tmp > 1 else 0
        dy_val = (ymax_list[-1] - ymin_list[-1]) / (ny_tmp - 1) if ny_tmp > 1 else 0
        dx_list.append(dx_val); dy_list.append(dy_val)

    x1, x2 = np.max(xmin_list), np.min(xmax_list)
    y1, y2 = np.max(ymin_list), np.min(ymax_list)
    if x1 > x2 or y1 > y2:
        raise ValueError("The input grids do not have a valid overlapping area.")

    dxmax, dymax = np.max(dx_list), np.max(dy_list)
    ncols = int(np.ceil((x2 - x1) / dxmax)) + 1 if dxmax > 0 else 1
    nrows = int(np.ceil((y2 - y1) / dymax)) + 1 if dymax > 0 else 1
    xnew, ynew = np.linspace(x1, x2, ncols), np.linspace(y1, y2, nrows)
    dxnew = xnew[1] - xnew[0] if ncols > 1 else dxmax
    dynew = ynew[1] - ynew[0] if nrows > 1 else dymax
    X_new, Y_new = np.meshgrid(xnew, ynew)

    data_new = []
    validdataAll = np.ones_like(X_new, dtype=bool)
    for i in range(len(grds)):
        data_coeff = -1.0 if MinMax[i] == 'Min' else 1.0
        points = np.vstack([x_coords[i].ravel(), y_coords[i].ravel()]).T
        original_values = data_coeff * grds[i].ravel()
        valid_indices = ~np.isnan(original_values)
        interpolated_data = griddata(points[valid_indices], original_values[valid_indices], (X_new, Y_new), fill_value=np.nan, method=interp_method)
        validity_mask_original = (~np.isnan(grds[i].ravel())).astype(float)
        interpolated_validity_mask = griddata(points, validity_mask_original, (X_new, Y_new), fill_value=0, method='nearest')
        interpolated_data[interpolated_validity_mask < 0.5] = np.nan
        data_new.append(interpolated_data)
        validdataAll &= ~np.isnan(interpolated_data)

    mean_data, std_data = [], []
    for i in range(len(grds)):
        valid_values = data_new[i][validdataAll]
        mean_data.append(np.mean(valid_values) if valid_values.size > 0 else np.nan)
        std_data.append(np.std(valid_values) if valid_values.size > 0 else np.nan)
        if std_data[-1] == 0: std_data[-1] = 1e-9

    maskBoth = np.ones_like(X_new, dtype=bool); maskOr = np.zeros_like(X_new, dtype=bool)
    maskBoth.fill(True); maskOr.fill(False)
    for j in range(len(grds)):
        if np.isnan(mean_data[j]) or np.isnan(std_data[j]): continue
        temp_data = np.copy(data_new[j]); temp_data[np.isnan(temp_data)] = -np.inf 
        threshold = mean_data[j] + float(sigma[j]) * std_data[j]
        dataMask = temp_data > threshold
        maskBoth &= dataMask; maskOr |= dataMask
    maskBoth &= validdataAll; maskOr &= validdataAll
    fit_idx = np.sum(maskBoth) / np.sum(maskOr) if np.sum(maskOr) > 0 else 0.0
    areaOpt = np.sum(maskBoth) * dxnew * dynew

    data_norm_stack = []
    for i in range(len(grds)):
        if not np.isnan(mean_data[i]) and not np.isnan(std_data[i]) and std_data[i] > 0:
            normalized = (data_new[i] - mean_data[i]) / std_data[i]
            data_norm_stack.append(normalized[validdataAll])
    data_masked = np.full(X_new.shape, np.nan, dtype=np.float64)
    if data_norm_stack and all(s.size == np.sum(validdataAll) for s in data_norm_stack):
        data_masked[validdataAll] = np.amin(np.vstack(data_norm_stack), axis=0)
    final_masked_data = np.full(X_new.shape, np.nan, dtype=np.float64)
    final_masked_data[maskBoth] = data_masked[maskBoth]
    return [final_masked_data, validdataAll, x1, x2, y1, y2, areaOpt, fit_idx, mean_data, std_data]

# AGGIUNTA: Decoratore per la cache
@st.cache_data
def compute(filenames_and_data, MinMax, sigma, interp_method='nearest', colorscale='Jet'):
    """Main computation function with caching and flexible plotting options."""
    np.set_printoptions(threshold=np.inf)
    
    # Passa interp_method alla funzione di calcolo
    data_masked, validdataAll, x1, x2, y1, y2, areaOpt, fit_idx, mean_data, std_data = computePeak(filenames_and_data, MinMax, sigma, interp_method=interp_method)

    ncols, nrows = data_masked.shape[1], data_masked.shape[0]
    valid_data_for_header = data_masked[~np.isnan(data_masked)]
    min_val, max_val = (valid_data_for_header.min(), valid_data_for_header.max()) if valid_data_for_header.size > 0 else (0, 1)
    header = f"DSAA\n{ncols} {nrows}\n{x1} {x2}\n{y1} {y2}\n{min_val} {max_val}"
    
    fig = go.Figure(data=go.Heatmap(
        z=data_masked,
        x=np.linspace(x1, x2, ncols),
        y=np.linspace(y1, y2, nrows),
        colorscale=colorscale, # AGGIUNTA: Usa la colorscale passata come parametro
        colorbar=dict(title='Normalized Value'),
        hovertemplate='<b>X</b>: %{x:.2f}<br><b>Y</b>: %{y:.2f}<br><b>Value</b>: %{z:.3f}<extra></extra>'
    ))
    fig.update_layout(
        title_text='Peak Finder Result (Interactive)',
        xaxis_title='X-coordinate',
        yaxis_title='Y-coordinate',
        yaxis_scaleanchor="x"
    )

    formatter = {'float_kind': lambda x: f"{x:.2e}"}
    mean_data_str = np.array2string(np.absolute(np.array(mean_data)), formatter=formatter)
    std_data_str = np.array2string(np.array(std_data), formatter=formatter)
    return fig, data_masked, header, areaOpt, fit_idx, mean_data_str, std_data_str

# La funzione plot_single_grd rimane invariata
def plot_single_grd(file_data):
    """Loads and plots a single Surfer GRD file using Plotly."""
    try:
        x, y, data, (ny, nx), zmin, zmax = load_surfer(file_data)
        no_data_value = 1.70141E+38
        data[data >= no_data_value] = np.nan
        grid_data = data.reshape((ny, nx))
        x_coords = np.linspace(x.min(), x.max(), nx)
        y_coords = np.linspace(y.min(), y.max(), ny)
        fig = go.Figure(data=go.Heatmap(
            z=grid_data, x=x_coords, y=y_coords,
            colorscale='Viridis', colorbar=dict(title='Z-value'),
            hovertemplate='<b>X</b>: %{x:.2f}<br><b>Y</b>: %{y:.2f}<br><b>Z</b>: %{z:.3f}<extra></extra>'
        ))
        fig.update_layout(
            title_text='Input Data Preview (Interactive)',
            xaxis_title='X-coordinate', yaxis_title='Y-coordinate',
            yaxis_scaleanchor="x"
        )
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.update_layout(
            xaxis={'visible': False}, yaxis={'visible': False},
            annotations=[{'text': f"Error plotting file:<br>{e}", 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 16, 'color': 'red'}}]
        )
        return fig