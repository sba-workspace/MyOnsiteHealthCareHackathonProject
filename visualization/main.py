# import inspect
# import colorsys
# import textwrap
# import streamlit as st
# import pandas as pd
# import pydeck as pdk

# colors=[[252.53869454089883, 15.832939279468183, 15.832939279468183], [245.29492879295432, 166.22233345962402, 47.61344045962856], [206.53880386800535, 250.35405167454124, 31.277812641861907], [86.08444374035318, 252.58906414334174, 44.45828863960605], [49.77087522287998, 251.07449995280757, 130.29232511485108], [34.337683403663135, 253.71075041422856, 253.71075041422858], [23.507932628492572, 113.52512290553551, 248.55090832109997], [67.81854547282862, 22.683472932033485, 248.35883563600996], [201.5587841936545, 12.49408890653035, 248.82495801543558], [247.8891328653575, 47.34017156183642, 167.66954834394906]]

# ICON_URL = 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/d1/Google_Maps_pin.svg/137px-Google_Maps_pin.svg.png'

# icon_data = {
#     "url": ICON_URL,
#     "width": 137,
#     "height": 240,
#     "anchorY": 240,
#     "mask": True
# }

# def srcpath(src):
#     @st.cache
#     def lay(src):
#         layers = []
#         df=df0.query(f"`src`=={src}")

#         for i in range(df.shape[0]):
#             layer = pdk.Layer(
#                 type='PathLayer',
#                 data=df[i:i+1],
#                 rounded=True,
#                 billboard=True,
#                 pickable=True,
#                 width_min_pixels=3,
#                 auto_highlight=True,
#                 get_color=colors[i],
#                 get_path='path',
#             )
#             layers.append(layer)
        
#         srclayer = pdk.Layer(
#           "ScatterplotLayer",
#           data=dfs[src:src+1],
#           pickable=True,
#           stroked=True,
#           filled=True,
#           opacity=1,
#           get_radius=3999,
#           radius_max_pixels=12,
#           get_position="coordinates",
#           get_fill_color=[0,0,0],
#         )
        
#         layers.append(srclayer)
        
#         iconlayer = pdk.Layer(
#             type='IconLayer',
#           data=dfs[src:src+1],
#             billboard=True,
#             get_icon='icon_data',
#             get_size=79,
#             get_color=[155,155,155],
#             size_scale=1,
#             size_min_pixels=10,
#             opacity=0.6,
#             size_max_pixels=100,
#             get_position='coordinates',
#             pickable=True
#         )

#         layers.append(iconlayer)

#         return layers

#     layers = lay(src)

#     st.pydeck_chart(pdk.Deck(
#         map_style="mapbox://styles/mapbox/light-v9",
#         initial_view_state={"latitude": dfs["coordinates"][src][1], "longitude": dfs["coordinates"][src][0], "zoom": 7, "pitch": 30},
#         layers=layers,
#     ))

# def getpath(src, tp, veh):

#     @st.cache
#     def lay(src, tp, veh):
#         layers = []
#         df=df0.query(f"`src`=={src}").query(f"`type`=={tp}").query(f"`veh`=={veh}")

#         layer = pdk.Layer(
#             type='PathLayer',
#             data=df,
#             rounded=True,
#             billboard=True,
#             pickable=True,
#             width_min_pixels=3,
#             auto_highlight=True,
#             get_path='path',
#         )
#         layers.append(layer)

#         srclayer = pdk.Layer(
#           "ScatterplotLayer",
#           data=dfs[src:src+1],
#           pickable=True,
#           stroked=True,
#           filled=True,
#           opacity=1,
#           get_radius=3999,
#           radius_max_pixels=12,
#           get_position="coordinates",
#           get_fill_color=[0,0,0],
#         )
        
#         layers.append(srclayer)
        
#         iconlayer = pdk.Layer(
#             type='IconLayer',
#           data=dfs[src:src+1],
#             billboard=True,
#             get_icon='icon_data',
#             get_size=79,
#             get_color=[155,155,155],
#             size_scale=1,
#             size_min_pixels=10,
#             opacity=0.6,
#             size_max_pixels=100,
#             get_position='coordinates',
#             pickable=True
#         )

#         layers.append(iconlayer)

#         return layers

#     layers = lay(src, tp,veh)
#     st.pydeck_chart(pdk.Deck(
#         map_style="mapbox://styles/mapbox/light-v9",
#         initial_view_state={"latitude": dfs["coordinates"][src][1], "longitude": dfs["coordinates"][src][0], "zoom": 8, "pitch": 0},
#         layers=layers,
#     ))

# def fullpath():
#     @st.cache
#     def lay():
#         layers = []

#         for i in range(df0['src'].nunique()):
#             df=df0.query(f"`src`=={i}")
#             layer = pdk.Layer(
#                 type='PathLayer',
#                 data=df,
#                 rounded=True,
#                 billboard=True,
#                 pickable=True,
#                 width_min_pixels=5,
#                 auto_highlight=True,
#                 get_color=colors[i],
#                 get_path='path',
#             )
#             layers.append(layer)
#         return layers

#     st.pydeck_chart(pdk.Deck(
#         map_style="mapbox://styles/mapbox/light-v9",
#         initial_view_state={"latitude": 48.8566, "longitude": 2.3522, "zoom": 7, "pitch": 0},
#         layers=lay(),
#     ))

# def overview():
#     @st.cache
#     def lay():
#         layers = []

#         for i in range(dfd['labels'].nunique()):

#             df=dfd.query(f"`labels`=={i}")
         
#             layer = pdk.Layer(
#               "ScatterplotLayer",
#               data=df,
#               pickable=True,
#               stroked=True,
#               filled=True,
#               opacity=0.2,
#               get_radius=3999,
#               radius_max_pixels=19,
#               get_position="coordinates",
#               get_fill_color=colors[i],
#             )
            
#             layers.append(layer)

#         srclayer = pdk.Layer(
#           "ScatterplotLayer",
#           data=dfs,
#           pickable=True,
#           stroked=True,
#           filled=True,
#           opacity=1,
#           get_radius=3999,
#           radius_max_pixels=7,
#           get_position="coordinates",
#           get_fill_color=[0,0,0],
#         )
        
#         layers.append(srclayer)
        
#         iconlayer = pdk.Layer(
#             type='IconLayer',
#             data=dfs,
#             billboard=True,
#             get_icon='icon_data',
#             get_size=89,
#             size_scale=1,
#             size_min_pixels=10,
#             opacity=0.6,
#             size_max_pixels=100,
#             get_position='coordinates',
#             pickable=True
#         )
#         layers.append(iconlayer)
#         return layers

#     st.pydeck_chart(pdk.Deck(
#         map_style="mapbox://styles/mapbox/light-v9",
#         initial_view_state={"latitude": 48.8566, "longitude": 2.3522, "zoom": 7, "pitch": 40},
#         layers=lay(),
#     ))


# @st.cache
# def getdf():
#     df = pd.read_json('srcdf.json')
#     df['icon_data'] = None
#     for i in df.index:
#         df['icon_data'][i] = icon_data
#     return [pd.read_json(x) for  x in ['destdf.json','fulldf.json']] + [df]

# if __name__ == "__main__":
#     dfd, df0, dfs = getdf()

#     '''
# # Routero
# _-team Genesis_

# ## AI-based route optimization and visualization tool for sales vehicles.

# In the fast-developing logistics and supply chain management fields, one of the critical problems in the decision-making system is that how to arrange a proper supply chain for a lot of destinations and suppliers and produce a detailed supply schedule under a set of constraints. Solutions to the multisource vehicle routing problem (MDVRP) help in solving this problem in case of transportation applications.

# ### Clustered waypoints
#     '''
#     overview()
#     '''
# _(hover and scroll to zoom)_

# In the interactive map above, the locations of sources (pins) and destinations (circles) are shown after initial clustering. For such data the MDVRP requires the assignment of destination to sources and the vehicle routing for visiting them. Each vehicle originates from one source, serves the destinations assigned to that source, and returns to the same source. The objective of the MDVRP is to serve all destinations while minimizing the total travel distance (hence cost) under the constraint that the total demands of served destinations cannot exceed the capacity of the vehicle for each route.

# This project uses a heuristic algorithm to solve this problem. The proposed algorithm consists of four phases:

# * Phase 1: Find the destination nodes which will be catered by each source node using a modified k-means clustering algorithm.
# * Phase 2: Do preliminary analysis on the points for determining the type and number of the vehicles for given constraints.
# * Phase 3: Assign the vehicle to the subsets of the destination points using K-means while minimizing the fuel cost.
# * Phase 4: Optimize the routing for the allotted locations using _Travelling Salesman_ optimization) for each vehicle.

# Detailed analysis of the implementation can be found in the `prototype.ipynb` present in the project repository.

# ### Routing demo
# '''
#     dtype = st.radio("", ('Single vehicle', 'Full demo'), index=0)
#     if dtype=='Single vehicle':
#         src = st.selectbox(
#                 'Choose source depot ID',range(10),index=4)
#         srcdf0 = df0.loc[df0['src'] == src]
#         ntype = srcdf0['type'].nunique()
#         tp = st.selectbox(
#                 'Choose vehicle type',range(int(ntype)+1),index=2,
#                 format_func=(lambda x: ['Pickup','Truck','Any'][x]))
#         if tp != 2:
#             vehdf0 = srcdf0.loc[srcdf0['type']==tp]
#             veh = st.selectbox(
#                     'Choose vehicle ID',range(vehdf0.shape[0]),
#                         index=0)
#             getpath(src, tp, veh)
#         else:
#             srcpath(src)
#     else:
#         st.info('Full demo takes few seconds to load..')
#         fullpath()
#     st.write("_(hover and scroll to zoom)_")
#     '''
#     The hackathon was a great learning opportunity to get familiar with cloud-based development. Since we were free to choose our dataset, a considerable effort put into ensuring our synthetic data was sufficiently realistic. Apart from that, we experimented with multiple existing approaches to VRP, including genetic algorithms and recursive-DBSCAN. However, we found our strategy to be most performant when scaled to large data. One of the primary reasons is the relative simplicity and the speed of execution, which allows much-needed flexibility. 
# ### Team Members
# * Tejasvi S Tomar [@tejasvi](http://github.com/tejasvi)
# * Pooja Dhane [@poojaDh](http://github.com/poojadh)

# ### Data used
# Synthetic data based on [The heterogeneous fleet vehicle routing problem with overloads and time windows]( https://www.sciencedirect.com/science/article/pii/S0925527313000388) after heavy modification according to the given specifications.

# ### Azure Services Used
# * Azure Maps (distance API and routing)
# * Azure Machine Learning (training modified K-NN)
# * Azure Notebooks (prototyping and data analysis)
# * Azure App Service (Web-based deployment using MapBox and Streamlit)
#     '''
#########################
# app.py
#########################
import json
import math
import colorsys
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ---- YOUR CLUSTERING CLASS ----
from route_generator import GeoClustering   # adapt the import path!

# ------------------------------------------------------------------
# 1)  CONFIG
# ------------------------------------------------------------------
ICON_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d1/"
    "Google_Maps_pin.svg/137px-Google_Maps_pin.svg.png"
)
ICON_DATA = {"url": ICON_URL, "width": 137, "height": 240, "anchorY": 240, "mask": True}

SOURCE_JSON = Path("source.json")
DEST_JSON   = Path("destination.json")

# ------------------------------------------------------------------
# 2)  HELPERS
# ------------------------------------------------------------------
def _gen_colors(n: int) -> list[list[int]]:
    """n visually distinct RGB colours (0-255) using HSV wheel; noise = grey."""
    cols = [tuple(int(x * 255) for x in colorsys.hsv_to_rgb(h, 0.65, 0.90)) for h in np.linspace(0, 1, n, endpoint=False)]
    return [list(c) for c in cols] + [[130, 130, 130]]  # last entry for noise / unassigned


def _straight_paths(src_df: pd.DataFrame, dest_df: pd.DataFrame) -> pd.DataFrame:
    """Fallback – build straight-line paths Src → each Dest (lon/lat order)."""
    paths = []
    for i, row in src_df.iterrows():
        cluster = row["labels"]
        dests   = dest_df[dest_df["labels"] == cluster]
        for _, d in dests.iterrows():
            paths.append(
                {
                    "src": int(i),
                    "cluster": int(cluster),
                    "path": [
                        [row["lon"],  row["lat"]],
                        [d["lon"],    d["lat"]],
                    ],
                }
            )
    return pd.DataFrame(paths)


# ------------------------------------------------------------------
# 3)  DATA + CLUSTERING (cached)
# ------------------------------------------------------------------
@st.cache_data(show_spinner="Loading data …")
def load_raw() -> list[pd.DataFrame]:
    src_df  = pd.read_json(SOURCE_JSON)
    dest_df = pd.read_json(DEST_JSON)
    return [src_df, dest_df]


@st.cache_resource(show_spinner="Clustering …")
def compute_clusters(
    method: str,
    eps_km: float,
    min_samples: int,
    auto_k: bool,
    max_k: int,
    n_clusters: int,
):
    src_df, dest_df = load_raw()
    coords = list(zip(src_df.lon,  src_df.lat)) + list(zip(dest_df.lon, dest_df.lat))

    clusterer = GeoClustering(cluster_method=method)

    if method == "dbscan":
        labels, centres = clusterer.cluster_coordinates(
            coords, eps_km=eps_km, min_samples=min_samples
        )

    elif method == "kmeans":
        if auto_k:
            labels, centres = clusterer.cluster_coordinates(
                coords, auto=True, max_k=max_k
            )
        else:
            labels, centres = clusterer.cluster_coordinates(
                coords, n_clusters=n_clusters
            )

    else:  # distance
        labels, centres = clusterer.cluster_coordinates(coords)

    # --- build dataframes expected by the old pydeck code ----------
    colours = _gen_colors(max(len(set(labels)) - (1 if -1 in labels else 0), 1))

    # sources = cluster centres (these will get the pin icon)
    centres_df = pd.DataFrame(centres, columns=["lon", "lat"])
    centres_df["coordinates"] = centres_df[["lat", "lon"]].apply(lambda r: [r[1], r[0]], axis=1)
    centres_df["labels"] = centres_df.index
    centres_df["icon_data"] = [ICON_DATA] * len(centres_df)

    # destinations = original points
    full_df = pd.DataFrame(coords, columns=["lon", "lat"])
    full_df["labels"] = labels
    full_df["coordinates"] = full_df[["lat", "lon"]].apply(lambda r: [r[1], r[0]], axis=1)

    # a per-source dataframe duplicating original structure
    path_df = _straight_paths(centres_df, full_df)  # replace if you have real path geometry

    return centres_df, full_df, path_df, colours


# ------------------------------------------------------------------
# 4)  SIDEBAR  – choose parameters
# ------------------------------------------------------------------
st.sidebar.title("Routero – demo controller")
algo = st.sidebar.selectbox("Clustering algorithm", ("dbscan", "kmeans", "distance"))

if algo == "dbscan":
    eps   = st.sidebar.slider("ε (km)", 0.1, 10.0, 2.0, 0.1)
    m_pts = st.sidebar.slider("min_samples", 1, 10, 2, 1)
    auto_k = False
    max_k = n_clusters = 0

elif algo == "kmeans":
    auto_k = st.sidebar.checkbox("Automatic K (silhouette)", value=True)
    if auto_k:
        max_k = st.sidebar.slider("max K to search", 2, 20, 10, 1)
        n_clusters = 0
    else:
        n_clusters = st.sidebar.slider("Fixed K", 2, 20, 10, 1)
        max_k = 0
    eps = m_pts = 0

else:  # distance clustering – no params for now
    eps = m_pts = auto_k = max_k = n_clusters = 0

# ------------------------------------------------------------------
# 5)  RUN CLUSTERING
# ------------------------------------------------------------------
dfs, dfd, df0, colours = compute_clusters(
    algo, eps, m_pts, auto_k, max_k, n_clusters
)   # centres_df, full_df, path_df, colours

# rename to match original variable names your layers expect
dfs = dfs.rename(columns={"coordinates": "coordinates"})   # sources
dfd = dfd.rename(columns={"coordinates": "coordinates"})   # destinations
df0 = df0.rename(columns={"path": "path"})                 # paths (if any)

# ------------------------------------------------------------------
# 6)  MAP LAYERS
# ------------------------------------------------------------------
def scatter_layer(df, radius_px, opacity, colour):
    return pdk.Layer(
        "ScatterplotLayer",
        data=df,
        pickable=True,
        stroked=True,
        filled=True,
        opacity=opacity,
        get_position="coordinates",
        get_radius=4000,
        radius_max_pixels=radius_px,
        get_fill_color=colour,
    )

def icon_layer(df):
    return pdk.Layer(
        "IconLayer",
        data=df,
        billboard=True,
        get_icon="icon_data",
        get_size=90,
        size_scale=1,
        size_min_pixels=10,
        size_max_pixels=100,
        get_color=[120, 120, 120],
        get_position="coordinates",
        pickable=True,
    )

layers = []

# coloured dots for each cluster label
for lbl in sorted(dfd.labels.unique()):
    colour = colours[int(lbl)] if lbl != -1 else colours[-1]
    layers.append(scatter_layer(dfd[dfd.labels == lbl], 18, 0.25, colour))

# source pins
layers.append(scatter_layer(dfs, 8, 1.0, [0, 0, 0]))
layers.append(icon_layer(dfs))

# straight-line paths (optional)
if not df0.empty:
    layers.append(
        pdk.Layer(
            "PathLayer",
            data=df0,
            get_path="path",
            get_color="cluster",
            width_min_pixels=4,
            rounded=True,
            pickable=True,
            auto_highlight=True,
        )
    )

# ------------------------------------------------------------------
# 7)  SHOW MAP
# ------------------------------------------------------------------
initial = {
    "latitude": dfd.lat.mean(),
    "longitude": dfd.lon.mean(),
    "zoom": 6,
    "pitch": 35,
}

st.pydeck_chart(
    pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=initial,
        layers=layers,
        tooltip={"text": "Cluster {labels}"},
    )
)

st.caption("Hover & scroll to zoom – parameters in the sidebar.")
