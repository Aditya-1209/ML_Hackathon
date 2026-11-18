"""Descriptive development-trip statistics and a reproducible map."""
from pathlib import Path

import folium
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def summarize(trip, reference, threshold):
    obs,nodes,arcs,meta=(trip[k] for k in ("observations","nodes","arcs","metadata"))
    good=obs[obs.valid]
    times=good.sort_values("order_index").timestamp_s.to_numpy()
    dt=np.diff(times)
    u,v=arcs.from_node_id.to_numpy(),arcs.to_node_id.to_numpy()
    graph=coo_matrix((np.ones(len(u),dtype=bool),(u,v)),shape=(len(nodes),len(nodes)))
    consecutive=continuous=None
    if reference is not None:
        order=reference.arc_id.to_numpy()
        consecutive=max(0,len(order)-1)
        continuous=int((v[order[:-1]]==u[order[1:]]).sum())
    return dict(trip_id=meta["trip_id"],split=meta["split"],observations=len(obs),invalid=int((~obs.valid).sum()),
                missing_lon=int(obs.lon.isna().sum()),missing_lat=int(obs.lat.isna().sum()),missing_timestamp=int(obs.timestamp_s.isna().sum()),
                time_reversals=int((np.diff(good.timestamp_s)<0).sum()),duplicate_times=int((dt==0).sum()),intervals=len(dt),
                one_second_steps=int((dt==1).sum()),gaps=int((dt>threshold).sum()),max_gap_s=dt.max() if len(dt) else None,
                gps_length_m=np.hypot(np.diff(good.sort_values("order_index").x_m),np.diff(good.sort_values("order_index").y_m)).sum(),
                lon_min=good.lon.min(),lon_max=good.lon.max(),lat_min=good.lat.min(),lat_max=good.lat.max(),
                nodes=len(nodes),arcs=len(arcs),zero_length=int((arcs.length_m==0).sum()),
                weak_components=connected_components(graph,directed=False,return_labels=False),
                isolated_nodes=len(nodes)-len(np.union1d(u,v)),reference_steps=len(reference) if reference is not None else None,
                directed_reference_pairs=continuous,reference_pairs=consecutive)


def sample_map(trip,out):
    obs=trip["observations"].query("valid").sort_values("order_index")
    if obs.empty:
        return
    nodes, arcs=trip["nodes"],trip["arcs"]
    xy=nodes[["x_m","y_m"]].to_numpy()
    start,end=xy[arcs.from_node_id],xy[arcs.to_node_id]
    lo=obs[["x_m","y_m"]].min().to_numpy()-100
    hi=obs[["x_m","y_m"]].max().to_numpy()+100
    visible=((np.minimum(start,end)<=hi)&(np.maximum(start,end)>=lo)).all(axis=1)
    coords=nodes[["lon","lat"]].to_numpy()
    lines=np.stack((coords[arcs.from_node_id[visible]],coords[arcs.to_node_id[visible]]),axis=1)
    m=folium.Map(tiles="OpenStreetMap")
    folium.GeoJson({"type":"MultiLineString","coordinates":lines.tolist()},name="Road network in trip window",
                   style_function=lambda _:{"color":"#666666","weight":1}).add_to(m)
    folium.PolyLine(obs[["lat","lon"]].values.tolist(),color="#164ccc",weight=2,tooltip="Cleaned GPS in time order").add_to(m)
    points=folium.FeatureGroup(name="Valid raw GPS observations").add_to(m)
    for lat,lon in obs[["lat","lon"]].itertuples(index=False,name=None):
        folium.CircleMarker([lat,lon],radius=2,stroke=False,fill=True,fill_color="#e45722",fill_opacity=.85).add_to(points)
    m.fit_bounds([[obs.lat.min(),obs.lon.min()],[obs.lat.max(),obs.lon.max()]])
    folium.LayerControl().add_to(m)
    m.save(str(out))


def plot_inventory(table, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),layout="constrained")
    for ax,field,title in zip(axes,["observations","max_gap_s"],["GPS observations per trip","Longest sampling gap (seconds)"]):
        for split,group in table.groupby("split"):
            ax.scatter(range(1,len(group)+1),group[field].sort_values(),s=12,label=split)
        ax.set(xlabel="Trip rank within split",ylabel=title)
        ax.legend()
    fig.savefig(out,dpi=160)
    plt.close(fig)
