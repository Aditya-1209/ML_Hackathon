import json
import folium
from .data_io import load_processed_trip


def save_map(workspace,trip_id,predictions,out):
    trip=load_processed_trip(workspace/"data/processed",trip_id)
    obs=trip["observations"].query("valid").sort_values("order_index")
    nodes=trip["nodes"].set_index("node_id")
    arcs=trip["arcs"].set_index("arc_id")
    m=folium.Map(tiles="OpenStreetMap")
    gps=folium.FeatureGroup(name="Raw GPS").add_to(m)
    for row in obs.itertuples():
        folium.CircleMarker([row.lat,row.lon],radius=2,fill=True,color="#e57b18").add_to(gps)
    roads=folium.FeatureGroup(name="Predicted road segments").add_to(m)
    selected=predictions[predictions.trip_id.eq(trip_id)].predicted_arc_id.dropna().astype(int).unique()
    for arc_id in selected:
        arc=arcs.loc[arc_id]
        coords=nodes.loc[[arc.from_node_id,arc.to_node_id],["lat","lon"]].values.tolist()
        folium.PolyLine(coords,color="#0564c5",weight=4).add_to(roads)
    # References appear only in this offline diagnostic map.
    ref=workspace/"data/reference"/trip_id/"route.parquet"
    if ref.exists():
        import pandas as pd
        reference=folium.FeatureGroup(name="Supplied reference route",show=False).add_to(m)
        for arc_id in pd.read_parquet(ref).arc_id.unique():
            arc=arcs.loc[arc_id]
            folium.PolyLine(nodes.loc[[arc.from_node_id,arc.to_node_id],["lat","lon"]].values.tolist(),
                            color="#27953b",weight=3).add_to(reference)
    m.fit_bounds([[obs.lat.min(),obs.lon.min()],[obs.lat.max(),obs.lon.max()]])
    folium.LayerControl().add_to(m)
    m.save(str(out))
