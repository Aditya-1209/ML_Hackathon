from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from pathsnap_ml import data_io
from pathsnap_ml.models import knn, logistic_regression


def test_rows_and_reference_boundary(tmp_path):
    folder=tmp_path/"train"/"001"
    folder.mkdir(parents=True)
    content={"track":"8 49 3\n\n8 49 1\n8 49 1\n999 49 5\n","nodes":"8 49\n8.001 49\n",
             "arcs":"0 1\n0 1\n1 1\n"}
    fields={}
    for kind,value in content.items():
        f=folder/f"001.{kind}"
        f.write_text(value)
        fields[f"{kind}_path"]=str(f.relative_to(tmp_path))
    row=SimpleNamespace(trip_id="001",split="train",**fields)
    cfg=dict(graph_directed=True,directionality_basis="test")
    result=data_io.prepare_trip(tmp_path,row,cfg)
    obs=result["observations"]
    assert obs.obs_id.tolist()==[0,1,2,3,4]
    assert obs.order_index.fillna(-1).tolist()==[2,-1,0,1,-1]
    assert result["arcs"].arc_id.tolist()==[0,1,2]
    assert result["arcs"].length_m.iloc[-1]==0
    for k in ("observations","nodes","arcs"):
        data_io.write_table(result[k],tmp_path/"processed/001"/f"{k}.parquet")
    import json
    (tmp_path/"processed/001/metadata.json").write_text(json.dumps(result["metadata"]))
    pd.testing.assert_frame_equal(obs,data_io.load_processed_trip(tmp_path/"processed","001")["observations"])
    (folder/"001.arcs").write_text("0 9\n")
    with pytest.raises(ValueError,match="endpoint"):
        data_io.prepare_trip(tmp_path,row,cfg)
    row.split="evaluator_test"
    with pytest.raises(ValueError,match="Development"):
        data_io.prepare_trip(tmp_path,row,cfg)


@pytest.mark.parametrize("module",[knn,logistic_regression])
def test_model_pipelines(module):
    X=np.array([[i, np.nan] for i in range(10)])
    model=module.build_estimator()
    model.fit(X,[0]*5+[1]*5)
    assert model.named_steps["scale"].n_features_in_==2
    scores=model.predict_proba(X)
    assert scores.shape==(10,2)
    np.testing.assert_allclose(scores.sum(axis=1),1)
