import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from pathsnap_ml.contracts import FEATURES,KEY,score_keys
from pathsnap_ml.candidates import make_candidates
from pathsnap_ml.features import make_features
from pathsnap_ml.decoder import decode_trip
from pathsnap_ml.evaluation import evaluate
from pathsnap_ml.road_routing import Router
from pathsnap_ml.experiments import positive_scores


def tiny_trip():
    nodes=pd.DataFrame(dict(trip_id=["T"]*4,node_id=[0,1,2,3],x_m=[0.,100.,200.,400.],y_m=[0.]*4))
    arcs=pd.DataFrame(dict(trip_id=["T"]*3,arc_id=[0,1,2],from_node_id=[0,1,2],to_node_id=[1,2,3],length_m=[100.,100.,200.]))
    obs=pd.DataFrame(dict(trip_id=["T"]*3,obs_id=[0,1,2],order_index=[0,1,2],timestamp_s=[0.,10.,20.],
                           x_m=[10.,110.,210.],y_m=[1.,1.,1.],valid=[True]*3))
    meta=dict(trip_id="T",graph_directed=True)
    return obs,nodes,arcs,meta


def test_candidate_feature_decoder_contract():
    obs,nodes,arcs,meta=tiny_trip()
    cfg=dict(radius_m=50,fallback_radius_m=200,max_candidates=8,heading_min_step_m=2,heading_max_gap_s=60)
    candidates=make_candidates(obs,nodes,arcs,meta,cfg)
    features=make_features(obs,nodes,arcs,candidates,meta,cfg)
    assert list(features)==KEY+FEATURES
    scores=candidates[KEY].copy()
    scores["p_match"]=np.where(scores.candidate_arc_id.eq(scores.obs_id),.9,.1)
    config=dict(beta_m=50,epsilon=1e-9,gap_s=60,max_speed_mps=55,slack_m=30)
    expected,route=decode_trip(obs,nodes,arcs,candidates,scores,meta,config)
    actual,_=decode_trip(obs,nodes,arcs,candidates,scores.sample(frac=1,random_state=1),meta,config)
    assert expected.predicted_arc_id.tolist()==[0,1,2]
    assert expected.equals(actual) and route["runs"][0]["arc_ids"]==[0,1,2]
    with pytest.raises(ValueError,match="different keys"):
        score_keys(candidates,scores.iloc[1:])


def test_target_miss_remains_in_denominator():
    obs,_,_,_=tiny_trip()
    labels=pd.DataFrame(dict(trip_id=["T"]*3,obs_id=[0,1,2],target_arc_id=[0,1,2],label_status=["assigned"]*3))
    scores=pd.DataFrame(dict(trip_id=["T"]*2,obs_id=[0,1],candidate_arc_id=[0,0],p_match=[.9,.1]))
    result=evaluate(scores,labels,obs)
    assert result["top1_accuracy"]==pytest.approx(1/3)
    assert result["candidate_miss_rate"]==pytest.approx(2/3)
    assert result["no_candidate_rate"]==pytest.approx(1/3)


def test_projected_position_directionality():
    obs,nodes,arcs,meta=tiny_trip()
    router=Router(nodes,arcs,meta)
    left=SimpleNamespace(candidate_arc_id=0,projection_fraction=.8)
    right=SimpleNamespace(candidate_arc_id=0,projection_fraction=.2)
    assert not np.isfinite(router.between(left,right,1000)[0])
    router=Router(nodes,arcs,{**meta,"graph_directed":False})
    assert router.between(left,right,1000)[0]==pytest.approx(60)


def test_probability_class_order():
    class Estimator:
        classes_=np.array([1,0])
        def predict_proba(self,X):
            return np.array([[.8,.2]])
    frame=pd.DataFrame([[0.]*8],columns=FEATURES)
    assert positive_scores(Estimator(),frame).tolist()==[.8]
