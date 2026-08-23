from __future__ import annotations
import importlib.util,json,random,tempfile
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'scripts/build_event_geometry_provenance_v1156.py';spec=importlib.util.spec_from_file_location('geo1156',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
CFG=ROOT/'configs/event_geometry_provenance_v1156.json'
def test_exact_contract_and_supersession():
 d=m.build(CFG);assert d['metrics']['max_gap_deg']==pytest.approx(251.6565487763591,abs=1e-10);assert d['metrics']['occupied_sector_count']==3;assert d['source_binding']['station_observation_rows']==144;assert d['source_binding']['deduplicated_station_rows']==48;assert d['supersession']['old_056_status']=='INVALID_NONREPRODUCIBLE_SUPERSEDED';assert not d['supersession']['old_056_allowed_as_authority_gate'];assert not any(d['prohibitions'].values())
def test_deterministic_byte_output():
 d1=m.build(CFG);d2=m.build(CFG);assert m.canonical(d1)==m.canonical(d2)
def test_formula_and_source_hash_tamper_failclosed():
 base=json.loads(CFG.read_text())
 for key,value in [('ellipsoid','sphere'),('coordinate_order','latitude_longitude'),('direction','station_to_event'),('formal_db_sha256','0'*64)]:
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'c.json';x=dict(base);x[key]=value;p.write_text(json.dumps(x))
   with pytest.raises(ValueError):m.build(p)
def test_source_row_tamper_and_duplicate_failclosed():
 rows=[{'station_id':'a','acquisition_source':'nied_hinet_waveform'},{'station_id':'b','acquisition_source':'nied_hinet_waveform'}];m.validate_source_rows(rows,'nied_hinet_waveform');bad=[dict(x) for x in rows];bad[0]['acquisition_source']='other'
 with pytest.raises(ValueError):m.validate_source_rows(bad,'nied_hinet_waveform')
 with pytest.raises(ValueError):m.validate_source_rows(rows+[dict(rows[0])],'nied_hinet_waveform')
def test_order_invariant_row_hash_and_order_sensitive_tamper():
 rows=[{'station':'b','channel':'U','station_id':'2','lat':'x'},{'station':'a','channel':'E','station_id':'1','lat':'y'}];sh=list(rows);random.Random(4).shuffle(sh);assert m.stable_rows_hash(rows,('station','channel','station_id'))==m.stable_rows_hash(sh,('station','channel','station_id'));tam=[dict(x) for x in rows];tam[0]['lat']='z';assert m.stable_rows_hash(rows,('station','channel','station_id'))!=m.stable_rows_hash(tam,('station','channel','station_id'))
def test_forward_back_projected_distinction():
 d=m.build(CFG);g=d['geodesic_contract'];assert g['direction_used_for_gap']=='event_to_station_forward_azimuth';assert g['back_azimuth_recorded_not_used_for_gap'];assert not g['projected_xy_azimuth_used'];assert all('station_to_event_back_azimuth_deg' in x for x in d['station_geodesics_station_order'])
