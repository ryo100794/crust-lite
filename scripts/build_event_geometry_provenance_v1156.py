#!/usr/bin/env python3
"""Build a deterministic, formal-DB-bound event/station geometry contract."""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
from typing import Any
import duckdb
import pyproj
from pyproj import Geod

ROOT=Path('/workspace/equake/crust-lite')
SCHEMA='formal-hinet-event-station-geometry-provenance-v1156'
REQ='NR-SCIENCE-EVENT-GEOMETRY-PROVENANCE-064'
EVENT_SQL="SELECT event_id,CAST(time_utc AS VARCHAR),lat,lon,depth_km,magnitude,magnitude_type,catalog_source,is_authenticated_source FROM event WHERE event_id=? ORDER BY event_id"
STATION_SQL="SELECT station_id,network,station,CAST(location AS VARCHAR),channel,lat,lon,station_elevation_m,station_depth_m,acquisition_source,acquisition_batch FROM station_observation WHERE event_id=? ORDER BY station,channel,station_id"
def sha256(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def canonical(value:Any)->bytes:return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def canonical_hash(value:Any)->str:return hashlib.sha256(canonical(value)).hexdigest()
def fhex(x:Any)->str|None:return None if x is None else float(x).hex()
def fnum(x:Any)->float|None:return None if x is None else float(x)
def stable_rows_hash(rows:list[dict[str,Any]],keys:tuple[str,...])->str:
 return canonical_hash(sorted(rows,key=lambda r:tuple(str(r.get(k,'')) for k in keys)))
def validate_config(c:dict[str,Any])->None:
 expected={'schema':'event-geometry-provenance-build-config-v1156','requirement_id':REQ,'ellipsoid':'WGS84','coordinate_order':'longitude_latitude','direction':'event_to_station_forward_azimuth','duplicate_policy':'one_exact_coordinate_per_station_after_channel_dedup','sector_width_deg':45.0,'gap_definition':'maximum_adjacent_sorted_forward_azimuth_difference_including_circular_wrap'}
 for k,v in expected.items():
  if c.get(k)!=v:raise ValueError(f'config contract mismatch: {k}')
 if len(str(c.get('formal_db_sha256','')))!=64:raise ValueError('formal DB hash is not full64')
def validate_source_rows(rows:list[dict[str,Any]],expected_source:str)->None:
 if not rows:raise ValueError('station observation rows absent')
 if any(r['acquisition_source']!=expected_source for r in rows):raise ValueError('non-formal acquisition source')
 if len({r['station_id'] for r in rows})!=len(rows):raise ValueError('duplicate station observation natural key')
def atomic(path:Path,payload:dict[str,Any])->None:
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(payload,indent=2,sort_keys=True,ensure_ascii=False)+'\n');os.replace(tmp,path)
def build(config_path:Path)->dict[str,Any]:
 cfg=json.loads(config_path.read_text());validate_config(cfg);db_path=ROOT/cfg['formal_db_path'];actual_db=sha256(db_path)
 if actual_db!=cfg['formal_db_sha256']:raise ValueError('formal DB SHA-256 mismatch')
 con=duckdb.connect(str(db_path),read_only=True);eid=cfg['event_id'];events=con.execute(EVENT_SQL,[eid]).fetchall()
 if len(events)!=1:raise ValueError('formal event row count differs')
 e=events[0];event={'event_id':e[0],'time_utc':e[1],'lat':float(e[2]),'lat_binary64_hex':fhex(e[2]),'lon':float(e[3]),'lon_binary64_hex':fhex(e[3]),'depth_km':float(e[4]),'magnitude':float(e[5]),'magnitude_type':e[6],'catalog_source':e[7],'is_authenticated_source':str(e[8]).lower()=='true'}
 if not event['is_authenticated_source'] or 'NIED Hi-net' not in event['catalog_source']:raise ValueError('event is not authenticated NIED Hi-net')
 raw=[]
 for r in con.execute(STATION_SQL,[eid]).fetchall():
  raw.append({'station_id':r[0],'network':r[1],'station':r[2],'location':r[3],'channel':r[4],'lat':float(r[5]),'lat_binary64_hex':fhex(r[5]),'lon':float(r[6]),'lon_binary64_hex':fhex(r[6]),'station_elevation_m':fnum(r[7]),'station_depth_m':fnum(r[8]),'acquisition_source':r[9],'acquisition_batch':r[10]})
 validate_source_rows(raw,cfg['station_acquisition_source']);stations=[]
 for name in sorted({r['station'] for r in raw}):
  rr=[r for r in raw if r['station']==name];coords={(r['lat_binary64_hex'],r['lon_binary64_hex']) for r in rr};channels=sorted(r['channel'] for r in rr)
  if len(coords)!=1 or channels!=['E','N','U']:raise ValueError(f'station coordinate/channel dedup differs: {name}')
  stations.append({'station':name,'base_station_id':'.'+name+'.','lat':rr[0]['lat'],'lat_binary64_hex':rr[0]['lat_binary64_hex'],'lon':rr[0]['lon'],'lon_binary64_hex':rr[0]['lon_binary64_hex'],'source_channels':channels,'source_row_count':len(rr)})
 if len(raw)!=cfg['expected_station_observation_rows'] or len(stations)!=cfg['expected_unique_stations']:raise ValueError('formal station counts differ')
 geod=Geod(ellps='WGS84');azrows=[]
 for s in stations:
  fwd,back,dist=geod.inv(event['lon'],event['lat'],s['lon'],s['lat']);azrows.append({**s,'event_to_station_forward_azimuth_deg':float(fwd%360.0),'station_to_event_back_azimuth_deg':float(back%360.0),'geodesic_distance_m':float(dist)})
 ordered=sorted(azrows,key=lambda r:(r['event_to_station_forward_azimuth_deg'],r['station']));gaps=[]
 for i,a in enumerate(ordered):
  b=ordered[(i+1)%len(ordered)];start=a['event_to_station_forward_azimuth_deg'];end=b['event_to_station_forward_azimuth_deg']+(360.0 if i==len(ordered)-1 else 0.0);gaps.append({'from_station':a['station'],'to_station':b['station'],'start_deg':start,'end_unwrapped_deg':end,'gap_deg':end-start,'wrap':i==len(ordered)-1})
 maxgap=max(gaps,key=lambda g:g['gap_deg']);sectors=sorted({int(r['event_to_station_forward_azimuth_deg']//cfg['sector_width_deg']) for r in ordered})
 if abs(maxgap['gap_deg']-cfg['expected_max_gap_deg'])>cfg['numeric_tolerance_deg'] or len(sectors)!=cfg['expected_occupied_sector_count']:raise ValueError('expected geometry metric differs')
 schema=[]
 for table in ('event','station_observation'):
  cols=con.execute("SELECT column_name,data_type,is_nullable,ordinal_position FROM information_schema.columns WHERE table_name=? ORDER BY ordinal_position",[table]).fetchall();schema.append({'table':table,'columns':[list(x) for x in cols]})
 con.close();event_hash=canonical_hash(event);raw_hash=stable_rows_hash(raw,('station','channel','station_id'));dedup_hash=stable_rows_hash(stations,('station',))
 return {'schema':SCHEMA,'requirement_id':REQ,'status':'IMMUTABLE_CANDIDATE_NOT_ACTIVATED','event_id':eid,'source_binding':{'formal_db_path':cfg['formal_db_path'],'formal_db_sha256':actual_db,'config_path':str(config_path.relative_to(ROOT)),'config_sha256':sha256(config_path),'event_table_schema_sha256':canonical_hash(schema[0]),'station_observation_table_schema_sha256':canonical_hash(schema[1]),'event_query':EVENT_SQL,'station_query':STATION_SQL,'event_row_sha256':event_hash,'station_observation_rows':len(raw),'station_observation_rows_sha256':raw_hash,'deduplicated_station_rows':len(stations),'deduplicated_station_rows_sha256':dedup_hash,'station_acquisition_source':cfg['station_acquisition_source']},'event':event,'station_observation_rows':raw,'deduplicated_stations':stations,'geodesic_contract':{'library':'pyproj','pyproj_version':pyproj.__version__,'proj_version':pyproj.proj_version_str,'ellipsoid':'WGS84','coordinate_order':'Geod.inv(event_lon,event_lat,station_lon,station_lat)','direction_used_for_gap':'event_to_station_forward_azimuth','back_azimuth_recorded_not_used_for_gap':True,'projected_xy_azimuth_used':False,'formula':'fwd_deg = Geod(ellps=WGS84).inv(event_lon,event_lat,station_lon,station_lat)[0] % 360','duplicate_policy':cfg['duplicate_policy']},'station_geodesics_station_order':azrows,'station_geodesics_azimuth_order':ordered,'circular_gaps':gaps,'metrics':{'max_gap_deg':maxgap['gap_deg'],'max_gap_from_station':maxgap['from_station'],'max_gap_to_station':maxgap['to_station'],'angular_span_deg':360.0-maxgap['gap_deg'],'sector_width_deg':cfg['sector_width_deg'],'occupied_sector_indices':sectors,'occupied_sector_count':len(sectors),'sector_definition':'count unique floor(event_to_station_forward_azimuth_deg / 45) bins on [0,360)'},'supersession':{'old_056_gap_deg':cfg['superseded_old_gap_deg'],'old_056_status':'INVALID_NONREPRODUCIBLE_SUPERSEDED','old_056_absolute_error_deg':abs(maxgap['gap_deg']-cfg['superseded_old_gap_deg']),'old_056_allowed_as_authority_gate':False,'replacement_metric':'metrics.max_gap_deg'},'prohibitions':{'GS_or_model_accessed':False,'event_reselected':False,'science_package_modified':False,'pointer_modified':False,'public_modified':False}}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',default=str(ROOT/'configs/event_geometry_provenance_v1156.json'));ap.add_argument('--output');a=ap.parse_args();cfg=Path(a.config);cfg=cfg if cfg.is_absolute() else ROOT/cfg;d=build(cfg);out=Path(a.output) if a.output else ROOT/json.loads(cfg.read_text())['output_path'];out=out if out.is_absolute() else ROOT/out;atomic(out,d);print(json.dumps({'output':str(out),'sha256':sha256(out),'max_gap_deg':d['metrics']['max_gap_deg'],'sectors':d['metrics']['occupied_sector_count']},indent=2))
if __name__=='__main__':main()
