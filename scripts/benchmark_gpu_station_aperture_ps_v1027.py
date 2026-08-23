#!/usr/bin/env python3
"""v1027: extend the audited v1023 CUDA station kernel to the S branch."""
from __future__ import annotations

from pathlib import Path


SOURCE=Path(__file__).with_name("benchmark_gpu_station_aperture_v1023.py")
source=SOURCE.read_text(encoding="utf-8")
old='''    if phase!="P":raise NotImplementedError("v1023 proof currently validates P; S remains CPU until transverse branch audit")
    device=torch.device("cuda")'''
new='''    if phase not in {"P", "S"}: raise ValueError(phase)
    device=torch.device("cuda")'''
if source.count(old)!=1:raise RuntimeError("v1023 phase anchor differs")
source=source.replace(old,new)

old='''        projector=np.eye(3)-np.outer(direct_ray,direct_ray)
        combined=torch.zeros(len(xyz),device=device,dtype=torch.complex128)'''
new='''        direct_polarization=stream.dominant_direct_polarization(
            migration,components,orientation,station_pick,direct_ray,phase)
        projector=np.eye(3,dtype=np.complex128)-np.outer(direct_polarization,np.conjugate(direct_polarization))
        combined=torch.zeros(len(xyz),device=device,dtype=torch.complex128)'''
if source.count(old)!=1:raise RuntimeError("v1023 projector anchor differs")
source=source.replace(old,new)

old='''            effective=projector@inverse_matrix
            covariance=projector@(inverse_matrix@inverse_matrix.T)@projector.T
            effective_t=torch.as_tensor(effective,device=device,dtype=torch.complex128)
            displacement=effective_t@samples_t[component_indices][:,selected]
            selected_ray=ray[:,selected]
            scalar=torch.sum(displacement*selected_ray,dim=0)
            covariance_t=torch.as_tensor(covariance,device=device,dtype=torch.float64)
            variance=torch.einsum("in,ij,jn->n",selected_ray,covariance_t,selected_ray)
            gain=torch.sqrt(torch.clamp(variance,min=1e-12))
            combined[selected]=scalar/gain;reconstructed[selected]=True'''
new='''            effective=projector@inverse_matrix
            covariance=projector@(inverse_matrix@inverse_matrix.T)@np.conjugate(projector.T)
            covariance=np.asarray(np.real(covariance),float)
            effective_t=torch.as_tensor(effective,device=device,dtype=torch.complex128)
            displacement=effective_t@samples_t[component_indices][:,selected]
            selected_ray=ray[:,selected]
            parallel=torch.sum(displacement*selected_ray,dim=0)
            covariance_t=torch.as_tensor(covariance,device=device,dtype=torch.float64)
            parallel_variance=torch.einsum("in,ij,jn->n",selected_ray,covariance_t,selected_ray)
            if phase=="P":
                scalar=parallel
                variance=parallel_variance
            else:
                transverse=displacement-selected_ray*parallel
                scalar=torch.sqrt(torch.sum(transverse*transverse,dim=0))
                variance=torch.trace(covariance_t)-parallel_variance
            gain=torch.sqrt(torch.clamp(variance,min=1e-12))
            combined[selected]=scalar/gain;reconstructed[selected]=True'''
if source.count(old)!=1:raise RuntimeError("v1023 projection anchor differs")
source=source.replace(old,new)

old='''"P_path_validated":phase=="P","all_station_node_geometry_on_cuda":True,'''
new='''"phase_path_validated":phase in {"P","S"},"all_station_node_geometry_on_cuda":True,'''
if source.count(old)!=1:raise RuntimeError("v1023 check anchor differs")
source=source.replace(old,new)
source=source.replace("cuda-station-node-scattered-aperture-v1023","cuda-station-node-scattered-aperture-ps-v1027")
source=source.replace("cuda-station-node-scattered-aperture-audit-v1023","cuda-station-node-scattered-aperture-ps-audit-v1027")
namespace={"__name__":"__main__","__file__":str(SOURCE)}
exec(compile(source,str(SOURCE)+"#ps-v1027","exec"),namespace)
