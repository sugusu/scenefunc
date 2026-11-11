"""Ordinary exact rational coverage for a convex polygon by source triangles.

All supplied finite floats are interpreted as exact dyadic rationals. This is
not an epsilon/hole-size heuristic and never turns a genuinely uncovered area
into support. Bounded fragmentation returns unknown instead of an invented label.
"""
from fractions import Fraction as F


def rational_polygon(points):
    p=[tuple(F(float(v)) for v in q) for q in points]
    if p and p[0]==p[-1]:p=p[:-1]
    return p


def cross(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
def signed_area(p):return sum((p[i][0]*p[(i+1)%len(p)][1]-p[(i+1)%len(p)][0]*p[i][1]) for i in range(len(p)))/2 if len(p)>=3 else F(0)


def halfplane(p,a,b,inside):
    if not p:return []
    result=[]
    def keep(d):return d>=0 if inside else d<=0
    for s,e in zip(p,p[1:]+p[:1]):
        ds,de=cross(a,b,s),cross(a,b,e)
        ks,ke=keep(ds),keep(de)
        if ks!=ke:
            t=ds/(ds-de)
            result.append((s[0]+t*(e[0]-s[0]),s[1]+t*(e[1]-s[1])))
        if ke:result.append(e)
    compact=[]
    for q in result:
        if not compact or compact[-1]!=q:compact.append(q)
    if len(compact)>1 and compact[0]==compact[-1]:compact.pop()
    return compact if len(compact)>=3 and signed_area(compact)!=0 else []


def subtract_triangle(poly,triangle):
    t=list(triangle)
    if signed_area(t)<0:t=t[::-1]
    if signed_area(t)==0:return [poly]
    residual=[];inside=poly
    for a,b in zip(t,t[1:]+t[:1]):
        outside=halfplane(inside,a,b,False)
        if outside:residual.append(outside)
        inside=halfplane(inside,a,b,True)
        if not inside:break
    return residual


def coverage(points,triangles,fragment_cap=256):
    target=rational_polygon(points)
    if signed_area(target)<0:target=target[::-1]
    assert len(target)>=3 and signed_area(target)>0
    residual=[target];used=[]
    for index,vertices in triangles:
        t=rational_polygon(vertices)
        residual=[q for p in residual for q in subtract_triangle(p,t)]
        used.append(int(index))
        if len(residual)>fragment_cap:
            return dict(status='unknown_fragment_cap',source_triangles=used,fragment_cap=fragment_cap)
        if not residual:break
    remaining=sum((abs(signed_area(p)) for p in residual),F(0))
    return dict(status='covered' if remaining==0 else 'uncovered',
        target_area_exact=str(abs(signed_area(target))),uncovered_area_exact=str(remaining),
        uncovered_area_m2=float(remaining),source_triangles=used,residual_fragments=len(residual),
        rational_semantics='Exact finite input-coordinate rationals; no epsilon or area-threshold admission')
