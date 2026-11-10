"""Ordinary world-AABB penetration upper-bound veto for a native distance query.

This does not calculate exact penetration or certify overall room validity. It
only removes a candidate that cannot exceed an unchanged reporting threshold.
Remaining original distances and ordering are preserved, including their limits.
"""
from contextlib import contextmanager
import itertools
import numpy as np


def outward_world_bounds(points, matrix):
    """Enclose affine transforms of the supplied finite-double vertices.

    Each multiplication/addition is rounded outward separately. This avoids
    treating a single final nextafter as a complete dot-product error bound.
    The geometry and transform doubles themselves are the represented inputs.
    """
    points=np.asarray(points,dtype=np.float64);matrix=np.asarray(matrix,dtype=np.float64)
    if points.ndim!=2 or points.shape[1]!=3 or not len(points) or matrix.shape!=(4,4):
        raise ValueError('Invalid geometry array')
    if not np.isfinite(points).all() or not np.isfinite(matrix).all():
        raise ValueError('Nonfinite geometry cannot certify a veto')
    if not np.array_equal(matrix[3],[0.,0.,0.,1.]):raise ValueError('Expected an affine transform')
    lows=[];highs=[]
    for axis in range(3):
        low=np.zeros(len(points));high=np.zeros(len(points))
        for k in range(3):
            prod=points[:,k]*matrix[axis,k]
            low=np.nextafter(low+np.nextafter(prod,-np.inf),-np.inf)
            high=np.nextafter(high+np.nextafter(prod,np.inf),np.inf)
        lows.append(np.nextafter(low+matrix[axis,3],-np.inf).min())
        highs.append(np.nextafter(high+matrix[axis,3],np.inf).max())
    if not np.isfinite(lows).all() or not np.isfinite(highs).all():
        raise ValueError('Overflow cannot certify a veto')
    return np.asarray(lows),np.asarray(highs)


def penetration_upper_bound(first,second):
    """A separating translation for enclosing boxes bounds shape penetration.

    Separating the AABBs necessarily separates their contained shapes. At least
    one signed axis translation of the reported length suffices; minimizing
    these six lengths yields an upper bound, never a positive lower bound.
    """
    amin,amax=first;bmin,bmax=second
    moves=np.concatenate((np.nextafter(amax-bmin,np.inf),np.nextafter(bmax-amin,np.inf)))
    return max(0.,float(moves.min()))


class ConservativeQuery:
    def __init__(self,query,receipt,threshold):
        if threshold<=0 or not np.isfinite(threshold):raise ValueError('Positive existing threshold required')
        self.query=query;self.receipt=receipt;self.threshold=threshold;self.cache={}
    def __getattr__(self,name):return getattr(self.query,name)
    def bounds(self,gid):
        if gid in self.cache:return self.cache[gid]
        inspector=self.query.inspector();shape=inspector.GetShape(gid)
        name=type(shape).__name__
        if name=='Box':
            half=np.array([shape.width(),shape.depth(),shape.height()])/2
            points=np.array(list(itertools.product((-1.,1.),repeat=3)))*half
        elif name in ('Mesh','Convex'):
            hull=shape.GetConvexHull()
            points=np.asarray([hull.vertex(i) for i in range(hull.num_vertices())])
        else:raise ValueError('Unqualified shape type '+name)
        pose=self.query.GetPoseInWorld(inspector.GetFrameId(gid)).multiply(inspector.GetPoseInFrame(gid))
        result=outward_world_bounds(points,pose.GetAsMatrix4());self.cache[gid]=result
        return result
    def ComputeSignedDistancePairwiseClosestPoints(self,max_distance=0.0):
        if max_distance!=0.0:raise ValueError('Control qualified for penetrating-pair reports only')
        original=self.query.ComputeSignedDistancePairwiseClosestPoints(max_distance=max_distance)
        inspector=self.query.inspector();kept=[]
        for pair in original:
            record=dict(a=inspector.GetName(pair.id_A),b=inspector.GetName(pair.id_B),
                        original_signed_distance_m=float(pair.distance),threshold_m=self.threshold)
            try:
                a=self.bounds(pair.id_A);b=self.bounds(pair.id_B);upper=penetration_upper_bound(a,b)
                # Native reporting includes distance == -threshold, so equality
                # must be retained. This is a strict mathematical veto only.
                veto=upper<self.threshold
                record.update(upper_bound_m=upper,veto=veto,
                              first_enclosing_AABB=[v.tolist() for v in a],
                              second_enclosing_AABB=[v.tolist() for v in b],
                              original_depth_exceeds_upper_bound=-pair.distance>upper)
            except (ValueError,RuntimeError,AttributeError) as exc:
                veto=False;record.update(veto=False,bounds_unknown=str(exc))
            self.receipt.append(record)
            if not veto:kept.append(pair)
        return kept


@contextmanager
def native_conservative_contact_query(physics_module,receipt,*,threshold):
    """Temporary same-reporter query hook; never modifies upstream source files."""
    original=physics_module.create_drake_plant_and_scene_graph_from_scene
    class Port:
        def __init__(self,port):self.port=port
        def Eval(self,context):return ConservativeQuery(self.port.Eval(context),receipt,threshold)
    class Graph:
        def __init__(self,graph):self.graph=graph
        def __getattr__(self,name):return getattr(self.graph,name)
        def get_query_output_port(self):return Port(self.graph.get_query_output_port())
    def factory(*args,**kwargs):
        plant,graph=original(*args,**kwargs);return plant,Graph(graph)
    physics_module.create_drake_plant_and_scene_graph_from_scene=factory
    try:yield
    finally:physics_module.create_drake_plant_and_scene_graph_from_scene=original
