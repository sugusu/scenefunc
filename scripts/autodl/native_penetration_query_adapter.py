"""Ordinary alternative contact-query backend, preserving native report filters.

This is a bounded engineering control, not a learned scene algorithm or general
physics certificate. The original source file and all report thresholds stay
unchanged. The wrapper applies only in its explicit temporary runtime context.
"""
from contextlib import contextmanager
from types import SimpleNamespace


class PenetrationQueryProxy:
    def __init__(self,query,receipt):self.query=query;self.receipt=receipt
    def __getattr__(self,name):return getattr(self.query,name)
    def ComputeSignedDistancePairwiseClosestPoints(self,max_distance=0.0):
        if max_distance!=0.0:
            raise ValueError('Penetration backend cannot supply separated-pair distances')
        pairs=self.query.ComputePointPairPenetration()
        inspector=self.query.inspector()
        self.receipt.extend(dict(a=inspector.GetName(x.id_A),b=inspector.GetName(x.id_B),
            depth_m=float(x.depth),p_WCa=x.p_WCa.tolist(),p_WCb=x.p_WCb.tolist()) for x in pairs)
        return [SimpleNamespace(id_A=x.id_A,id_B=x.id_B,distance=-float(x.depth)) for x in pairs]


@contextmanager
def native_penetration_query(physics_module,receipt):
    original=physics_module.create_drake_plant_and_scene_graph_from_scene
    class Port:
        def __init__(self,port):self.port=port
        def Eval(self,context):return PenetrationQueryProxy(self.port.Eval(context),receipt)
    class Graph:
        def __init__(self,graph):self.graph=graph
        def __getattr__(self,name):return getattr(self.graph,name)
        def get_query_output_port(self):return Port(self.graph.get_query_output_port())
    def factory(*args,**kwargs):
        plant,graph=original(*args,**kwargs)
        return plant,Graph(graph)
    physics_module.create_drake_plant_and_scene_graph_from_scene=factory
    try:yield
    finally:physics_module.create_drake_plant_and_scene_graph_from_scene=original
