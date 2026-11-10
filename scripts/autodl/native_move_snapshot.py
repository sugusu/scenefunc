"""Ordinary move/rescale branch snapshot, including the native guard history.

The old rescale snapshot is unchanged. This extension is qualified only by the
specified native probe; it does not cover asset generation, arbitrary new files,
other stages, external workers, GPU/RNG state or a whole simulator snapshot.
"""
import inspect

from native_edit_snapshot import NativeEditSnapshot


def actual_loop_detectors(furniture_tools):
    from scenesmith.agent_utils.loop_detector import LoopDetector
    result={}
    for name in ('_add_furniture_to_scene_impl','_move_furniture_impl','_remove_furniture_impl'):
        wrapped=getattr(furniture_tools,name)
        values=inspect.getclosurevars(wrapped).nonlocals.values()
        found=[v for v in values if isinstance(v,LoopDetector)]
        if len(found)!=1:raise ValueError('Expected original native loop-detector closure: '+name)
        result[name]=found[0]
    if len({id(v) for v in result.values()})!=1:
        raise ValueError('Requalify changed per-tool guard topology')
    return result


class NativeMoveSnapshot(NativeEditSnapshot):
    admitted_tools=frozenset({'rescale_furniture_tool','move_furniture_tool'})

    def __init__(self,env,*,owned_root,snapshot_dir):
        super().__init__(env,owned_root=owned_root,snapshot_dir=snapshot_dir)
        self.furniture_tools=env.agent.furniture_tools
        self.detectors=actual_loop_detectors(self.furniture_tools)
        self.detector=next(iter(self.detectors.values()))
        self.history=list(self.detector.recent_calls)
        self.settings=(self.detector.enabled,self.detector.max_attempts,
                       self.detector.window_size,self.detector.recent_calls.maxlen)
        self.noise_profile=self.furniture_tools.active_noise_profile

    def restore(self,*,restore_assets=True):
        super().restore(restore_assets=restore_assets)
        now=actual_loop_detectors(self.furniture_tools)
        if any(now[k] is not self.detectors[k] for k in now):
            raise RuntimeError('Original tool/guard closure identities changed')
        settings=(self.detector.enabled,self.detector.max_attempts,
                  self.detector.window_size,self.detector.recent_calls.maxlen)
        if settings!=self.settings:
            raise RuntimeError('Branch changed guard settings; do not disable or relax guard')
        if self.furniture_tools.active_noise_profile is not self.noise_profile:
            raise RuntimeError('Noise profile changed outside admitted operations')
        # Restore the parent's exact logical guard state in place. Resetting
        # always to empty would erase legitimate repeated calls in the parent.
        self.detector.recent_calls.clear()
        self.detector.recent_calls.extend(self.history)
