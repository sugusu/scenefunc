"""Exercise actual PEFT LoRA enable/disable on a tiny deterministic linear net.

No language model, pretrained weights, optimizer, backward pass, or scene task.
This qualifies one library behavior only, not the 8B model or its GPU footprint.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time


def main(a):
    a.out.mkdir(parents=True,exist_ok=False);start=time.monotonic();result={}
    sys.path.insert(0,str(a.package.resolve()))
    try:
        import torch
        from torch import nn
        import peft,transformers
        from peft import LoraConfig,get_peft_model
        from local_tool_vlm import require_reversible_adapter
        assert peft.__version__=='0.17.1'
        torch.set_num_threads(1)
        class LinearProbe(nn.Module):
            def __init__(self):
                super().__init__();self.config={'model_type':'custom'}
                self.proj=nn.Linear(4,4,bias=False)
                with torch.no_grad():self.proj.weight.copy_(torch.eye(4))
            def forward(self,x):return self.proj(x)
        base=LinearProbe().eval();x=torch.tensor([[1.,2.,3.,4.]])
        with torch.inference_mode():reference=base(x).clone()
        model=get_peft_model(base,LoraConfig(r=2,lora_alpha=2,target_modules=['proj'],lora_dropout=0,bias='none')).eval()
        # Diagnostic constants ensure nonzero adapter effect; not optimization.
        with torch.no_grad():
            for name,p in model.named_parameters():
                if 'lora_A' in name:p.fill_(.25)
                if 'lora_B' in name:p.fill_(.125)
        def snapshot():
            return {n:hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest() for n,p in model.named_parameters()}
        before=snapshot()
        with torch.inference_mode():
            enabled=model(x).clone()
            assert not torch.equal(enabled,reference)
            with model.disable_adapter():
                disabled=model(x).clone()
                assert torch.equal(disabled,reference)
                assert model.get_model_status().enabled is False
                with model.disable_adapter():assert torch.equal(model(x),reference)
                assert model.get_model_status().enabled is False
            restored=model(x).clone()
            assert torch.equal(restored,enabled)
            try:
                with model.disable_adapter():
                    assert torch.equal(model(x),reference)
                    raise RuntimeError('Deliberate exceptional exit')
            except RuntimeError as e:assert str(e)=='Deliberate exceptional exit'
            assert model.get_model_status().enabled is True
            assert torch.equal(model(x),enabled)
        assert snapshot()==before
        require_reversible_adapter(model)
        # A distinct probe: do not merge or mutate the measured reference.
        merged=get_peft_model(LinearProbe().eval(),LoraConfig(r=2,lora_alpha=2,target_modules=['proj'],lora_dropout=0,bias='none')).eval()
        merged.merge_adapter()
        try:require_reversible_adapter(merged)
        except ValueError as exc:assert 'Merged or irregular' in str(exc)
        else:raise AssertionError('Merged adapter accepted')
        files=[Path(peft.__file__).parent/'peft_model.py',Path(peft.__file__).parent/'tuners/lora/layer.py',
               Path(peft.__file__).parent/'tuners/tuners_utils.py']
        result={'status':'passed','peft':peft.__version__,'torch':torch.__version__,'transformers':transformers.__version__,
            'base_output':reference.tolist(),'adapter_output':enabled.tolist(),'disabled_output':disabled.tolist(),
            'base_exact_when_disabled':True,'adapter_restored_after_normal_and_exceptional_exit':True,
            'nested_disable_preserves_disabled_state':True,'all_parameters_unchanged_by_contexts':True,
            'actual_merged_peft_adapter_rejected':True,
            'parameter_hashes':before,'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            'cuda_initialized':torch.cuda.is_initialized()}
    except BaseException as exc:
        result.update(status='failed',error=type(exc).__name__+': '+str(exc));raise
    finally:
        result.update(wall_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            language_model_calls=0,training_updates=0,pretrained_weights_loaded=False,new_scene_generations=0,
            scope='Actual PEFT on a 4x4 linear probe with explicitly assigned diagnostic constants; not actual Qwen/8B adapter validation.')
        (a.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
