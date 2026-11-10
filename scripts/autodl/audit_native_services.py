"""Compose the pinned real configuration and exercise the production preflight.

No native SDK imports, provider clients, model loads, servers, or network calls.
The main_async tripwire is a test double; the Hydra composition is real.
"""
import argparse
import ast
import asyncio
import hashlib
import html
import json
from pathlib import Path
import time
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from full_generation_bridge import GenerationLedger, run_native_trial
from native_service_contract import inspect_config, reject_known_external_images


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')


def compose_native(source, integration):
    with initialize_config_dir(config_dir=str(integration.resolve()), version_base=None):
        return compose(config_name='scene_designer', overrides=[
            f'hydra.searchpath=[file://{(source / "configurations").resolve()}]',
            '+name=native_service_audit'])


async def main(a):
    a.out.mkdir(parents=True, exist_ok=False)
    tick = time.monotonic()
    manifest = json.loads((a.source.parent/'source_manifest.json').read_text())
    for row in manifest['files']:
        assert sha(a.source/row['path']) == row['sha256']
    old = [a.native_source, a.integration/'scene_designer.yaml', a.asset_manager]
    before = {str(p): sha(p) for p in old}
    cfg = compose_native(a.source, a.integration)
    rows = inspect_config(cfg)
    save(a.out/'composed_service_fields.json', rows)

    # Read actual stage loop; do not infer implemented stages from config keys.
    tree = ast.parse(a.native_source.read_text())
    main_node = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'main_async')
    loops = [n for n in ast.walk(main_node) if isinstance(n, ast.For)
             and isinstance(n.target, ast.Tuple)
             and isinstance(n.target.elts[0], ast.Name) and n.target.elts[0].id == 'stage']
    assert len(loops) == 1 and isinstance(loops[0].iter, ast.List)
    stages = ['floor_plan'] + [n.elts[0].value for n in loops[0].iter.elts]
    assert stages == ['floor_plan', 'furniture', 'wall', 'ceiling']

    manager = ast.parse(a.asset_manager.read_text())
    cls = next(n for n in manager.body if isinstance(n, ast.ClassDef) and n.name == 'AssetManager')
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    eager = [n for n in init.body if isinstance(n, ast.Assign)
             and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
             and n.value.func.id == 'create_image_generator']
    assert len(eager) == 1  # Unconditional top-level constructor statement.
    findings = dict(controller_stages=stages, config_stages=[r['stage'] for r in rows],
        omitted_config_stage='manipuland',
        furniture_tool_scope='generate_assets explicitly requests FURNITURE and excludes surface objects',
        eager_image_constructor_line=eager[0].lineno,
        no_full_task_coverage=True,
        limitation='No proof of zero hidden provider calls elsewhere; SDK summarization and all remaining service branches still need qualification.')
    save(a.out/'source_findings.json', findings)

    # Real composed cfg, production guard, no model/server construction.
    calls=[]
    async def forbidden_main(args):
        calls.append('unexpected_main_entry')
        raise AssertionError('Native startup must not be reached')
    def composer(name, out):
        calls.append('compose')
        return cfg, None
    ledger = GenerationLedger(a.out/'preflight_ledger.json')
    module = SimpleNamespace(compose_full_cfg=composer, main_async=forbidden_main)
    args = SimpleNamespace(config='scene_designer', results_dir=a.out,
        skip_furniture=False, skip_wall=False, skip_ceiling=False)
    declarations = {k:dict(verified=True, evidence='Deliberately optimistic fixture declaration')
        for k in ('full_stage_assets','tool_side_no_paid_api','common_evaluation','common_action_scope')}
    try:
        await run_native_trial(module, args, runtime_admission=declarations, ledger=ledger)
    except ValueError as exc:
        error = str(exc)
    else:
        raise AssertionError('Unqualified configuration admitted')
    assert calls == ['compose'] and 'Native image service' in error

    # Switching to retrieval and disabling texture fallback is insufficient:
    # the real constructor is still eager. This is a config counterexample,
    # not a retrieval run, asset substitution, or proposed scene condition.
    retrieval = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    with open_dict(retrieval):
        for stage in ('furniture','wall','ceiling','manipuland'):
            am = retrieval[stage+'_agent'].asset_manager
            am.general_asset_source='hssd'
            am.router.strategies.thin_covering.generator.enabled=False
    try:
        reject_known_external_images(retrieval, active_stages=stages)
    except ValueError as exc:
        retrieval_error=str(exc)
    else:
        raise AssertionError('Eager constructor incorrectly bypassed')
    after = {str(p): sha(p) for p in old}
    assert before == after
    save(a.out/'source_integrity.json', dict(before=before, after=after, unchanged=True))
    summary=dict(configuration='Scene-Designer overlay + pinned SceneSmith defaults',
        pinned_source_files_verified=len(manifest['files']),
        actual_hydra_composition=True,
        actual_service_startups=0, actual_model_calls=0, paid_api_calls=0,
        actual_scenes_generated=0, training_updates=0,
        startup_tripwire_reached=False, production_preflight_rejection=error,
        retrieval_counterexample_rejection=retrieval_error,
        controller_stages=stages, full_runtime_ready=False,
        task_with_required_manipulands_qualified=False,
        seconds=time.monotonic()-tick,
        scope='Configuration execution and production preflight, not native scene generation or provider inference.')
    save(a.out/'summary.json', summary)
    (a.out/'stage_report.html').write_text('<!doctype html><meta charset="utf-8"><title>原生服务与阶段核查</title>'
        '<h1>整场生成尚未具备执行资格：实际配置核查</h1>'
        '<p>Hydra 已真实展开默认配置与 Scene-Designer 覆盖。四类资产代理均配置 generated/SAM3D/OpenAI 图像；'
        '三个启用图像材质回退。已有本地 VLM 绑定不覆盖图像生成。即使切换 HSSD，AssetManager 仍先构造图像客户端。</p>'
        '<p>当前驱动遍历 floor_plan、furniture、wall、ceiling；配置中的 manipuland 没有驱动入口。'
        '家具生成工具明确只生成落地家具。不能把四阶段结束等同于包含指定桌面物体的完整生成。</p>'
        '<p>已把实际配置预检接入生产包装器，在服务启动之前拒绝这些路径。没有创建模型、调用 API 或生成场景。'
        '这修正实验资格判断，不构成新算法，也不证明更改配置就能完成任务。</p>'
        '<h2>对算法比较的影响</h2><p>决定条件化、等材料辅助监督直出动作、普通搜索必须共用真实完整能力；'
        '不能让某组缺少小物件生成或让其他组使用未计费的图像模型。'
        '下一步应复用上游的小物件阶段与明确的本地资产路径，先验证完整输入的必需实例与关系，不能删除要求换取运行。</p>'
        '<h2>实际记录</h2><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','integration','native-source','asset-manager','out'):
        p.add_argument('--'+name, type=Path, required=True)
    asyncio.run(main(p.parse_args()))
