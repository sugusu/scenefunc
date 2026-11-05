"""Execute the two frozen candidates once, stopping on the first failed stage."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args()
    repo=Path(__file__).resolve().parents[2];p=repo/'runs/920'/a.run;t=repo.parent/'runtime'/a.run
    py='/data/wyh/conda/envs/c15_genrec/bin/python'
    geo=str(repo.parent/'envs/geometry311/bin/python');native=str(repo.parent/'envs/scene_designer311_native/bin/python')
    choice=json.loads((p/'CHOICE_PLAN.json').read_text());assert choice['budget']['native_candidates']==2
    def stage(name,seconds,command):
        subprocess.run([py,'scripts/920/run_stage.py','--receipt-dir',str(t/name),'--timeout',str(seconds),'--',*command],cwd=repo,check=True)
    stage('build_stage',120,[native,'scripts/920/build_reference_room.py','--plan',str(p/'REFERENCE_BUILD_PLAN.json'),
        '--scenesmith',str(repo.parent/'resources/full_runtime_20261104/scenesmith'),'--out',str(t/'reference')])
    stage('configure_stage',60,[geo,'scripts/920/configure_family_choice.py','--run',a.run])
    for label in choice['candidates']:
        stage(label+'/preflight_stage',60,[geo,'scripts/920/construct_and_switch_room.py','--plan',str(p/(label+'_PLAN.json')),
            '--phase','preflight','--out',str(t/label/'preflight')])
        check=json.loads((t/label/'preflight/RESULT.json').read_text())
        assert check['all_declared_targets_reachable'], 'Retain infeasible candidate; stop this batch before execution.'
    for label in choice['candidates']:
        stage(label+'/native_stage',choice['budget']['native_seconds_per_candidate'],
            [native,'scripts/920/construct_and_switch_room.py','--plan',str(p/(label+'_PLAN.json')),
             '--phase','execute','--out',str(t/label/'native')])


if __name__=='__main__':main()
