"""Versioned prospective corpus counts; preserve historical 22-example study."""
from train_edit_learning_study import EditExamples


class CandidateEditExamples(EditExamples):
    def __init__(self, records, *, expected_examples, expected_families):
        if not records or len(records) != expected_examples:
            raise ValueError('Frozen candidate example count mismatch')
        if len({r['id'] for r in records}) != len(records) or len({r['arm'] for r in records}) != 1:
            raise ValueError('Duplicated examples or mixed training arms')
        self.records = records
        self.families = sorted({r['family_id'] for r in records})
        if len(self.families) != expected_families:
            raise ValueError('Frozen candidate family count mismatch')
        for row in records:
            if row['target_kind'] not in ('tool_action', 'terminal_control'):
                raise ValueError('Unknown supervision type')
            if row['action_tokens'] != row['tool_tokens'] + row['terminal_tokens']:
                raise ValueError('Control-token accounting mismatch')
            if row['target_kind'] == 'terminal_control' and (row['decision_tokens'] or row['tool_tokens']):
                raise ValueError('Terminal control cannot count as a professional decision or tool action')
        self.counts = {f: {c:sum(r[c+'_tokens'] for r in records if r['family_id']==f)
                          for c in ('action','decision')} for f in self.families}
