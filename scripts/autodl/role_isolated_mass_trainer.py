"""Serial SFT integration for the existing OriginalMassTrainer and role scope.

No CLI, loader, train call, new objective, teacher distillation or automatic
optimizer update. Model must already be the validated two-adapter PEFT object.
Real8B Trainer execution/peak memory remain unqualified by CPU dispatch checks.
"""
from phase_trainer import OriginalMassTrainer
from shared_learning_roles import require_student_only_optimizer


class RoleIsolatedMassTrainer(OriginalMassTrainer):
    def __init__(self, *, learning_roles, **kwargs):
        if kwargs.get("model") is not learning_roles.model:
            raise ValueError("Explicit already-created shared model must be supplied")
        if kwargs.get("model_init") is not None:
            raise ValueError("Trainer model_init could replace the shared base")
        self.learning_roles = learning_roles
        self._role_train_open = False
        super().__init__(**kwargs)
        self._check_shared_model()

    def _check_shared_model(self):
        if self.model is not self.learning_roles.model:
            raise ValueError("Trainer and native roles must retain the same model object")
        if getattr(self.args, "load_best_model_at_end", False):
            raise ValueError("Automatic checkpoint replacement is outside this shared-base contract")
        if any(getattr(self.args, k, False) for k in
               ("torch_compile", "jit_mode_eval", "auto_find_batch_size")):
            raise ValueError("Compilation/JIT or automatic retry paths are not qualified")
        if getattr(self, "model_init", None) is not None:
            raise ValueError("Model replacement is outside this shared-base contract")
        # Trainer clips model.parameters(), not only optimizer-owned parameters.
        # Stale non-student gradients could therefore change the student's clip
        # factor even though the optimizer never owns base/teacher parameters.
        if any(getattr(p, "grad", None) is not None for n, p in self.model.named_parameters()
               if n not in self.learning_roles.student_names):
            raise ValueError("Stale base/teacher gradients must be resolved before serial SFT")

    def create_optimizer(self):
        if not self._role_train_open or self.learning_roles.active != "student_train":
            raise RuntimeError("Create/inspect optimizer inside the serial student training scope")
        self._check_shared_model()
        optimizer = super().create_optimizer()
        require_student_only_optimizer(optimizer, self.learning_roles)
        return optimizer

    def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None, **kwargs):
        if (resume_from_checkpoint is not None and resume_from_checkpoint is not False) or trial is not None or kwargs:
            raise ValueError("Checkpoint/HP/model replacement paths need a separately qualified protocol")
        self._check_shared_model()
        if self._role_train_open or self.learning_roles.active is not None:
            raise RuntimeError("Nested training/teacher/tool role is not supported")
        # The entire inherited loop is locked: includes optimizer.step, which
        # Trainer executes AFTER training_step, plus checkpoint recomputation.
        with self.learning_roles.role("student_train"):
            self._role_train_open = True
            try:
                if self.optimizer is not None:
                    require_student_only_optimizer(self.optimizer, self.learning_roles)
                return super().train(resume_from_checkpoint=None, trial=None,
                                     ignore_keys_for_eval=ignore_keys_for_eval)
            finally:
                self._role_train_open = False

    def training_step(self, model, inputs, num_items_in_batch=None):
        if (not self._role_train_open or self.learning_roles.active != "student_train"
                or model is not self.learning_roles.model):
            raise RuntimeError("Forward/backward must stay on the shared model inside the serial training role")
        return super().training_step(model, inputs, num_items_in_batch)
