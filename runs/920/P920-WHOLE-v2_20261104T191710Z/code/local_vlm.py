"""Local-only shared Qwen engine and SceneSmith VLMService-compatible adapter.

This is an engineering adapter, not a new policy or a validated replacement for
asset orientation/material estimation. Loading the processor does not load the
model. No network client, remote-image loader or fallback is implemented.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import os
import sys
import threading
import time
from pathlib import Path

# This one pinned small dependency is isolated here; the previously verified
# torch/CUDA environment is not rewritten by installing the adapter.
VENDOR = Path(__file__).resolve().parent / 'deps'
if VENDOR.is_dir():
    sys.path.insert(0, str(VENDOR))


class LocalQwenEngine:
    def __init__(self, checkpoint, log_dir, *, max_pixels=65536,
                 max_input_tokens=16384, max_new_tokens=2048):
        self.checkpoint = Path(checkpoint).resolve(strict=True)
        self.log_dir = Path(log_dir).resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_pixels = int(max_pixels)
        self.max_input_tokens = int(max_input_tokens)
        self.max_new_tokens = int(max_new_tokens)
        if min(self.max_pixels, self.max_input_tokens, self.max_new_tokens) <= 0:
            raise ValueError('Budgets must be positive')
        self._processor = self._model = None
        self._lock = threading.Lock()

    @property
    def processor(self):
        if self._processor is None:
            # Explicit local_files_only remains effective even if a parent has
            # enabled network access for other parts of the tool environment.
            from transformers import AutoProcessor
            self._processor = AutoProcessor.from_pretrained(
                str(self.checkpoint), local_files_only=True, trust_remote_code=False)
            ip = self._processor.image_processor
            ip.max_pixels = self.max_pixels
            if isinstance(getattr(ip, 'size', None), dict):
                ip.size = dict(ip.size, longest_edge=self.max_pixels)
                ip.size['shortest_edge'] = min(ip.size.get('shortest_edge', 4096), self.max_pixels)
            if getattr(ip, 'min_pixels', None) is not None:
                ip.min_pixels = min(ip.min_pixels, self.max_pixels)
        return self._processor

    @staticmethod
    def normalize_messages(messages):
        """Preserve message/image order; decode only local files or inline PNG/JPEG."""
        from PIL import Image
        normalized, images, evidence = [], [], []
        for msg in copy.deepcopy(messages):
            if msg.get('role') not in ('system', 'user', 'assistant'):
                raise ValueError('Unsupported role; tool results must use native memory messages')
            content = msg.get('content')
            if isinstance(content, str):
                normalized.append(msg)
                continue
            if not isinstance(content, list):
                raise ValueError('Expected string or ordered content list')
            new_content = []
            for item in content:
                if item.get('type') == 'text':
                    if not isinstance(item.get('text'), str):
                        raise ValueError('Text content must be a string')
                    new_content.append(item)
                    continue
                if item.get('type') == 'image_url':
                    raw = item.get('image_url', {}).get('url')
                elif item.get('type') == 'image':
                    raw = item.get('image', item.get('path'))
                else:
                    raise ValueError('Only text and still images are supported')
                raw = str(raw) if isinstance(raw, Path) else raw
                if not isinstance(raw, str):
                    raise ValueError('Expected a local image path or inline image')
                if raw.startswith(('data:image/png;base64,', 'data:image/jpeg;base64,')):
                    data = base64.b64decode(raw.split(',', 1)[1], validate=True)
                    origin = 'inline_data_uri'
                else:
                    if '://' in raw or raw.startswith('data:'):
                        raise ValueError('Remote images and unsupported data URIs are not allowed')
                    path = Path(raw).resolve(strict=True)
                    if not path.is_file():
                        raise ValueError('Not a local image file')
                    data, origin = path.read_bytes(), str(path)
                if len(data) > 16 * 1024**2:
                    raise ValueError('Image exceeds the bounded 16 MiB input size')
                with Image.open(io.BytesIO(data)) as src:
                    if src.format not in ('PNG', 'JPEG') or src.width * src.height > 16_777_216:
                        raise ValueError('Unsupported image format or resolution')
                    image = src.convert('RGB')
                index = len(images)
                images.append(image)
                # The processor receives decoded PIL images separately. This
                # item only emits the image placeholder in the chat template.
                new_content.append({'type': 'image', 'image': f'local_image_{index}'})
                evidence.append({'index': index, 'origin': origin,
                    'sha256': hashlib.sha256(data).hexdigest(), 'size': list(image.size)})
            msg['content'] = new_content
            normalized.append(msg)
        return normalized, images, evidence

    def encode(self, messages):
        normalized, images, evidence = self.normalize_messages(messages)
        text = self.processor.apply_chat_template(
            normalized, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images or None, return_tensors='pt')
        length = int(inputs['input_ids'].shape[1])
        if length > self.max_input_tokens:
            raise ValueError(f'Input has {length} tokens, exceeds {self.max_input_tokens}; not truncated')
        receipt = {'images': evidence, 'input_tokens': length,
            'input_shapes': {k: list(v.shape) for k, v in inputs.items()},
            'chat_template_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'max_pixels': self.max_pixels, 'max_input_tokens': self.max_input_tokens}
        return inputs, receipt

    def _load_model(self):
        if self._model is not None:
            return
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('No visible CUDA device; processor checks remain available, no CPU weight load attempted')
        from transformers import Qwen3VLForConditionalGeneration
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(self.checkpoint), local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.float16, device_map={'': 'cuda:0'},
            low_cpu_mem_usage=True, attn_implementation='eager')
        self._model.eval()
        self._model.config.use_cache = True

    def complete(self, messages, *, purpose, max_new_tokens=None, metadata=None):
        """One serialized local forward, no hidden retries or API fallback."""
        budget = self.max_new_tokens if max_new_tokens is None else int(max_new_tokens)
        if not 0 < budget <= self.max_new_tokens:
            raise ValueError('Requested generation exceeds the configured token budget')
        with self._lock:
            call_dir = self.log_dir / f'{time.time_ns()}_{os.getpid()}'
            call_dir.mkdir()
            # Exact input, including inline images, is recoverable. Do not log
            # credentials; this interface has no credential parameters.
            (call_dir / 'request.json').write_text(json.dumps(messages, ensure_ascii=False, indent=2, default=str))
            t0 = time.monotonic()
            result = {'purpose': purpose, 'checkpoint': str(self.checkpoint),
                'max_new_tokens': budget, 'metadata': metadata or {},
                'generation_started': False, 'paid_api_calls': 0,
                'fallbacks': 0, 'device': 'cuda:0', 'dtype': 'float16',
                'attention': 'eager', 'sampling': False, 'quality_validated': False}
            try:
                inputs, encoded = self.encode(messages)
                result.update(encoded)
                self._load_model()
                import torch
                inputs = inputs.to(self._model.device)
                torch.cuda.reset_peak_memory_stats()
                result['generation_started'] = True
                with torch.inference_mode():
                    output = self._model.generate(**inputs, max_new_tokens=budget,
                        do_sample=False, pad_token_id=self.processor.tokenizer.pad_token_id
                        if self.processor.tokenizer.pad_token_id is not None else self.processor.tokenizer.eos_token_id)
                new = output[0, inputs['input_ids'].shape[1]:]
                raw = self.processor.tokenizer.decode(new, skip_special_tokens=False)
                clean = self.processor.tokenizer.decode(new, skip_special_tokens=True)
                (call_dir / 'raw_output.txt').write_text(raw)
                (call_dir / 'text_output.txt').write_text(clean)
                # Length-limited output is retained as evidence, never used as a
                # complete program or an inferred physical property record.
                eos = self._model.generation_config.eos_token_id
                eos = [eos] if isinstance(eos, int) else (eos or [])
                ended = bool(len(new) and int(new[-1]) in eos)
                result.update(output_tokens=len(new), ended_with_eos=ended,
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
                if len(new) >= budget and not ended:
                    raise RuntimeError('Output reached token cap without EOS; retained but not executed')
                result['status'] = 'completed'
                return {'raw': raw, 'text': clean, 'receipt_dir': str(call_dir)}
            except Exception as exc:
                result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                raise
            finally:
                result['elapsed_seconds'] = time.monotonic() - t0
                (call_dir / 'receipt.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))


class LocalVLMService:
    """Inject into AssetManager(..., vlm_service=...). Same local engine as policy.

    Native reasoning/verbosity/detail controls do not have equivalent semantics
    here: their requested values are logged; input resolution uses max_pixels.
    Asset material/mass predictions are model estimates, not physical ground truth.
    """
    def __init__(self, engine):
        self.engine = engine

    def create_completion(self, model, messages, reasoning_effort, verbosity,
                          response_format=None, vision_detail='auto'):
        if response_format not in (None, {'type': 'json_object'}):
            raise ValueError('Only plain text or strict JSON object responses supported')
        requested = copy.deepcopy(messages)
        if response_format:
            requested.append({'role': 'user', 'content':
                'Return only one valid JSON object. Do not add markdown or commentary.'})
        result = self.engine.complete(requested, purpose='native_asset_analysis', metadata={
            'requested_model_label': model, 'actual_backend': 'local_shared_checkpoint',
            'requested_reasoning_effort': reasoning_effort, 'requested_verbosity': verbosity,
            'requested_vision_detail': vision_detail, 'response_format': response_format,
            'native_reasoning_verbosity_detail_semantics_preserved': False})
        text = result['text'].strip()
        if response_format:
            try:
                def reject_constant(value):
                    raise ValueError(f'Non-finite JSON constant: {value}')
                obj = json.loads(text, parse_constant=reject_constant)
                if not isinstance(obj, dict):
                    raise ValueError('Expected JSON object')
            except (ValueError, TypeError) as exc:
                (Path(result['receipt_dir']) / 'consumer_validation.json').write_text(
                    json.dumps({'passed': False, 'error': str(exc)}))
                raise ValueError('Local asset analysis did not return valid JSON; no repair or retry') from exc
        return text


class LocalPolicy:
    """Native generate(obs) shape; caller supplies the exact pinned parser."""
    def __init__(self, engine, parse_assistant_text):
        self.engine = engine
        self.parse_assistant_text = parse_assistant_text

    def generate(self, obs):
        result = self.engine.complete(obs['messages'], purpose='native_scene_policy')
        think, tool_calls = self.parse_assistant_text(result['raw'])
        return {'raw': result['raw'], 'think': think, 'tool_calls': tool_calls,
                'local_receipt_dir': result['receipt_dir']}
