from __future__ import annotations

import uuid

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pagedserve.attention.qwen_paged_attention import qwen_paged_attention
from pagedserve.attention.batched_paged_attention import batched_qwen_paged_attention
from pagedserve.engine.paged_kv_cache import PagedKVCache
from pagedserve.engine.request import Request, RequestStatus
from pagedserve.engine.scheduler import Scheduler


class PagedServeEngine:
    """
    Single-request inference engine using PagedServe's KV cache
    and custom Qwen attention implementation.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        num_blocks: int = 2048,
        block_size: int = 16,
        attention_backend: str = "pytorch",
    ):
        if attention_backend not in {"pytorch", "triton"}:
            raise ValueError(
                "attention_backend must be either 'pytorch' or 'triton'"
            )

        if attention_backend == "triton" and not device.startswith("cuda"):
            raise ValueError("The Triton attention backend requires a CUDA device")

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")

        self.device = torch.device(device)
        self.dtype = dtype
        self.block_size = block_size
        self.attention_backend = attention_backend

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            attn_implementation="eager",
        ).to(self.device).eval()

        config = self.model.config

        self.cache = PagedKVCache(
            num_blocks=num_blocks,
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            block_size=block_size,
            dtype=dtype,
            device=self.device,
        )

        self._request_counter = 0

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 32,
    ) -> str:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative.")

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        )

        input_ids = encoded["input_ids"].to(self.device)

        if input_ids.shape[1] == 0:
            raise ValueError("Prompt must contain at least one token.")

        if max_new_tokens == 0:
            return self.tokenizer.decode(
                input_ids[0],
                skip_special_tokens=True,
            )

        request_id = f"request-{uuid.uuid4().hex}"
        prompt_length = input_ids.shape[1]
        max_positions = getattr(
            self.model.config, "max_position_embeddings", 32768
        )

        if prompt_length + max_new_tokens > max_positions:
            raise ValueError(
                "Prompt plus requested generation exceeds the model's "
                f"maximum position count ({max_positions})."
            )

        total_capacity = prompt_length + max_new_tokens
        self.cache.allocate(request_id, total_capacity)

        generated_ids = input_ids

        try:
            # Prefill: execute each transformer layer using PagedServe.
            outputs = self._forward_paged(
                input_ids=input_ids,
                request_id=request_id,
                position_start=0,
            )

            next_token = outputs[:, -1, :].argmax(
                dim=-1, keepdim=True
            )

            generated_ids = torch.cat(
                [generated_ids, next_token],
                dim=1,
            )

            # Decode: each iteration consumes the previously generated
            # token and reuses the cached K/V history.
            for step in range(1, max_new_tokens):
                if (
                    self.model.generation_config.eos_token_id is not None
                    and next_token.item()
                    == self.model.generation_config.eos_token_id
                ):
                    break

                outputs = self._forward_paged(
                    input_ids=next_token,
                    request_id=request_id,
                    position_start=prompt_length + step - 1,
                )

                next_token = outputs[:, -1, :].argmax(
                    dim=-1, keepdim=True
                )

                generated_ids = torch.cat(
                    [generated_ids, next_token],
                    dim=1,
                )

            return self.tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True,
            )

        finally:
            self.cache.free(request_id)

    @torch.inference_mode()
    def generate_many(
        self,
        prompts: list[str],
        max_new_tokens: int = 32,
    ) -> list[str]:
        """
        Generate responses for multiple prompts using independent
        paged KV-cache allocations and round-robin decode scheduling.

        This version interleaves requests but does not yet combine
        their model forward passes into a single GPU batch.
        """
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative.")

        if not prompts:
            return []

        if any(not isinstance(prompt, str) for prompt in prompts):
            raise TypeError("Every prompt must be a string.")

        if max_new_tokens == 0:
            return [
                self.tokenizer.decode(
                    self.tokenizer(
                        prompt,
                        add_special_tokens=True,
                    )["input_ids"],
                    skip_special_tokens=True,
                )
                for prompt in prompts
            ]

        requests = []
        allocated_ids = []

        eos_token_id = self.model.generation_config.eos_token_id
        if isinstance(eos_token_id, list):
            eos_token_ids = set(eos_token_id)
        elif eos_token_id is None:
            eos_token_ids = set()
        else:
            eos_token_ids = {eos_token_id}

        max_positions = getattr(
            self.model.config,
            "max_position_embeddings",
            32768,
        )

        try:
            # Tokenize and reserve each request's KV-cache capacity.
            # Allocation failure is reported instead of silently
            # dropping or partially processing requests.
            for index, prompt in enumerate(prompts):
                encoded = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    add_special_tokens=True,
                )

                input_ids = encoded["input_ids"].to(self.device)

                if input_ids.shape[1] == 0:
                    raise ValueError(
                        f"Prompt at index {index} contains no tokens."
                    )

                prompt_length = input_ids.shape[1]
                capacity = prompt_length + max_new_tokens

                if capacity > max_positions:
                    raise ValueError(
                        f"Prompt at index {index} plus requested output "
                        f"exceeds the model's maximum position count "
                        f"({max_positions})."
                    )

                request_id = f"batch-{uuid.uuid4().hex}"

                self.cache.allocate(request_id, capacity)
                allocated_ids.append(request_id)

                requests.append({
                    "request_id": request_id,
                    "input_ids": input_ids,
                    "prompt_length": prompt_length,
                    "generated": [],
                    "next_token": None,
                    "done": False,
                })

            # Prefill each request independently.
            for request in requests:
                logits = self._forward_paged(
                    input_ids=request["input_ids"],
                    request_id=request["request_id"],
                    position_start=0,
                )

                next_token = logits[:, -1, :].argmax(
                    dim=-1,
                    keepdim=True,
                )

                token_id = next_token.item()
                request["generated"].append(token_id)

                if token_id in eos_token_ids:
                    request["done"] = True
                else:
                    request["next_token"] = next_token

            # Batched decode: all active requests share one model forward
            # per iteration, while retaining independent KV-cache allocations.
            while any(not request["done"] for request in requests):
                active = [
                    request
                    for request in requests
                    if not request["done"]
                ]

                batch_input_ids = torch.cat(
                    [request["next_token"] for request in active],
                    dim=0,
                )

                request_ids = [
                    request["request_id"] for request in active
                ]

                positions = [
                    request["prompt_length"]
                    + len(request["generated"]) - 1
                    for request in active
                ]

                logits = self._forward_paged_batch(
                    input_ids=batch_input_ids,
                    request_ids=request_ids,
                    positions=positions,
                )

                next_tokens = logits[:, -1, :].argmax(
                    dim=-1,
                    keepdim=True,
                )

                for row, request in enumerate(active):
                    next_token = next_tokens[row:row + 1]
                    token_id = next_token.item()
                    request["generated"].append(token_id)

                    if (
                        token_id in eos_token_ids
                        or len(request["generated"]) >= max_new_tokens
                    ):
                        request["done"] = True
                        request["next_token"] = None
                    else:
                        request["next_token"] = next_token

            # Preserve the original prompt-plus-completion behavior
            # of generate().
            results = []

            for request in requests:
                full_ids = torch.cat(
                    [
                        request["input_ids"],
                        torch.tensor(
                            [request["generated"]],
                            dtype=request["input_ids"].dtype,
                            device=self.device,
                        ),
                    ],
                    dim=1,
                )

                results.append(
                    self.tokenizer.decode(
                        full_ids[0],
                        skip_special_tokens=True,
                    )
                )

            return results

        finally:
            # Release every allocation, including when prefill or
            # decoding raises an exception partway through.
            for request_id in allocated_ids:
                if self.cache.block_manager.has_request(request_id):
                    self.cache.free(request_id)



    @torch.inference_mode()
    def generate_scheduled(
        self,
        prompts: list[str],
        max_new_tokens: int = 32,
        max_running_requests: int = 8,
    ) -> list[str]:
        """Generate prompts through the FIFO scheduler and batched decoder."""
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative")
        if max_running_requests <= 0:
            raise ValueError("max_running_requests must be positive")
        if not prompts:
            return []
        if any(not isinstance(prompt, str) for prompt in prompts):
            raise TypeError("Every prompt must be a string")

        eos = self.model.generation_config.eos_token_id
        eos_ids = (
            set() if eos is None
            else set(eos) if isinstance(eos, (list, tuple))
            else {eos}
        )

        if max_new_tokens == 0:
            return [
                self.tokenizer.decode(
                    self.tokenizer(p, add_special_tokens=True)["input_ids"],
                    skip_special_tokens=True,
                )
                for p in prompts
            ]

        max_positions = getattr(
            self.model.config, "max_position_embeddings", 32768
        )
        scheduler = Scheduler(
            max_running_requests=max_running_requests,
            kv_cache=self.cache,
        )
        requests = []
        input_ids_by_id = {}
        next_token_by_id = {}
        request_order = []

        # Validate and submit all requests before allocating cache blocks.
        for index, prompt in enumerate(prompts):
            encoded = self.tokenizer(
                prompt, return_tensors="pt", add_special_tokens=True
            )
            input_ids = encoded["input_ids"].to(self.device)
            if input_ids.shape[1] == 0:
                raise ValueError(f"Prompt at index {index} contains no tokens")

            if input_ids.shape[1] + max_new_tokens > max_positions:
                raise ValueError(
                    f"Prompt at index {index} plus requested output exceeds "
                    f"the model's maximum position count ({max_positions})"
                )

            request_id = f"scheduled-{uuid.uuid4().hex}"
            request = Request(
                request_id=request_id,
                prompt_token_ids=input_ids[0].tolist(),
                max_new_tokens=max_new_tokens,
            )
            scheduler.submit(request)
            requests.append(request)
            request_order.append(request_id)
            input_ids_by_id[request_id] = input_ids

        try:
            while scheduler.waiting_count or scheduler.running_count:
                admitted = scheduler.schedule()

                # Prefill newly admitted requests.
                for request in admitted:
                    rid = request.request_id
                    logits = self._forward_paged(
                        input_ids=input_ids_by_id[rid],
                        request_id=rid,
                        position_start=0,
                    )
                    token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    token_id = token.item()
                    request.add_generated_token(token_id)

                    if token_id in eos_ids or request.remaining_tokens == 0:
                        scheduler.complete(rid)
                    else:
                        next_token_by_id[rid] = token

                # Decode one token for every active request in one forward pass.
                active = [
                    r for r in scheduler.running_requests
                    if r.request_id in next_token_by_id
                ]

                if not active:
                    if scheduler.waiting_count:
                        raise RuntimeError(
                            "Waiting request cannot fit in the available "
                            "KV-cache capacity"
                        )
                    continue

                active_ids = [r.request_id for r in active]
                batch_ids = torch.cat(
                    [next_token_by_id[rid] for rid in active_ids], dim=0
                )
                positions = [
                    r.num_prompt_tokens + r.num_generated_tokens - 1
                    for r in active
                ]

                logits = self._forward_paged_batch(
                    input_ids=batch_ids,
                    request_ids=active_ids,
                    positions=positions,
                )
                tokens = logits[:, -1, :].argmax(dim=-1, keepdim=True)

                for row, request in enumerate(active):
                    rid = request.request_id
                    token = tokens[row:row + 1]
                    token_id = token.item()
                    request.add_generated_token(token_id)

                    if token_id in eos_ids or request.remaining_tokens == 0:
                        next_token_by_id.pop(rid, None)
                        scheduler.complete(rid)
                    else:
                        next_token_by_id[rid] = token

            results = []
            for rid in request_order:
                request = scheduler.requests[rid]
                full_ids = (
                    input_ids_by_id[rid][0].tolist()
                    + request.generated_token_ids
                )
                results.append(self.tokenizer.decode(
                    full_ids, skip_special_tokens=True
                ))
            return results

        except Exception:
            for request in requests:
                if request.is_terminal:
                    continue
                try:
                    if request.status == RequestStatus.RUNNING:
                        scheduler.fail(
                            request.request_id,
                            "Generation aborted after an inference error",
                        )
                    else:
                        scheduler.cancel(
                            request.request_id,
                            "Generation aborted after an inference error",
                        )
                except Exception:
                    pass
            raise


    @torch.inference_mode()
    def _forward_paged_batch(
        self,
        input_ids: torch.Tensor,
        request_ids: list[str],
        positions: list[int],
    ) -> torch.Tensor:
        """
        Execute a single-token forward pass for multiple independent requests.

        Args:
            input_ids: Token IDs with shape [batch_size, 1].
            request_ids: KV-cache request ID for each batch row.
            positions: Logical position of each token.

        Returns:
            Vocabulary logits with shape [batch_size, 1, vocab_size].
        """
        if input_ids.ndim != 2 or input_ids.shape[1] != 1:
            raise ValueError("input_ids must have shape [batch_size, 1]")

        batch_size = input_ids.shape[0]

        if batch_size == 0:
            raise ValueError("Batch must not be empty")

        if len(request_ids) != batch_size or len(positions) != batch_size:
            raise ValueError(
                "request_ids and positions must match the batch size"
            )

        if len(set(request_ids)) != batch_size:
            raise ValueError("Each batch row must have a unique request ID")

        if input_ids.device != self.device:
            raise ValueError("input_ids must be on the engine device")

        model = self.model.model

        position_ids = torch.tensor(
            positions,
            dtype=torch.long,
            device=self.device,
        ).view(batch_size, 1)

        hidden_states = model.embed_tokens(input_ids)

        position_embeddings = model.rotary_emb(
            hidden_states,
            position_ids,
        )

        for layer in model.layers:
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            attention_output = batched_qwen_paged_attention(
                attention=layer.self_attn,
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                cache=self.cache,
                request_ids=request_ids,
                positions=positions,
                attention_backend=self.attention_backend,
            )

            hidden_states = residual + attention_output

            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

        hidden_states = model.norm(hidden_states)
        return self.model.lm_head(hidden_states)

    def _forward_paged(
        self,
        input_ids: torch.Tensor,
        request_id: str,
        position_start: int,
    ) -> torch.Tensor:
        """
        Execute the Qwen transformer layers using PagedServe attention.

        Returns final vocabulary logits with shape [1, sequence, vocab].
        """
        model = self.model.model
        batch_size, sequence_length = input_ids.shape

        position_ids = torch.arange(
            position_start,
            position_start + sequence_length,
            device=self.device,
        ).unsqueeze(0)

        hidden_states = model.embed_tokens(input_ids)

        position_embeddings = model.rotary_emb(
            hidden_states,
            position_ids,
        )

        for layer in model.layers:
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            attention_output = qwen_paged_attention(
                attention=layer.self_attn,
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                cache=self.cache,
                request_id=request_id,
                query_start_position=position_start,
            )

            hidden_states = residual + attention_output

            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

        hidden_states = model.norm(hidden_states)

        return self.model.lm_head(hidden_states)
