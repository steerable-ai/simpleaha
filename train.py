import copy
import gc
import json
import os
import types
from datetime import datetime

import datasets
import lmfunctions as lmf
import math_verify as mv
import peft
import pydantic
import torch
import transformers


class Config:
    def __init__(self):
        base_model = "qwen/Qwen2.5-1.5B"
        self.model = {
            "name": base_model,
            "lora": {"r": 160, "lora_alpha": 160},
        }
        self.training = {
            "seed": 42,
            "dataset": {
                "name": "gsm8k",  # Default training dataset
                "dataset_name": "openai/gsm8k",
                "dataset_config": "main",  # Config name for the dataset
                "dataset_split": "train",
                "question_key": "question",
                "answer_key": "answer",
                "answer_format": "####",
            },
            "total_steps": 101,
            "prompts_per_step": 32,
            "generations_per_prompt": 16,
            "rollouts_temperature": 1.0,
            "lr_init": 5e-4,
            "lr_final": 5e-4,
            "standardize_rewards": False,
            "kl_reg": 0.01,
            "grad_clip": 1e3,
            "amp": True,
            "bf16": True,
            "scale_interval": 100,
        }
        self.evaluation = {
            "datasets": {
                "gsm8k": {
                    "enabled": True,
                    "size": 1,
                    "dataset_name": "openai/gsm8k",
                    "dataset_config": "main",  # Config name for the dataset
                    "dataset_split": "test",
                    "question_key": "question",
                    "answer_key": "answer",
                    "answer_format": "####",  # Special format for GSM8K (extract answer after ####)
                },
                "math500": {
                    "enabled": True,
                    "size": 1,
                    "dataset_name": "HuggingFaceH4/MATH-500",
                    "dataset_config": None,  # No config needed
                    "dataset_split": "test",
                    "question_key": "problem",
                    "answer_key": "answer",
                },
                "aime2024": {
                    "enabled": True,
                    "size": 1,
                    "dataset_name": "HuggingFaceH4/aime_2024",
                    "dataset_config": None,  # No config needed
                    "dataset_split": "train",  # AIME 2024 uses "train" as the main split
                    "question_key": "problem",
                    "answer_key": "answer",
                },
            },
            "evaluation_temperature": 0.0,
            "eval_every": 10,
        }
        self.vllm = {
            "model": base_model,
            "chat": False,
            "seed": 42,
            "gpu_memory_utilization": 0.4,
            "enable_prefix_caching": True,
            "sampling_params": {
                "max_tokens": 2048,
                "truncate_prompt_tokens": 512,
            },
        }
        self.out = {"model": "saved_models", "log": "logs"}


class GRPOAgent:
    def __init__(self, model, train_config):
        self.policy = model
        self.cfg = train_config
        self.G = self.cfg["generations_per_prompt"]  # Group size
        self.standardize_rewards = self.cfg["standardize_rewards"]  # Standardize rewards
        self.kl_reg = self.cfg["kl_reg"]  # Regularization coefficient
        self.grad_clip = self.cfg["grad_clip"]  # Gradient clipping threshold
        self.params = [p for p in self.policy.parameters() if p.requires_grad]
        self.dtype = torch.bfloat16 if self.cfg["bf16"] else torch.float32
        self.autocast = torch.amp.autocast(
            device_type=self.policy.device.type,
            dtype=self.dtype,
            enabled=self.cfg["amp"],
        )
        self.scaler = torch.amp.GradScaler(
            device=self.policy.device,
            enabled=self.cfg["amp"],
            growth_interval=self.cfg["scale_interval"],
        )
        self.opt = torch.optim.AdamW(self.params, lr=self.cfg["lr_init"], fused=True)
        self.sched = torch.optim.lr_scheduler.LinearLR(
            self.opt,
            1.0,
            self.cfg["lr_final"] / self.cfg["lr_init"],
            total_iters=self.cfg["total_steps"],
        )

    def update(self, states, actions, rewards):
        B = len(states)
        metrics_keys = ["obj", "loss", "reg", "len"]
        metrics = {k: 0.0 for k in metrics_keys}
        self.opt.zero_grad()  # Zero gradients
        rewards = torch.tensor(rewards, device=self.policy.device).view(-1, self.G)  # Reshape rewards
        advantages = rewards - rewards.mean(1, keepdim=True)
        if self.standardize_rewards:
            advantages = advantages / (rewards.std(1, keepdim=True) + 1e-6)
        advantages = advantages.flatten()
        for i, (s, a) in enumerate(zip(states, actions)):  # Loop over individual samples to keep low memory footprint
            with self.autocast:  # Automatic mixed precision context
                logp, logp0 = self.policy.logprobs(s, a)  # Compute log probabilities for policy and reference
                loss = -(logp * advantages[i])  # Policy gradient loss
                reg = self.kl_reg * (logp - logp0) ** 2  # Deviation from reference policy (also a valid KL divergence estimator)
                obj = loss + reg  # Loss + Regularization
            self.scaler.scale(obj / B).backward()  # Accumulate gradients scaled by batch size
            for k, v in zip(metrics_keys, [obj.item(), loss.item(), reg.item(), len(a)]):
                metrics[k] += v / B  # Accumulate metrics
        self.scaler.unscale_(self.opt)  # Unscale gradients prior to clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.grad_clip)  # Clip gradients
        self.scaler.step(self.opt)  # Optimizer step (does not actually scale anything, just skips optimization step if the gradient has NaNs or infs)
        self.scaler.update()  # Update the scaler state
        self.sched.step()  # Step the learning rate scheduler
        return metrics | {
            "avg_reward": rewards.mean().item(),
            "grad_norm": grad_norm.item(),
            "batch_size": B,
            "lr": self.opt.param_groups[0]["lr"],
            "scale": self.scaler.get_scale(),
        }


def logprobs(self, x, y):
    """Compute log-probability of a completion conditioned on the prompt using the model with and without LORA"""
    if not y:
        return (torch.zeros(1, device=self.device) for _ in (0, 1))  # Return zero log-probabilities for empty completions, so they don't affect the objective
    p, c = (torch.tensor(t, device=self.device, dtype=torch.long) for t in (x, y))  # Convert inputs to tensors
    tensor, idx = torch.cat([p, c]).unsqueeze(0), slice(len(p) - 1, -1)  # Concatenate inputs and compute slice for completion
    logp = self(input_ids=tensor).logits[0, idx].log_softmax(-1).gather(-1, c.unsqueeze(-1)).sum()  # Compute log-probability of completion
    with torch.no_grad(), self.disable_adapter():  # Disable LORA adapter and gradients for reference computation
        logp0 = self(input_ids=tensor).logits[0, idx].log_softmax(-1).gather(-1, c.unsqueeze(-1)).sum()  # Compute reference log-probability
    return logp, logp0


def main():
    config = Config()

    # Load datasets and build question-answer mapping
    q2a = {}
    eval_datasets = {}

    # Load training dataset
    train_cfg = config.training["dataset"]
    train_dataset = datasets.load_dataset(train_cfg["dataset_name"], train_cfg["dataset_config"], split=train_cfg["dataset_split"])

    # Process training data answers for reward function
    q2a.update(
        dict(
            zip(
                train_dataset[train_cfg["question_key"]],
                [a.split(train_cfg.get("answer_format", ""))[-1].strip() if "answer_format" in train_cfg else a for a in train_dataset[train_cfg["answer_key"]]],
            )
        )
    )

    # Load evaluation datasets from configuration
    for name, cfg in config.evaluation["datasets"].items():
        if not cfg["enabled"]:
            continue

        # Load dataset and apply size limit
        dataset = datasets.load_dataset(cfg["dataset_name"], cfg["dataset_config"], split=cfg["dataset_split"])
        if cfg["size"]:
            dataset = dataset.select(range(cfg["size"]))

        # Extract questions and process answers
        questions = dataset[cfg["question_key"]]
        answers = [a.split(cfg.get("answer_format", ""))[-1].strip() if "answer_format" in cfg else a for a in dataset[cfg["answer_key"]]]

        # Update mappings
        q2a.update(dict(zip(questions, answers)))
        eval_datasets[name] = {"questions": questions}

    # Define the task (reason and solve)
    class Solution(pydantic.BaseModel):
        thinking: str = pydantic.Field(description="Step-by-step thinking process")
        solution: str = pydantic.Field(description="Solution")

    @lmf.lmdef
    def think_and_solve(problem: str) -> Solution:
        """Think step-by-step and solve the problem"""
        ...

    # Define a reward function
    def reward(question, answer):
        try:
            return float(mv.verify(mv.parse(q2a.get(question, None)), mv.parse(str(answer.solution))))
        except:
            return 0.0

    # Setup Model and Tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model["name"])
    model = peft.get_peft_model(
        transformers.AutoModelForCausalLM.from_pretrained(config.model["name"], use_cache=False).to(device),
        peft.LoraConfig(**config.model["lora"]),
    ).train()
    model.print_trainable_parameters()
    model.logprobs = types.MethodType(logprobs, model)  # Monkey patch the model with the logprobs method

    # Setup Agent
    agent = GRPOAgent(model=model, train_config=config.training)

    vllm_backend = lmf.backends.VLLMBackend(**config.vllm)
    lmf.default.backend = vllm_backend
    global states, actions
    states, actions = [], []
    lmf.default.event_manager.handlers["success"] = [
        lambda **v: states.append(tokenizer.encode(v["backend_input"])),
        lambda **v: actions.append(tokenizer.encode(v["completion"])),
    ]

    # Setup Logging
    os.makedirs(config.out["log"], exist_ok=True)
    log_file = os.path.join(config.out["log"], f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
    log = lambda metrics: open(log_file, "a").write(json.dumps({**metrics, "timestamp": datetime.now().isoformat()}) + "\n")
    log({"type": "config", **vars(config), "task": think_and_solve.dumps()})

    # Training Loop (With Optional Evaluation)
    step = 0
    # Create DataLoader for the configured training dataset
    train_dl = torch.utils.data.DataLoader(train_dataset[train_cfg["question_key"]], batch_size=config.training["prompts_per_step"], shuffle=True)
    while step < config.training["total_steps"]:
        # Cycle through the dataset as needed
        for batch in train_dl:

            if step % config.evaluation["eval_every"] == 0:
                # Run evaluation on all enabled datasets
                vllm_backend.sampling_params["temperature"] = config.evaluation["evaluation_temperature"]
                eval_metrics = {"step": step, "type": "eval"}

                for name, dataset in eval_datasets.items():
                    questions = dataset["questions"]
                    model_answers = think_and_solve(questions, batch_call=True)
                    correct = sum(reward(q, a) for q, a in zip(questions, model_answers))
                    eval_metrics.update({f"acc_{name}": correct / len(questions), f"correct_{name}": correct, f"total_{name}": len(questions)})

                log(eval_metrics)
                lmf.utils.panelprint(eval_metrics, title=f"Eval Step:{eval_metrics['step']}")

            # Generate rollouts
            vllm_backend.sampling_params["temperature"] = config.training["rollouts_temperature"]
            questions = [p for p in batch for _ in range(config.training["generations_per_prompt"])]
            states, actions = [], []
            gsm8k_answers = think_and_solve(questions, batch_call=True)
            rewards = [reward(q, a) for q, a in zip(questions, gsm8k_answers)]

            # Policy update
            metrics = agent.update(states, actions, rewards)

            # Sync weights with VLLM backend
            vllm = lmf.default.backend.lm.llm_engine.model_executor.driver_worker.worker.get_model()
            vllm.load_weights(iter(copy.deepcopy(model).merge_and_unload().state_dict().items()))
            gc.collect(), torch.cuda.empty_cache(), torch.cuda.synchronize()

            # Logging
            train_metrics = {"step": step, "type": "train"} | metrics
            log(train_metrics)
            lmf.utils.panelprint(train_metrics, title=f"Train Step:{train_metrics['step']}")
            lmf.utils.panelprint(
                {
                    "problem": questions[0],
                    "solution": gsm8k_answers[0],
                    "expected": q2a.get(questions[0], None),
                    "reward": rewards[0],
                }
            )

            step += 1
            if step >= config.training["total_steps"]:
                break

    os.makedirs(config.out["model"], exist_ok=True)
    save_path = os.path.join(config.out["model"], f"final_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    agent.policy.merge_and_unload().save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)


if __name__ == "__main__":
    main()
