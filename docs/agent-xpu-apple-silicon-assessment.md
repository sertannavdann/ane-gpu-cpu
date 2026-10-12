**Agent.xpu on Apple Silicon: separate assessment of the R and L series**

Evidence reviewed: 19 September 2026. This is a source and recorded-results review, with arithmetic recomputed from R4 window records and its power log. No model or hardware benchmark was rerun. Existing experimental artifacts and gates are preserved.

**Assessment.** An Apple implementation of Agent.xpu's scheduling ideas remains technically plausible. R establishes native ANE execution and a useful, but limited, placement tradeoff. It does not yet establish an Agent.xpu execution engine or satisfy the original adoption gate. L establishes a diagnostic framework for one compression hypothesis; it has no native ANE performance result. The largest unresolved question is whether useful ANE prefill can overlap GPU decode after accounting for state exchange, CPU work, memory traffic and scheduling overhead.

The evidence is specific to the recorded M4 Max / 36 GB host, macOS 26.5 build 25F5068a, and pinned ane.cpp commit `b87bf57356c2`. Private runtime behavior, compiler decisions and measured performance must not be generalized to every Apple Silicon generation.

**What Agent.xpu actually requires**

The supplied [paper](pasted-text.txt) describes a single central LLM with a heterogeneous execution graph. Token-wise prefill projections and FFNs primarily use the NPU; attention and all decode use the GPU by default. It combines bounded prefill kernels, decode batching, preserved per-request state, cooperative preemption, and contention-aware admission. Some prefill placement is elastic. This is substantially more than assigning a background model to an accelerator. The implementation and evaluation use Intel Core Ultra and OpenVINO; the published gains are not Apple performance predictions. See also the [versioned paper](https://arxiv.org/html/2506.24045v2).

The current R4 arrangement is a GPU foreground model plus a separate background model placed either on GPU or on ANE/CPU. Moving the entire background request also moves its decode work. Consequently, slow ANE decode in R4 does not refute the paper's principal placement hypothesis: that hypothesis keeps decode on GPU.

**R series: what the measurements establish**

| Evidence | Established result | Interpretation and limit |
| --- | --- | --- |
| R0/R1 | Native compile/load/evaluate works; the visible projection lowering reproduces upstream MIL. The later R1 report has 378 numerical cases and zero gated failures, including a real activation fixture. | ANE linear computation is feasible on this host. This is a backend foundation, not heterogeneous scheduling. |
| R1 timing | Fits across records are roughly 50–61 microseconds fixed cost plus 7.0–11.8 microseconds per decimal MB of FP16 weights. | Both dispatch and weight traffic matter. Approximately 85–143 GB/s is a fit-derived effective rate, not a measured physical ANE bandwidth ceiling. |
| R2 | Splitting the FFN intermediate dimension into two/four dispatches loses latency. The exp-form SiLU lowers worst normal-input relative L2 from about 0.022 to 0.0026 at similar invocation time. | Preserve fusion and numerical validation. This rejects the tested FFN split; it does not reject chunking prompt tokens to improve preemption. Extreme and basis-input qualifications remain. |
| R3 | Qwen3.5-4B fixture v2: ANE conditions 24/30 correct; BF16 MLX 23/30; all have 30/30 schema validity. | No task-score regression is observed on this small set. Neither backend reaches the 27/30 gate. Similar scores do not establish numerical equivalence or broad task quality. |
| R4 | Concurrent GPU foreground and ANE/CPU background operate successfully with one ANE session. | There is an observed latency/power/throughput tradeoff, not a general performance win. |

Sources: [R1](../results/ane-r1-projection/report.md), [R2](../results/ane-r2-ffn/report.md), [R3 v2](../results/ane-r3-model-4b-v2/report.md), and [the R plan](agentic-offload-prototype-plan.md). The latest R1 report's 61 microseconds + 7.0 microseconds/MB fit supersedes treating the earlier 89 GB/s estimate as a constant. Three related small projection sizes cannot establish a universal roofline.

R4's primary record is the later energy run, after the GPU generator/KV retention defect was fixed. Its foreground is Qwen2.5-3B 4-bit; its background is Qwen3.5-4B, BF16 under MLX and FP16 hybrid under ane.cpp. These are useful deployment-path comparisons, but precision and runtime are not independently controlled.

| Metric | Foreground alone | Background on GPU | Background on ANE/CPU |
| --- | ---: | ---: | ---: |
| Foreground TTFT p95 | 194 ms | 296 ms | 223 ms |
| Foreground median decode rate | 163.0 tok/s | 162.2 tok/s | 135.6 tok/s |
| Correct background tasks/min | — | 37.73 | 6.80 |
| Average reported CPU+GPU+ANE power | 1.5 W | 18.5 W | 3.8 W |
| Reported rail energy/correct background task | — | 29.5 J | 34.0 J |
| Correct background tasks / offered | — | 566/743 = 76.2% | 102/129 = 79.1% |

Source: [R4 report](../results/ane-r4-energy/report.md), [energy aggregation](../results/ane-r4-energy/energy.md), and [raw windows](../results/ane-r4-energy/windows.jsonl).

Against GPU placement, ANE placement gives about 25% lower foreground p95 TTFT, 16% lower foreground decode speed, and 82% lower correct background throughput. Its aggregate correct tasks per kJ are about 13% worse. The two ANE windows without the large swap increase have TTFT p95 of 161–162 ms; the first reaches 347 ms. Both aggregate and per-window results matter.

Foreground response completion also deserves attention: per-window p95 completion times are approximately 669–686 ms alone, 722–765 ms with GPU background, and 744–943 ms with ANE background. Earlier first output does not automatically mean an earlier finished response, particularly for longer answers.

The existing single-GPU scheduler already demonstrates substantial value: foreground p95 TTFT falls from 1,206 to 180 ms with token-boundary priority on the recorded replay. This is evidence that scheduling contributes independently of adding an accelerator. It is a different workload from R4, so those numbers should not be combined into one speedup. Sources: [baseline replay](../results/mlx-baseline-run2/report.md), [priority replay](../results/mlx-priority-run2/report.md).

**R4 conclusions that need qualification**

1. **The written correctness gate is unmet.** The plan requires correct background completions / offered requests >=90%. The energy run yields 76.2% and 79.1%. Approximately 99.6% and 97.7% merely finished within the window. The recorded owner exception allowing exploratory R4 despite R3 is valid scope history; it does not turn these outcomes into passed correctness gates.
2. **The energy scope is CPU+GPU+ANE.** That is the field integrated by [energy.py](../prototype/r4/energy.py) from [powermetrics](../results/power-r4.log). Calling it whole-system or battery energy is unsupported; the recording does not independently establish total device consumption or DRAM energy coverage.
3. **Marginal energy is encouraging, but is an estimate.** Subtracting recorded foreground-alone energy gives `(3470−1317)/102 ≈ 21.1 J` for ANE and `(16681−1317)/566 ≈ 27.1 J` for GPU, approximately 22% lower for ANE. Adjusting the floor by actual recorded duration gives approximately 20.8 versus 27.1 J, or 23% lower. The narrative's exact 20.3 J is not reproduced by these aggregate calculations. These estimates assume a comparable foreground floor despite changed execution time and clocks.
4. **Low-rate benefit has not been directly measured.** Background submissions are completion-driven in [run_r4.py](../prototype/r4/run_r4.py): each placement accepts another case when it finishes the last. GPU therefore offers 743 jobs and ANE 129. This measures saturated capacity, not equal offered demand. A GPU that completes a few jobs quickly and idles could compare differently with a continuously active ANE. The slower path also sees fewer full cycles of the fixture. Replay identical arrivals and case order at several rates before recommending a low-rate operating policy.
5. **Energy and completion windows differ slightly.** Foreground-alone spans end around 296.8 s; GPU spans around 300.1 s; ANE spans around 303.8–305.1 s because the worker is joined. Energy integrates those actual spans, while throughput and completion use nominal 300 s. Use one explicit interval and consistent treatment of unfinished work for final adoption comparisons. Startup/load energy is largely outside these windows and needs separate accounting.
6. **Interference is observed; its cause is unresolved.** The 16% decode reduction is compatible with CPU scheduling, memory/cache contention, power management or their combination. Aggregate CPU/GPU/ANE power does not identify their contributions. ANE/CPU placement does not provide complete SoC isolation.
7. **Memory observations need precise naming.** The first ANE energy window's approximately 4.5 GB increase is net swap used, not a measured count of bytes swapped in. Near-zero free pages alone also do not prove pressure. The [27B smoke](../results/ane-r4-27b-fg-smoke/report.md) shows a much stronger practical failure: about 6.7 GB swap growth and 6.9 s foreground TTFT p95 in its ANE window. Use pressure, page-in/page-out counters and total relevant residency, including load peaks.
8. **The comparison has limited coverage.** There are three mixed-workload windows per placement on one host, repeating a small fixture; repeated tasks are not independent quality samples. The manifest records battery power. Graphics frame times, sustained thermal behavior, other Apple chips and a tuned quantized GPU background model were not evaluated. Keep the matched-precision experiment and the best deployable GPU comparison as separate tracks.

These issues support a narrower decision than the existing recommendation: retain ANE as a candidate for bounded background work; demonstrate the proposed rate, quality, memory and energy contract before adoption.

**What R still needs to become an Agent.xpu-style engine**

| Missing capability or evidence | Why it matters | Smallest decisive experiment |
| --- | --- | --- |
| Efficient token-wise ANE prefill | R4 does not establish the compute-heavy ANE stage the paper exploits. | Profile useful token throughput at several prompt chunks, with packing, attention and synchronization included. |
| One request continued across ANE and GPU | Separate model workers exchange text, not the execution state of one inference. | Execute one selected prefill block through ANE and resume the same request through GPU with matched outputs. |
| Explicit buffer and state ownership | Shared DRAM does not prove compatible layouts, shared allocation or synchronization. | Measure a complete producer/consumer round trip, including any copies and dtype conversion. |
| Resumable prefill | A Python `next()` includes initial prompt processing; foreground cannot interrupt it merely because decode yields per token. | Yield between bounded prompt chunks or selected kernels, preserving activations, positions and cache state. |
| Safe batching and per-flow state | Qwen3.5 multi-session lanes are reported corrupt beyond the first. | Verify session isolation, repeated resets and mixed lengths before counting batching gains. |
| Apple contention model | Concurrent compute units still share memory and power resources. | Pair ANE prefill with GPU decode, then memory-heavy work with memory-heavy work, and compare both to isolated controls. |
| Load-sensitive scheduling | Current GPU policy parks background work while foreground runs; it does not implement adaptive decode co-batching or an HEG coordinator. | Admit background work against measured token-gap and TTFT budgets; test bursts and sustained foreground load. |
| Model/task fidelity | R3 uses BF16/FP16 paths and comparisons restricted to intersecting top-16 logits before divergence. | Capture full logits on identical forced prefixes, add matched FP16 controls and a fresh task confirmation set. |

There is a concrete prefill issue worth investigating before building general scheduling machinery: [Qwen3.5 prefill](../prototype/ane/vendor/ane.cpp/models/llm/qwen3_5.cpp) defaults to chunks of four but still calls `ane_matvec` separately for each token's first projection. Other sections use batched operations. This is a visible source of incomplete weight reuse, not proof of the total bottleneck. Instrument the loaded binary's actual path before attributing the roughly 25–27 prompt tok/s to ANE hardware.

Qwen3.5 also carries recurrent DeltaNet state as well as full-attention KV state. An Apple port using this model needs to preserve both. The paper's dense Llama implementation is not a ready-made state contract for it. gRPC transport or a more general router would not resolve these execution gaps.

**L series: what exists and what it can contribute**

The [L1 report](../experiments/ane-latent/L1-REPORT-2026-09-19.md) records 34 passing framework tests, 84 synthetic CPU cases and four analytical estimates. It explicitly records no ANE execution, no real-model calibration, and no live native/MLX replay adapters. The report's earlier comments about unfinished R4 and unavailable energy are historical; the later R4 energy artifacts supersede them.

The synthetic outcomes are useful diagnostics: constructed low rank and zeros pass; flat spectra, shifted activation support, tiny amplitudes and unseen overflow expose failure modes. The 32 passing and 52 failing numerical cases are not a real-model success probability. They show that the framework recognizes intentionally constructed conditions.

L approximates one Qwen3-0.6B FFN down-projection, `Wd: [3072,1024]`, with two factors:

`Y = H Wd`, `A = Wd V`, `B = Vᵀ`, and `Yapprox = (H A) B`.

Here `V` contains leading eigenvectors of the uncentered calibration second moment `YᵀY`. This can reduce streamed weight bytes. It does not add preemption, shared state, batching, or stage coordination, and it changes the model's numerical function. It should therefore remain an independent optimization experiment while R establishes scheduling value.

| Rank | Dense → factor weight bytes | Down-projection reduction | Estimated saving with no extra host call | With one extra call |
| --- | --- | ---: | ---: | ---: |
| 256 | 6.29 → 2.10 MB | 66.7% | 29.4–49.5 microseconds | −31.6 to −0.5 microseconds |
| 512 | 6.29 → 4.19 MB | 33.3% | 14.7–24.7 microseconds | −46.3 to −25.3 microseconds |

These are [cost-model estimates](../experiments/ane-latent/theories.py), not timings. They omit intermediate traffic, compiler choices and changes in effective throughput. Both convolutions need to remain within one host invocation to have a credible starting case. Even then, one MIL program does not prove one hardware operation or no DRAM spill.

**Two bounds make the L feasibility question sharper.**

First, in exact arithmetic on the calibration population,

`relative Frobenius error² = sum(discarded eigenvalues) / sum(all eigenvalues)`.

Thus meeting a 0.005 relative-L2 target on this calibration population requires at least **99.9975% of output energy** to lie in the retained subspace, even before factor rounding and ANE arithmetic. That is a demanding condition at ranks 256 or 512. Passing the aggregate calibration condition does not imply passing the per-case held-out gate. Balancing can improve numerical range; it cannot restore discarded directions. Measure the real activation spectrum before investing in native lowering. This is a derivation from the implemented [factorization](../experiments/ane-latent/numerics.py), not an observed property of Qwen activations.

Second, 66.7% less down-projection storage means only **22.2% less weight storage across the three FFN matrices**, and about **13.3% less across the R1 model's listed projection/FFN weights per layer**. Fixed dispatch, attention, norms and sampling remain. Against the old approximately 14 ms projection-only token estimate, saving 29–50 microseconds in the single selected layer is only about **0.2–0.4%**. Even hypothetical equal savings in all 28 layers would be approximately 0.8–1.4 ms before other model costs. Those all-layer savings are outside L's scope and unvalidated. L cannot reasonably explain away R4's roughly 5.5× throughput gap; R4 also uses a different model.

The missing L evidence is therefore specific: representative disjoint calibration and evaluation activations; native compilation and actual graph selection; factorized versus dense full-invocation timing; full-FFN timing with the same retained SiLU correction; live full-logit fixed-prefix replay; and task preservation on a qualified model. Its [plan](../experiments/ane-latent/PLAN.md) currently gates native work on qualification or an explicit kernel-only scope revision. No gate is changed by this assessment.

**Using Apple Silicon's strengths and working around its costs**

| SoC/stack characteristic | Opportunity | Constraint and design response |
| --- | --- | --- |
| Unified physical memory | Keep request state resident and avoid discrete-device transfer paths. | Prove allocation, layout and synchronization compatibility. Current native code packs/unpacks IOSurfaces; MLX sharing is not automatically ANE interoperability. |
| ANE tensor layout and aligned lanes | Batch useful prompt positions or independent, validated sessions to reuse weights. | One decode token leaves little useful sequence work. Preserve channel-first layout across several operations rather than repeatedly transposing. |
| Specialized low-power ANE execution | Test bounded evaluation, embedding/reranking or compute-heavy prefill while GPU serves interactive work. | These are candidates, not measured wins here. Compare equal demand and useful quality, including all relevant energy. |
| Strong GPU and shared memory bandwidth | Keep dynamic attention and decode on GPU, with bounded prefill and opportunistic batches. | Concurrent memory-heavy ANE work may reduce GPU decode speed. Dispatch according to measured interference, not idle-engine availability alone. |
| Static specialization | Reuse a small set of prompt shapes and precompiled kernels. | Account for padding, compile/load peaks and cache growth. Online compilation is optional and must earn its CPU/memory cost. |
| Finite unified-memory capacity | Reuse buffers and cap resident contexts and model representations. | Loading an ANE worker can disturb GPU residency. Gate on peak load memory as well as steady state. |
| Compiler-selected precision and operation fusion | Use fused graphs and precision-aware validation. | Algebraically equivalent expressions can differ numerically, as R2 shows. Separate compression loss from dtype and implementation effects. |
| Public and private execution interfaces | Public Core ML is a deployment route; the existing private path offers useful research control. | Neither current route establishes the required cross-runtime scheduler. Pin private dependencies and test the public route separately where distribution matters. |

Apple's transformer guidance supports channel-first `[B,C,1,S]`, linear-to-1×1-convolution lowering, 64-byte alignment of the final axis, minimizing layout copies, and improving weight reuse for short sequences. This independently motivates the shape and traffic experiments above; its older model/device results do not predict this host's speed. [Apple's ANE transformer guidance](https://machinelearning.apple.com/research/neural-engine-transformers).

Avoid two overly broad architectural claims. Core ML supports stateful models and flexible/enumerated input shapes; neither dynamic length nor persistent state is categorically impossible on Apple. Their placement and latency still need verification. [Stateful models](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html), [flexible inputs](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html). Apple also documents increased M4 ANE INT8×INT8 throughput and recommends per-channel scales for NE quantization. The FP16 prototype has not exhausted compressed execution options, although format support does not guarantee a speed or quality win. [Quantization performance](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-perf.html).

Core ML's compute plan exposes anticipated device usage and estimated operation costs. That is useful profiling information, not evidence of application-controlled operation preemption. MLX's documented Apple device support is CPU/GPU; its unified arrays alone do not supply the missing ANE backend. [MLComputePlan](https://developer.apple.com/documentation/coreml/mlcomputeplan-85vdw), [MLX](https://github.com/ml-explore/mlx).

The host's recorded 36 GB capacity and 14 CPUs match the 14-core M4 Max configuration, whose advertised bandwidth is **410 GB/s**, rather than the 16-core variant's 546 GB/s. Accordingly, the plan's “one sixth of M4 Max bandwidth” comparison is configuration-sensitive and should not be used as a utilization claim. Neither advertised bandwidth nor the local timing slope measures concurrent usable bandwidth. [Apple M4 Max specifications](https://support.apple.com/en-us/121553).

**Recommended next decisions, in order**

1. **Establish a clean decision baseline.** Freeze a fresh task confirmation set; retain precision-matched controls alongside optimized deployment comparisons. Replay equal offered background arrivals, fixed case order and exact measurement intervals. Report correct completion, TTFT, token gaps, completed-response latency, rail energy and memory. Include the recorded swap-affected result as an operational outcome, with warm steady state reported separately.
2. **Bound GPU prefill before adding mechanisms.** Make the scheduler yield between prompt chunks and compare with the existing priority baseline. This directly tests whether the remaining GPU TTFT penalty needs a second accelerator. R2's rejected intermediate-dimension splitting does not answer this question.
3. **Test the paper's actual Apple opportunity.** Use one validated model and an ANE prefill block overlapping GPU decode. Sweep a small chunk set. Measure the complete exchange path and both tasks' slowdown, with CPU-only remainder and residency controls. Retain offload only when the saved GPU work exceeds handoff, conversion, synchronization and induced interference, or when an explicitly measured energy objective justifies the latency tradeoff.
4. **Make state and scheduling explicit only as needed.** Preserve stage/layer/chunk progress, activation ownership, KV and recurrent state. Validate resumable boundaries and session independence. Then add adaptive batching or placement from measured TTFT/token-gap slack. Do not promise paper-level sub-100 ms waits before bounding the actual nonpreemptible units.
5. **Keep L bounded and independent.** When its entry conditions permit, inspect the real spectrum first, test ranks 256/512 in one native graph, and compare the complete FFN. Stop if either numerical fidelity or complete-invocation benefit fails. A retained kernel is research evidence; model, energy and placement adoption remain separate decisions.

The strongest current direction is an Apple-specific scheduling policy with GPU decode and carefully bounded background work. ANE prefill is the main untested route toward Agent.xpu. L may reduce its weight traffic, but native execution, interoperability, reliable state and measured contention determine whether that improvement becomes a useful system.
