# Startup Development Guideline

## Customer value with minimal feature scope

Version 1.0 | 19 September 2026

Build the smallest complete workflow that delivers a defined customer outcome. Add work only to deliver that outcome, preserve essential quality, or resolve a material uncertainty.

This guideline defines a development process and architectural default for small teams with uncertain demand and limited resources. The SoC scheduling example illustrates their application.

## 1 Operating policy

### The development pattern

Use **outcome driven incremental development**: identify a problem, select an observable outcome, deliver a small end-to-end slice, and decide the next step from evidence. Each slice completes a customer task across the necessary interface, logic, and data or device operations.

The default structure is a **modular monolith**: one deployable application or runtime with clear internal responsibilities. Keep implementation details behind stable boundaries; add architectural mechanisms when a current constraint requires them. [1, 2]

### Rules for admitting work

1. **Name the customer and problem.** Every feature must support a current customer outcome. Infrastructure work must identify the outcome or operational constraint it enables.
2. **Apply the deletion test.** If removing an item would still allow the current acceptance scenario, quality guardrails, and planned learning to succeed, defer it.
3. **Complete one narrow workflow.** Minimize supported cases and options while preserving a usable beginning, result, and necessary failure handling.
4. **Start with existing capabilities.** Prefer a manual trial, configuration change, or existing component when it can establish the same evidence at lower cost.
5. **Keep scope fixed during a slice.** New ideas enter a later backlog. Urgent additions require a recorded scope trade-off or a new commitment.
6. **Measure before optimizing.** Add an optimization only against an observed shortfall, with a baseline and a predefined acceptance threshold.
7. **Treat correctness and essential reliability as scope.** Protect the data, resources, and user promises involved in the selected workflow. [3]
8. **Stop when the outcome is met.** Further capability needs a new customer justification. Remove experiments and temporary mechanisms that no longer serve a decision.

**Must** is a requirement; **should** permits a documented exception; **may** is optional. Record an exception's constraint, owner, consequence, and review trigger in the existing issue or pull request.

### Ownership without extra bureaucracy

Assign an outcome owner for customer success and scope, and a delivery owner for technical readiness. One person may hold both roles. Keep the brief, scope, and outcome evidence in one issue; link implementation evidence from the pull request and identify the shipped version. Add documents only for concrete decisions. [1]

<!-- pagebreak -->

## 2 Define and deliver a specific goal

### The one page outcome brief

Complete this brief before committing to a production slice. Use existing evidence honestly; if the problem or baseline is unknown, the next task is a discovery experiment.

| Field | Required content |
| --- | --- |
| Customer and situation | Who experiences the problem, and when it occurs. |
| Problem and evidence | The current obstacle, its consequence, and the observation supporting it. |
| Outcome | What the customer will be able to achieve. Avoid naming an implementation as the outcome. |
| Success measure | Baseline, target, measurement method, population, and observation window. |
| Smallest workflow | The complete scenario to deliver and its functional acceptance example. |
| Included and deferred | Explicit scope boundaries, including unsupported cases. |
| Quality guardrails | Relevant limits on errors, response time, resource use, data handling, or cost. |
| Delivery commitment | Outcome owner, delivery owner, effort cap, and review date. |
| Decision after review | Conditions for keeping, changing, expanding, or stopping the work. |

For a rate, specify both numerator and denominator. For latency, specify start and end events, units, workload, and percentile. Include failed attempts in the outcome report rather than allowing them to disappear from a success metric. When evidence is qualitative, record the observed behavior and its limitations.

### The delivery cycle

1. **Understand.** Observe the customer task and current workaround. Identify the most consequential uncertainty. An interview can establish a problem hypothesis; actual use tests whether the solution helps.
2. **Reduce.** Apply the deletion test to every proposed item. Choose the smallest complete workflow or the cheapest experiment that can answer the immediate question.
3. **Build.** Implement that workflow through the necessary components. Keep work small enough to review and revert. Use one active customer slice per small delivery team as the starting work limit.
4. **Verify and release.** Demonstrate the acceptance scenario, check the relevant failure paths, and release to the intended pilot or customer group with a recovery path.
5. **Observe and decide.** Compare the result with the brief. Record customer evidence, quality results, and the decision before increasing scope.

A research spike must have a question, effort cap, and a decision it will inform. Its output may be a benchmark, prototype, or finding. Production integration is a separate scope decision.

### Choosing the next item

Address defects that break the current customer promise. Then choose the smallest experiment that resolves the largest uncertainty about value or feasibility. Once those uncertainties are acceptable, deliver the next measurable outcome. Use effort and ongoing operating cost to break ties. When an effort cap is reached, reduce scope or explicitly replan; do not expand the commitment silently.

<!-- pagebreak -->

## 3 Keep the architecture proportional to the product

### Default structure

Organize one deployable application or runtime around the current workflow. Separate responsibilities in code where they have different reasons to change. The following are logical boundaries; they do not require separate services, packages, or one class per operation. [2]

| Responsibility | Owns | Boundary rule |
| --- | --- | --- |
| Entry and presentation | UI, API, CLI, or device input; input validation and result presentation. | Calls application operations without embedding product policy. |
| Application and domain | The customer workflow, business rules, decisions, and state transitions. | Defines the behavior; avoids dependence on vendor-specific details where they would obstruct change or testing. |
| Infrastructure adapters | Storage, external services, inference engines, or device execution. | Exposes only operations the current workflow needs; contains integration details. |

Use direct calls inside the application. Give each piece of mutable state a clear owner. Add persistent storage only for information that must survive execution, and start with one authoritative store when persistence is required. For an existing project, use established boundaries and refactor locally when they prevent the current outcome.

An interface is justified when it isolates a volatile dependency, expresses an important contract, or supports a needed test. It does not need a speculative second implementation. General-purpose frameworks and extension points require actual variation that the project must support. [2]

### Evidence required for additional complexity

| Proposed mechanism | Add it when | Start with |
| --- | --- | --- |
| Separate service | A current scaling, deployment, ownership, or isolation constraint requires an independent boundary. | A module and direct calls. |
| Queue or asynchronous worker | Work must outlive the request, absorb a measured burst, or avoid blocking a required interaction. | A direct operation; if insufficient, a bounded queue with explicit ownership. |
| Cache | Repeated work is a measured bottleneck and freshness rules are defined. | The authoritative read or computation. |
| Plugin or strategy framework | Supported variants already require different behavior behind a stable contract. | One implementation or a small explicit function. |
| Custom engine or optimization | Existing components fail a defined constraint, demonstrated by a representative benchmark. | A proven component with a narrow adapter. |

Record the rejected simpler option and the evidence in the change that introduces the mechanism. Include how it will be removed or simplified if the expected benefit does not appear. Avoid mandatory architectural layers, generalized event buses, multiple data stores, and configuration options that have no current user or operational need.

For latency-sensitive or concurrent systems, adapt the execution model to measured requirements. Keep coordination, resource ownership, and device details explicit. A modular monolith can contain asynchronous execution without becoming a distributed system.

<!-- pagebreak -->

## 4 Build and release with essential discipline

### Keep change reproducible

Version source, dependency versions, build instructions, relevant configuration, and data or model identifiers. A release must identify what was shipped and how it was built. Keep secrets out of source control. Integrate small changes frequently; defer broad refactoring unless it removes an obstacle or risk in the current slice. [1]

Prefer established tools. Automate a repeatable build and the checks needed for the current workflow. Use a short change note to connect the implementation with its outcome brief. Document behavior that users or maintainers must rely on, including supported limits and recovery steps.

### Preserve the quality floor

Implement failure handling in proportion to the dependencies and resources the workflow actually uses. Put timeouts on operations that may stall; bound admitted work and retained state; release resources when work finishes or is cancelled. Validate known prerequisites before expensive work. Use circuit breakers or resource isolation when a dependency can repeatedly fail or exhaust shared capacity. [3]

Test observable acceptance behavior, critical state changes, and the failure paths that could lose data, duplicate actions, or block the core workflow. Use integration tests for consequential boundaries. A small cosmetic change needs proportionate verification; avoid tests that merely restate trivial implementation details.

Use the smallest useful operational evidence: whether the task succeeded, how long it took, and why it failed. Add resource measurements when resource limits matter. Avoid a telemetry platform until simpler logs or counters cannot support an actual decision.

### Two distinct completion decisions

**Engineering complete** means the acceptance scenario works, relevant checks pass, the release is identifiable, and recovery is defined. For data changes, code rollback alone may be insufficient; record a compatible migration, backup, or forward-recovery approach appropriate to the change.

**Outcome validated** means the intended users achieved the predefined result within the stated guardrails during the observation window. A successful build, demo, or benchmark establishes technical evidence; customer behavior establishes product evidence.

| Observed result | Required decision |
| --- | --- |
| Customer outcome met and guardrails hold | Keep the solution. Add scope only for the next accepted outcome. |
| Technical behavior works but value is absent | Revisit the problem, workflow, or customer group before adding features. |
| Outcome appears promising but a guardrail fails | Correct the demonstrated shortfall within the current scope. |
| Evidence is insufficient | Extend or change the observation method with an effort cap and review date. |

Track customer outcome results and delivery effort. Do not use feature count, lines of code, utilization, or benchmark speed alone as a substitute for value. Small pilots support local decisions; they do not establish universal demand or performance.

<!-- pagebreak -->

## 5 Worked example for an agentic LLM scheduler

### Customer outcome and first slice

An on-device assistant user needs an interactive response while background agent work is present. The customer problem is waiting or losing responsiveness; using every accelerator is an implementation choice. Agent.xpu provides relevant techniques for coordinating reactive and proactive inference on a heterogeneous SoC. [4]

Begin with one target device, one model, one inference runtime, and a recorded workload. Establish the existing runtime baseline. Define foreground responsiveness and acceptable background progress from the intended interaction. Include waiting time in user-visible latency, and evaluate correctness alongside speed.

The first slice admits a foreground request while bounded background inference is active, produces the response, and resumes or completes the background work. Give requests explicit lifecycle state, bound queues and retained context, and support cancellation with safe resource cleanup. Keep tool execution outside the scheduler unless the current acceptance scenario requires it.

### Apply the architectural default

Use one runtime with a small admission boundary, scheduling policy, state owner, and backend adapter. Keep policy separate from device execution. First prove the required behavior using the existing engine. If its execution boundaries cannot meet the response target, make the smallest engine change that the measurements justify.

| Include for the first accepted workflow | Defer until a measured constraint requires it |
| --- | --- |
| Foreground and background classification; bounded admission; observable lifecycle and cancellation. | General workflow designer, plugin ecosystem, multi-device support, or cloud routing. |
| Priority selection at supported execution boundaries; a benchmark of the actual interaction. | Adaptive batching, predictive scheduling, online compilation, or general graph optimization. |
| One validated placement policy; a second accelerator path if heterogeneous execution is required by the project objective. | Elastic tensor partitioning and broad operator placement machinery. |

A single-backend baseline is a development step. A release claiming heterogeneous scheduling must demonstrate correct execution on the declared accelerator combination and satisfy its stated acceptance criteria.

### Acceptance and expansion gates

Record numerical targets in the outcome brief before optimization. Measure foreground time to first token from request submission to first visible token, and inter-token gaps from successive visible tokens. Report the selected percentiles, workload, sample counts, failures, peak memory, and background completion rate. Foreground latency measures responsiveness; completed background requests measure useful progress.

Preemption must respect the backend's supported safe boundaries. A timeout does not prove that a device stopped using memory. Shared physical memory does not by itself establish a zero-copy contract. Priority does not guarantee background progress under sustained foreground saturation; define an admission limit or an explicit capacity reservation if progress is a requirement.

Add NPU placement, finer scheduling boundaries, or batching one at a time when a baseline shows the need. Reuse the same model, workload, and device conditions; keep a change only if its target improves without violating the declared guardrails. Paper results and measured latency percentiles are not hard real-time guarantees or portable performance promises. [4]

<!-- pagebreak -->

## 6 Enforce the rules in everyday development

### Pull request review

Use the existing pull request description to answer these questions. Link to evidence rather than maintaining duplicate reports.

1. Which outcome brief or demonstrated defect does this change address?
2. What would fail if this change were omitted? Which proposed items were deferred?
3. Does the change introduce a service, dependency, abstraction, configuration option, or operating obligation? What current constraint requires it?
4. What acceptance behavior and relevant failure paths were verified?
5. What baseline and result support any performance or reliability claim?
6. How will the change be released, observed, and recovered or removed?

A reviewer must reject unexplained scope. Where available, require linked work and passing build checks before merge. Human review enforces purpose and scope; automated checks enforce repeatable technical conditions. Templates alone do not enforce minimal development, and CI cannot establish customer value.

### Recognize and correct scope growth

| Warning sign | Immediate correction |
| --- | --- |
| The team is building a framework for possible future use. | Implement the current workflow and defer speculative capabilities. |
| The slice keeps expanding during implementation. | Reapply the deletion test and record the revised commitment. |
| An optimization has no representative baseline. | Measure the actual shortfall before changing the design. |
| The demo works but recovery or ownership is unclear. | Complete the relevant reliability and release work. |
| Process paperwork grows faster than decisions. | Consolidate records and remove steps that support no current decision. |

### Source basis

The outcome gates and scope controls are this guide's operating policy. The sources inform its architectural and engineering choices.

**[1] Brown, McCormick, and Thomas.** *AntiPatterns and Patterns in Software Configuration Management* (1999), Executive Summary, printed pp. xv-xvi, PDF pages 13-14. Minimal process with configuration management, requirements traceability, and testing.

**[2] Buschmann, Henney, and Schmidt.** *Pattern-Oriented Software Architecture Volume 4: A Pattern Language for Distributed Computing* (2007), section 3.5, pp. 49-51; Layers, pp. 185-187; Encapsulated Implementation, pp. 313-316; Strategy, pp. 455-456. Context-dependent patterns and explicit boundaries. The modular-monolith default is this guide's recommendation.

**[3] Nygard.** *Release It! Design and Deploy Production-Ready Software*, first edition (2007), chapter 5, pp. 111-133: Use Timeouts, Circuit Breaker, Bulkheads, Steady State, and Fail Fast. Bounded resources and failure handling.

**[4] Wei et al.** *Agent.xpu: Efficient Scheduling of Agentic LLM Workloads on Heterogeneous SoC*, arXiv:2506.24045v2, 6 January 2026, sections 3-5. Supplied *Pasted text.txt*; [versioned paper](https://arxiv.org/abs/2506.24045v2). Placement, contention, and scheduling.
