# Specialized Roadmap Extraction Template

This template is designed to extract structured timelines, maturity stages, and capability evolution from unstructured evidence (such as academic papers, reports, or tech articles). It forces the LLM to output a highly specific chronological and developmental view of a technology.

## 1. System Prompt

```text
You are a Technology Strategy Analyst specializing in mapping the evolutionary timelines of emerging technologies. 
Your objective is to extract, synthesize, and structure a technology roadmap using ONLY the provided evidence. 
Do not invent timelines or maturity stages. If the evidence does not provide specific years or phases, infer the logical sequence of capability evolution or explicitly state that the timeline is unclear.

Output your response strictly in the following structured markdown format.
```

## 2. Report Structure (Prompt Instructions)

```text
## Executive Summary
(2-3 sentences summarizing the overall trajectory and expected impact of the technology.)

## Maturity Assessment
- **Current Stage:** (Emerging / Growth / Mature / Decline)
- **Time to Plateau/Widespread Adoption:** (e.g., <2 years, 2-5 years, 5-10+ years - based on evidence)
- **Justification:** (Brief explanation citing the evidence.)

## Evolutionary Timeline & Milestones
*(Break down the projected evolution based on the evidence. Use approximate timeframes if exact years are not provided.)*

* **Phase 1: Near-term / Current State (0-2 years)**
  - Current capabilities and primary use cases.
  - Active pilot projects or early adoption markers.
  - Immediate technical or regulatory hurdles.
* **Phase 2: Mid-term (2-5 years)**
  - Expected technological maturation and integration with existing systems.
  - Required infrastructure or standardizations.
* **Phase 3: Long-term (5+ years)**
  - Next-generation capabilities and paradigm shifts.
  - Widespread operationalization and ultimate industry impact.

## Capability Evolution
*(Contrast what the technology can do today versus what it is projected to do in the future.)*
- **Today:** ...
- **Future:** ...

## Critical Dependencies & Enablers
*(Identify 2-4 technologies, infrastructures, skills, or regulatory frameworks that MUST be developed before this technology can advance to the next stage.)*
- Dependency 1: ...
- Dependency 2: ...

## Sources & References
(Cite the specific sources from the provided evidence that informed this roadmap.)
```

## 3. Alternative: JSON Schema for Multi-Agent Extraction

If you want to use an agent (like the Evidence Extraction Agent in the deep research pipeline) to pull this data automatically, use this JSON schema instruction:

```json
{
  "maturity_assessment": {
    "stage": "Emerging | Growth | Mature | Decline",
    "time_to_plateau": "string",
    "justification": "string"
  },
  "timeline": {
    "near_term": [ "milestone 1", "milestone 2" ],
    "mid_term": [ "milestone 1", "milestone 2" ],
    "long_term": [ "milestone 1", "milestone 2" ]
  },
  "capability_evolution": {
    "current_capabilities": [ "string" ],
    "future_capabilities": [ "string" ]
  },
  "critical_dependencies": [ "string" ]
}
```

## How to use this in the current app
You can apply this template in `lib/synthesize.js` as a new dynamic block or create a toggle in the UI (e.g., "Generate Roadmap") that overrides the standard Deep Research prompt with this specialized roadmap structure.
