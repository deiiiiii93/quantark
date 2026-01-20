## 1. Specification
- [x] 1.1 Add `autocallable-risk-report` spec delta and validate

## 2. Core Implementation (Snowball-first)
- [x] 2.1 Implement `AutocallablePathAnalyzer` (RN + historical replay + parametric)
- [x] 2.2 Implement cashflow attribution + event probability tables (per observation)
- [x] 2.3 Implement grid runner for surfaces (Spot×Vol, Spot×Dividend) and cross sensitivities
- [x] 2.4 Implement Markdown report generator with plot outputs (PNG) and tables
- [x] 2.5 Add CLI entrypoint (inputs: product config, pricing env, historical series)

## 3. Engine Integration (Optional Enhancement)
- [x] 3.1 Define an engine-level API for event stats / decomposition (interface + types)
- [x] 3.2 Provide an initial implementation path (MC-backed) and document QUAD/PDE roadmap

## 4. Skill Packaging
- [x] 4.1 Create Codex skill `autocallable-risk-report` with a guided workflow
- [x] 4.2 Bundle report templates and helper scripts in the skill
- [x] 4.3 Package `.skill` artifact and document installation

## 5. Tests
- [x] 5.1 Add unit tests for attribution/probabilities on simple Snowball configs
- [x] 5.2 Add smoke test for report generation output structure
