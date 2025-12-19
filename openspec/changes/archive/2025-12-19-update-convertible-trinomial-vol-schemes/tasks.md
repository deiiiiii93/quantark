## 1. Implementation
- [x] Add trinomial scheme enum and expose in tree params (default constant-vol scheme)
- [x] Update trinomial engine to branch by scheme (constant CRR, fixed-dx log, variable-dx log with re-gridding)
- [x] Add validation/warnings for unsupported term-structure usage under constant-vol scheme
- [x] Update API docs/comments in relevant modules

## 2. Tests
- [x] Add tests for scheme selection and parameter validation
- [x] Add term-structure regression tests for fixed-dx and variable-dx schemes
