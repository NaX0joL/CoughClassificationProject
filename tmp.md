# Additional Workflow Structure Rules

## Natural workflow

- Give a workflow-owning class one public orchestrator method when callers only need one operation.
- Write the public orchestrator so its statements read like a plain-language summary of the workflow.
- Keep the public orchestrator at one abstraction level: it coordinates named steps but does not perform their mechanics.
- Keep simple domain-level loops in the orchestrator when seeing the loop makes the workflow easier to understand.
- Make each helper perform only the step described by its name.
- Helpers must return control to their caller and must not trigger a later workflow stage themselves.

## Barebone readability

- Prefer the smallest implementation that a beginner can understand in one reading.
- Keep the main data flow visible; do not hide a short, important sequence behind several layers of forwarding methods.
- Split a large workflow class only when it owns distinct responsibilities or policies.
- Give each resulting collaborator one clear responsibility and one primary public operation.
- Use ordinary classes with explicit constructors for objects that perform work or coordinate dependencies.
- Reserve dataclasses for passive data containers whose main purpose is storing values.
- Avoid intermediary classes, wrapper objects, and configuration objects unless they remove real ambiguity or enforce an important invariant.
- Avoid chains of tiny helpers. A leaf helper may contain a short cohesive block of related mechanics.
- Prefer direct names such as `OuterFoldSplitter` and `DevelopmentSplitter` that state exactly what each collaborator does.
- Preserve the existing public constructor and output shape during structural simplification unless the requested change requires otherwise.

## Abstraction levels

- Put raw mechanics only in leaf helpers. Examples include opening files, parsing external data, iterating through raw collections, looking up registry entries, calling constructors, and invoking external processes.
- Distinguish raw-collection loops from domain-workflow loops: raw parsing belongs in a leaf helper, while a short loop over folds, stages, or tasks may remain visible in the orchestrator.
- Keep orchestration methods free from parsing details, type branching, constructor arguments, and error-handling mechanics.
- When a helper mixes coordination and mechanics, extract the mechanics into a clearly named leaf helper.
- Do not extract a helper that merely renames one obvious expression unless doing so is necessary to keep its caller at a consistent abstraction level.

## Reading order

- Order methods from highest to lowest abstraction.
- Place the public orchestrator first, followed by its helpers in the order they are called.
- When a helper has its own helper chain, keep that chain together before moving to the next unrelated workflow branch.
- A reader should be able to understand the overall behavior from the public method before reading any implementation details.

## Interfaces and naming

- Keep the public interface minimal; make implementation methods private.
- Name orchestration methods after the outcome they produce and leaf helpers after the concrete operation they perform.
- Use one domain term consistently. Do not alternate between names such as `configuration`, `document`, and `data` for the same concept without a real distinction.
- Prefer explicit intermediate variables when they make the workflow read more naturally.

## Control flow and errors

- Validate external input at the boundary before constructing domain objects.
- Add file paths and nested field names when re-raising configuration errors.
- Keep the successful path visually prominent; error details belong in validation or leaf helpers.
- Do not silently recover from invalid configuration or failed construction.

## Project conventions

- Follow the project annotation format with no spaces around `:` or `=` in signatures.
- End functions annotated with `-> None` using an explicit bare `return`.
- Keep classes above top-level functions.
- Use three blank lines around top-level classes and global configuration sections, two between top-level functions, and one between methods.
