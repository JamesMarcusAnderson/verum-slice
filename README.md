# Verum Slice Native

This tree is a clean restart from the uploaded prototype.  It intentionally has
no model-family switch and no fixed transformer layer struct.

The runtime is split into four native Objective-C responsibilities:

1. `VBSGGUFModel` keeps the GGUF file mmap-backed, preserves metadata, validates
   canonical GGML type geometry, and exposes block-aligned two-dimensional tiles.
2. `VBSGraphCompiler` derives a declarative operation graph from metadata, tensor
   names, shapes, and repeated scopes.  Ambiguous semantics are a hard error.
3. `VBSMetalCompiler` generates and compiles the exact MSL function required for
   an operation/encoding/tile shape.  There is no static Llama execution graph.
4. `VBSSliceExecutor` owns two bounded staging/private windows per GPU and streams
   rows *and* columns from mmap.  Tensor size is therefore independent of VRAM;
   only the selected packed tile and live activation state must fit.

`VBSFabric` treats multiple Metal devices as separate memory domains.  It probes
for peer groups when the SDK exposes them and otherwise uses an explicit shared
staging copy.  It never pretends that copying into a newly allocated destination
buffer changed the caller's source buffer.

## Build on the target Mac

```sh
make
./verum-slice --inspect /path/model.gguf
./verum-slice --compile /path/model.gguf
```

`--compile` validates the inferred graph and compiles its generated MSL kernels.
Execution is rejected when the file does not contain enough semantic evidence to
derive a unique graph; the engine does not guess a model family.

This clean restart currently ends at graph and GPU-harness compilation. It does
not falsely expose an interactive generation command before the inferred graph
executor and tokenizer are connected. The old tree could print `READY`, but it
could not correctly execute its own claimed memory or model contract.
