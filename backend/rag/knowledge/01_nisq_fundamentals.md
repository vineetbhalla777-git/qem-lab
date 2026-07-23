# NISQ Era & Quantum Computing Fundamentals

## What "NISQ" means

NISQ stands for Noisy Intermediate-Scale Quantum. Physicist John Preskill
coined the term in a 2018 paper ("Quantum Computing in the NISQ era and
beyond") to describe quantum processors with roughly tens to a few hundred
(and now over a thousand) physical qubits that operate without full
quantum error correction. The label was meant to be honest about where the
field actually stood: real hardware exists and can run meaningful
experiments, but noise -- not qubit count -- is the binding constraint on
what these devices can reliably compute. As of 2026, the field is
generally described as being in a late-NISQ / early-utility transitional
phase: still short of fault tolerance, but hybrid quantum-classical
workflows are running in production settings. IBM's public roadmap targets
a large-scale fault-tolerant system (Quantum Starling) later this decade.

## Why noise dominates in this era

Every physical qubit is an open quantum system coupled to its environment,
so it gradually loses its quantum state -- a process called decoherence.
Two time constants describe this: T1 (energy relaxation / amplitude
damping -- how long before a qubit "falls" from |1> to |0>) and T2
(dephasing / phase damping -- how long superposition phase information
survives). Gate operations are also imperfect (finite gate fidelity), and
reading out a qubit's state at the end of a circuit introduces its own
readout error. All of this compounds: the deeper and wider a circuit, the
more chances for an error to occur somewhere in it.

## Quantum volume

Because qubit count alone doesn't capture how *useful* a device is, IBM
introduced "quantum volume" as a benchmark that weighs circuit width and
depth together, based on how well a device can execute random circuits
(the "heavy output generation" problem). Two devices with the same qubit
count can have very different quantum volumes depending on their gate
fidelities and connectivity.

## Error mitigation vs. error correction

These are frequently confused. **Error correction** encodes one *logical*
qubit redundantly across many *physical* qubits so that errors can be
detected and actively fixed mid-computation -- but it requires large
qubit overhead that today's hardware doesn't yet have to spare. **Error
mitigation** takes the opposite trade: it doesn't fix errors on the
device at all; instead it runs extra circuits and/or extra shots and uses
classical post-processing to *estimate* what the noiseless answer would
have been. It's far cheaper in qubits, works on today's hardware, but only
targets expectation values / distributions statistically -- it can't
rescue an individual, arbitrarily long computation the way full error
correction eventually will.

---
*Sources consulted and paraphrased: Preskill, "Quantum Computing in the
NISQ era and beyond" (arXiv:1801.00862); Wikipedia, "Noisy
intermediate-scale quantum computing"; IBM Quantum roadmap coverage
(2026); general quantum-computing hardware literature on T1/T2 and
quantum volume (Cross et al. 2019).*
