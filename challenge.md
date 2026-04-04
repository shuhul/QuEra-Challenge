# Can a 1-Qubit Gate Be Harder Than a 2-Qubit Gate?

## Summary

On paper, a 1-qubit gate looks easy to implement when compared to a 2-qubit gate.
In this challenge you will investigate when and why this intuition breaks down in practice.

Your team will study the family of 1-qubit rotations

$$
R_z(\pi/2^k),
$$

and build a technical case around a surprising claim (for the uninitiated!):

> A 1-qubit gate can be harder to implement than a 2-qubit gate.

Your final submission will be a **short technical write-up supported by code, figures, and simulations** (you can learn more about the specific deliverables in the [README](https://www.notion.so/README.md) file).
The strongest write-ups will make creative choices about how to introduce the topic, what examples to emphasize, and how to guide the reader toward the current frontier of quantum computing research.

Below we give a path for you to follow and learn more about the statement, while gathering evidence (in the form of code and visualizations) for your argument.

Following our suggested path, you will explore:

- QuEra’s open-source tools for building and simulating circuits,
- the standard Clifford+$T$ universal gate set,
- settings in which the $T$ gate is not directly available on the data qubit,
- how the story changes when moving from one physical qubit to one logical qubit.
- QuEra’s recent results on the **transversal STAR architecture**.

### Note on tools

For creating circuits and simulating them, we ask that you use the tools from QuEra’s Bloqade ecosystem:

- **Squin** for building circuits
- **PyQrack** for simulating small circuits
- **Tsim** for simulating larger, mostly-Clifford circuits
- **Cirq** (optional) for adding noise models and creating noisy circuits

For plotting data and visualizing state evolution, circuit structure, or gate action, you may use any tools you prefer.

---

# Suggested path

We **strongly** suggest that you break the work up into two sub-teams:
- **Team Synthesis** focused on exploring how to optimally synthesize 1-qubit rotations both physically and logically
- **Team STAR** focused on implementing early fault-tolerant rotation gadgets  

For each team we provide 4 different tasks of increasing difficulty. 
You do not need to complete all of them for a valid write-up submission.
Remember that the write-up and the presentation, along with the code to support them, are the key deliverables.

## Team Synthesis Part 1: Learn the language of Clifford+ $T$

Learn how to build and simulate simple circuits using **Bloqade Squin** and **Bloqade PyQrack**.

Build a few small 1-qubit and 2-qubit examples. Confirm that you understand how $H, S, T,$ and $CNOT$ act on simple input states.
Use this part to get comfortable creating, simulating, and visualizing circuits constructed from the Clifford+$T := \{H, S, CNOT, T\}$ gateset.

**Goal:** Build intuition for Clifford+T circuits and the simulation workflow.

---

## Team Synthesis Part 2: Synthesize the rotation family

Focus on the family

$$
R_z(\pi/2^n), \qquad n \in \{0,1,2,3,4,5\}.
$$

How can we implement these “dyadic” $Z$-rotations using only Clifford+$T$?

Try synthesizing these rotations as well as you can using only our chosen gate set for one qubit (yes, only 1 qubit), and different values of $n$. Some implementations may be exact while others may involve approximations, that is okay. It is up to you to explore different synthesis strategies and compare the circuits you find.

We suggest spending time on finding ways to visualize how your approximations act on different initial states, and reflecting on what you tried, how you judged quality, and what changed as the target angle became smaller.

**Goal:** Explore how small $Z$-rotations can be built from Clifford+ $T$ and explain the synthesis strategies you explored.

> **Distance metric**
>
> When comparing a target gate $U$ to an implementation $V$, use the following global-phase-invariant distance:
> 
> $$
> d(U,V)=\sqrt{1-\frac{|\mathrm{Tr}(U^\dagger V)|}{2}}.
> $$
>  
> Interpretation:
> 
> - $d=0$ means exact agreement up to global phase,
> - larger $d$ means a worse approximation.

---

## Team Synthesis Part 3: Non-Clifford gates are expensive

How would you approximate the family of rotations above if I now suddenly told you that $T$ gates can no longer be applied to your qubit? If you spent enough time thinking about part 1, the correct answer should be somewhere in the realm of "not good." Unfortunately, this is what often happens in reality (you will learn more about why this happens in the next part). It is up to you to find and implement a protocol to "inject" the effect of $T$ gates onto the main qubit.

To counteract this not-good news, you will now have access to auxiliary qubits on which you will be allowed to apply $T$ gates. You will now also be able to apply $CNOT$ gates using your main qubit as either target or control, as well as across auxiliary qubits. Other than that, it should be just $S$ or $H$ gates on your main qubit. We want you to approximate the rotations on your main qubit and benchmark them using the previous distance metric, working within the bounds of the simulators (i.e. if the circuit cannot be simulated the approximation does not count).

Track the new costs that appear: ancilla count, 2-qubit gate count, circuit depth, repeated trials, feed-forward, or any other relevant overhead.

**Goal:** Rebuild the same 1-qubit rotations in a setting where the non-Clifford gates in the Clifford+ $T$ set must be supplied indirectly.

---

## Team Synthesis Part 4: Move from one physical qubit to one logical qubit

![steane_code_construction.png](./assets/steane_code_construction.png)

Figure taken from *"Logical quantum processor based on reconfigurable atom arrays"* by *Bluvstein et al.*

Research the $[[7, 1, 3]]$
 Steane code and learn about which gates in the Clifford+ $T$ are "transversal" in this code. Then, construct a kernel that implements the $[[7, 1, 3]]$
 code (note, the circuit above encodes the $|0\rangle$ state) in Squin and use it to represent one logical qubit.

Keeping in mind which gates in the Clifford+$T$ set are transversal and which ones are not, explore how to apply the same family of target rotations on this one logical qubit. This will require careful circuit design, smart synthesis, and some form of injecting $T$-gates (but now onto a logical qubit).

Like in Part 3, benchmark your approximations, track the new costs that appear, and reflect on the overhead that moving from one physical qubit to one logical qubit creates. You can also explore the transition from one physical qubit to one logical qubit in a noisy setting (the tutorial we shared should help you think about how to add noise).

**Goal:** Show how the challenge changes when one physical qubit becomes one logical qubit (optionally with noise).

---

## Team STAR Part 1: Get comfortable using Tsim
![surface_code.png](./assets/surface_code.png)

Familiarize yourself with the surface code and its stabilizer structure. 
Use Tsim to construct a distance-3 surface code and simulate two rounds of syndrome extraction.

Start by identifying the data and ancilla qubits, the stabilizer checks being measured, and how syndrome information is extracted over time. 
Use this part to get comfortable building the code in Tsim, running repeated syndrome cycles, and understanding how errors affect the circuit.

**Goal:** Build intuition for the distance-3 surface code, repeated syndrome extraction, and the simulation workflow in Tsim.

## Team STAR Part 2: Estimate STAR Fidelities
Review the STAR circuits that have been provided in the assets folder and use Tsim to simulate them in a noisy setting. 
Reproduce the fidelity plot below using data from your Tsim simulations.

![star_sim.svg](./assets/star_sim.svg)

The provided circuits are for a default rotation angle of 0.01*pi. To simulate different rotation angles, you can use the following function to compute the physical rotation angle needed to achieve a logical rotation of angle `logical_angle_in_pi` on `num_physical_rotations` physical rotations.

Also make sure to check out the comments in `circuits/star_d=3.stim` to get some hints about the circuit structure.
```python
def physical_angle(logical_angle_in_pi: float, num_physical_rotations: int) -> float:
    """
    Compute the physical rotation angle needed to achieve a logical rotation of
    angle `logical_angle_in_pi` on `num_physical_rotations` physical rotations.

    Args:
        logical_angle_in_pi (float): The logical rotation angle in units of pi.
        num_physical_rotations (int): The number of physical rotations that are applied.
    Returns:
        float: The physical rotation angle in units of pi.
    """

    assert (
        num_physical_rotations % 2 == 1 and num_physical_rotations > 0
    ), "k must be a positive odd integer"
    sign = -1 if (num_physical_rotations + 1) % 4 == 0 else 1
    logical_angle_in_rad = logical_angle_in_pi * np.pi
    x = np.tan(logical_angle_in_rad / 2) ** (1 / num_physical_rotations)
    theta_phys = 2 * np.arctan(x)
    return float(sign * theta_phys / np.pi)
```

**Goal:** Learn how to load and run pre-built circuits using Tsim. 

## Team STAR Part 3 — Teleport a non-Clifford rotation into a logical qubit
Now assume a noiseless setting but where non-Clifford gates cannot be applied directly to the main logical qubit. 
Construct a protocol that uses an ancillary logical qubit to teleport a small-angle rotation into the main qubit while assuming the STAR transversal architecture.
Important: you will need to use postselection to filter results because Tsim (unlike PyQrack) does not support feed-forwarded operations.

![rus.png](./assets/rus.png)

Figure taken from *"Partially Fault-Tolerant Quantum Computing Architecture with Error-Corrected Clifford Gates and Space-Time Efficient Analog Rotations"* by *Akahoshi et al.*

**Goal:** Show how STAR enables indirect implementation of small-angle non-Clifford logical rotations, and analyze the costs of doing so. 

## Team STAR Part 4: Apply STAR in a larger algorithm
Construct a 4-qubit QFT circuit in which all non-Clifford gates must be teleported in rather than applied directly.

**Goal:** Demonstrate how teleported non-Clifford gates can be integrated into a larger logical circuit, and evaluate the scalability of the approach.

## References

Some works you might find useful throughout the challenge are:

1. *“Quantum Computing, Universal Gate Sets”* https://www.scottaaronson.com/qclec/16.pdf
2. *“The Solovay-Kitaev Algorithm”* https://arxiv.org/pdf/quant-ph/0505030
3. *“Optimal ancilla-free Clifford+T approximation of z-rotations”* https://arxiv.org/abs/1403.2975
4. *“Efficient synthesis of universal Repeat-Until-Success circuits”* https://arxiv.org/abs/1404.5320
5. *“Partially Fault-tolerant Quantum Computing Architecture with Error-corrected
Clifford Gates and Space-time Efficient Analog Rotations”* https://arxiv.org/pdf/2303.13181
6. *“Practical quantum advantage on partially fault-tolerant quantum computer”* https://arxiv.org/pdf/2408.14848
7. *“Transversal STAR architecture for megaquop-scale quantum simulation
with neutral atoms”* https://arxiv.org/pdf/2509.18294
---