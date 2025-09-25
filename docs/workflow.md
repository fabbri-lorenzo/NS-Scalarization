# Workflow overview

This page describes the high‑level logic executed by the `main.py`
script.  The goal of the program is to find scalar field profiles
(`σ(r)`) that vanish at a large radius `r_max`, subject to a physical
equation of state and coupling parameters.  The workflow can be broken
into three phases: scanning the bracket for candidate roots, refining
the roots via a shooting method, and integrating the stellar structure
to compute physical observables.

## Main control flow

The diagram below summarises the sequence of operations performed in
`main.py`.  Rectangular boxes represent function calls or major steps,
while diamonds denote conditional branches.  The green subgraph shows
the *shooting algorithm* implemented in `Utils/shooting.py`, which
performs the core root‑finding to determine acceptable central scalar
amplitudes.

.. mermaid::
%% Main workflow for main.py
%% Use the "TD" directive for a top‑down layout
graph TD
    A[Set initial parameters and constants]\
        -- define EOS, density, xi, lambda, bracket --\> B[Pre‑scan: diagnostic_scan()]
    B -- returns sign‑change brackets and residual samples --\> C[Set acceptance thresholds]
    C \n\n-- call shoot_sigma0() to find all sigma0 roots --\> D{shoot_sigma0}
    D -- returns non‑empty root list --\> E[Accept s0_list]
    D -- no roots or missing relative to brackets --\> F[Fallback: probe_brackets_with_brent()]
    F -- append any additional roots --\> E
    E -- print candidate info and align maxima --\> G[print_candidate_info()]
    G -- determine number of sign‑change brackets --\> H{iterate over each σ0}
    H -- integrate to 2R and count nodes --\> I[per mode: integrate_star(), node_count_to_2R()]
    I -- group candidates by node count --\> J[select lowest |σ0| per mode]
    J -- for each selected σ0 integrate again --\> K[integrate_star() twice]
    K -- compute ADM mass & scalar charge --\> L[save results & gather plot entries]
    L -- final plotting and output --\> M[printResults_multi() and CSV]

    %% Subgraph for the shooting algorithm itself
    subgraph shoot_sigma0
        direction TB
        S1[Generate uniform seeds across bracket]
        S2[Compute residuals at each seed (_residual)]
        S3[Determine scale for fsolve]
        S4[Augment seeds near residual minima]
        S5[For each seed: solve f(s)=0 via fsolve]
        S6[Filter roots inside bracket and above |σ₀|min]
        S7[Evaluate residuals and apply absolute/relative cuts]
        S8[Regularity check via _sigma2_psi2 (Ψ₂>0)]
        S9[Return unique sorted list of acceptable σ₀]
        S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7 --> S8 --> S9
    end

```

## Notes on the shooting method

The shooting algorithm is responsible for determining the central
scalar amplitudes `σ₀` that produce vanishing scalar fields at the
outer boundary `r_max`.  The routine samples the bracket `[a,b]`
uniformly, evaluates the residual (`σ(r_max)`) at those samples, and
then uses SciPy’s `fsolve` to refine each seed.  Roots are accepted
when either the absolute residual is below a threshold or the
normalized residual (`|f(r)/r|`) is sufficiently small.  Finally, a
regularity check rejects solutions for which the auxiliary function
`Ψ₂` (computed in `Utils.TOV_EMG._sigma2_psi2`) is non‑positive.  The
workflow above highlights how this subroutine integrates into the
overall control flow of `main.py`.
