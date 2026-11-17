/-
# Proposition 3.10: optimal partitioning (bias–forgetting trade-off)

Formalized components of the proof in Appendix "Proof of Proposition 3.10":

1. `task_sub_bal_le`: risk mismatch `|𝓛_task - 𝓛_bal| ≤ (B/2) ∑ |p_c - q_c| = B · TV(p,q)`
   for class losses in `[0, B]`.
2. `tv_le`: for counts with local imbalance `max/min ≤ R`,
   `TV(p, uniform) ≤ (√R - 1)/(√R + 1)`.
3. `bal_penalty_le`: training on the imbalanced partition costs at most `2B (√R-1)/(√R+1)`
   in balanced risk.
4. `forgetting_le`: forgetting of one CL transition is at most
   `Δ_i · min(c_Φ/α_i, L^sm/(m_Ψ β_i))`.
5. `local_IF_exp`, `exists_gap_ge`, `equal_boundaries_optimal`: for exponential class counts,
   `R_i = IF^{u_i - u_{i-1}}` and equal boundaries minimize `max_i R_i`, giving
   `R_1 = ⋯ = R_N = IF^{1/N}`.
6. `optimal_N_antitone`: `η_{A_i} < η_{A_j} ⇒ N*_{A_i} ≥ N*_{A_j}` for *any* choice of
   minimizers of `S_η(N) = b(N) + (N-1)η`, and `exists_optimal_N` (existence when `η > 0`).
-/
import Mathlib

open Finset

namespace CLTR

section Mismatch

variable {κ : Type*}

/-- Step 1. If `∑ p = ∑ q` and `0 ≤ 𝓛_c ≤ B`, then
`|∑ (p_c - q_c) 𝓛_c| ≤ (B/2) ∑ |p_c - q_c|`. -/
theorem task_sub_bal_le (C : Finset κ) (p q L : κ → ℝ) {B : ℝ}
    (hsum : ∑ c ∈ C, p c = ∑ c ∈ C, q c) (hL : ∀ c ∈ C, 0 ≤ L c ∧ L c ≤ B) :
    |∑ c ∈ C, p c * L c - ∑ c ∈ C, q c * L c| ≤ B / 2 * ∑ c ∈ C, |p c - q c| := by
  have hzero : ∑ c ∈ C, (p c - q c) * (B / 2) = 0 := by
    rw [← sum_mul, sum_sub_distrib, hsum, sub_self, zero_mul]
  have hshift : ∑ c ∈ C, p c * L c - ∑ c ∈ C, q c * L c =
      ∑ c ∈ C, (p c - q c) * (L c - B / 2) := by
    calc ∑ c ∈ C, p c * L c - ∑ c ∈ C, q c * L c
        = ∑ c ∈ C, ((p c - q c) * (L c - B / 2) + (p c - q c) * (B / 2)) := by
          rw [← sum_sub_distrib]; exact sum_congr rfl fun c _ => by ring
      _ = ∑ c ∈ C, (p c - q c) * (L c - B / 2) := by rw [sum_add_distrib, hzero, add_zero]
  rw [hshift, mul_sum]
  refine (abs_sum_le_sum_abs _ _).trans (sum_le_sum fun c hc => ?_)
  rw [abs_mul, mul_comm]
  have := hL c hc
  exact mul_le_mul_of_nonneg_right (abs_le.2 ⟨by linarith, by linarith⟩) (abs_nonneg _)

/-- Chord inequality for `|r - 1|` on `[a, R a]` (the convex function lies below its chord). -/
theorem abs_sub_one_chord {a R r : ℝ} (ha1 : a ≤ 1) (hRa : 1 ≤ R * a) (hr1 : a ≤ r)
    (hr2 : r ≤ R * a) :
    (R * a - a) * |r - 1| ≤ (1 - a) * (R * a - r) + (R * a - 1) * (r - a) := by
  rcases le_total 1 r with h | h
  · rw [abs_of_nonneg (by linarith)]; nlinarith
  · rw [abs_of_nonpos (by linarith)]; nlinarith

/-- Step 2. Total-variation bound in terms of the local imbalance factor `R`:
for positive counts `n_c` with `n_c ≤ R n_d` for all `c, d`, `p_c = n_c / ∑ n` and the uniform
distribution `q_c = 1/|C|`,
`TV(p, q) = ½ ∑ |p_c - q_c| ≤ (√R - 1)/(√R + 1)`. -/
theorem tv_le (C : Finset κ) (hC : C.Nonempty) (n : κ → ℝ) {R : ℝ}
    (hpos : ∀ c ∈ C, 0 < n c) (hR : ∀ c ∈ C, ∀ d ∈ C, n c ≤ R * n d) :
    (1 / 2) * ∑ c ∈ C, |n c / (∑ d ∈ C, n d) - 1 / C.card| ≤
      (Real.sqrt R - 1) / (Real.sqrt R + 1) := by
  set N := ∑ d ∈ C, n d with hN
  set k : ℝ := (C.card : ℝ) with hk
  have hk0 : 0 < k := by rw [hk]; exact_mod_cast hC.card_pos
  have hN0 : 0 < N := sum_pos hpos hC
  obtain ⟨c0, hc0, hmin⟩ := exists_min_image C n hC
  have hR1 : 1 ≤ R := by
    have := hR c0 hc0 c0 hc0
    have := hpos c0 hc0
    nlinarith
  -- likelihood ratios `r_c = k n_c / N` and their minimum `a`
  set r : κ → ℝ := fun c => k * n c / N with hr
  set a := k * n c0 / N with ha
  have ha0 : 0 < a := by have := hpos c0 hc0; positivity
  have hra : ∀ c ∈ C, a ≤ r c := fun c hc => by
    simp only [hr, ha]; gcongr; exact hmin c hc
  have hrR : ∀ c ∈ C, r c ≤ R * a := fun c hc => by
    simp only [hr, ha]
    rw [mul_div_assoc', mul_left_comm]
    gcongr
    exact hR c hc c0 hc0
  have hmean : ∑ c ∈ C, (1 / k) * r c = 1 := by
    simp only [hr]
    have : ∑ c ∈ C, 1 / k * (k * n c / N) = (∑ c ∈ C, n c) / N := by
      rw [sum_div]; refine sum_congr rfl fun c _ => by field_simp
    rw [this, ← hN, div_self hN0.ne']
  have hq1 : ∑ _c ∈ C, (1 / k) = 1 := by
    rw [sum_const, nsmul_eq_mul, ← hk]; field_simp
  have ha1 : a ≤ 1 := by
    calc a = ∑ _c ∈ C, (1 / k) * a := by rw [← sum_mul, hq1, one_mul]
      _ ≤ ∑ c ∈ C, (1 / k) * r c := sum_le_sum fun c hc => by gcongr; exact hra c hc
      _ = 1 := hmean
  have hRa1 : 1 ≤ R * a := by
    calc 1 = ∑ c ∈ C, (1 / k) * r c := hmean.symm
      _ ≤ ∑ _c ∈ C, (1 / k) * (R * a) := sum_le_sum fun c hc => by gcongr; exact hrR c hc
      _ = R * a := by rw [← sum_mul, hq1, one_mul]
  -- `|p_c - q| = (1/k) |r_c - 1|`
  have hpq : ∀ c ∈ C, |n c / N - 1 / k| = (1 / k) * |r c - 1| := fun c _ => by
    have hk1 : (0 : ℝ) < 1 / k := one_div_pos.2 hk0
    rw [show n c / N - 1 / k = (1 / k) * (r c - 1) by simp only [hr]; field_simp, abs_mul,
      abs_of_pos hk1]
  rw [sum_congr rfl hpq, ← mul_sum]
  set S := ∑ c ∈ C, |r c - 1| with hS
  -- weighted chord inequality: `(Ra - a) · (S/k) ≤ 2 (1-a)(Ra-1)`
  have hchord : (R * a - a) * ((1 / k) * S) ≤ 2 * (1 - a) * (R * a - 1) := by
    calc (R * a - a) * ((1 / k) * S) = ∑ c ∈ C, (1 / k) * ((R * a - a) * |r c - 1|) := by
          rw [hS, mul_sum, mul_sum]; refine sum_congr rfl fun c _ => by ring
      _ ≤ ∑ c ∈ C, (1 / k) * ((1 - a) * (R * a - r c) + (R * a - 1) * (r c - a)) :=
          sum_le_sum fun c hc => by
            gcongr; exact abs_sub_one_chord ha1 hRa1 (hra c hc) (hrR c hc)
      _ = (1 - a) * (R * a) * ∑ c ∈ C, (1 / k) - (1 - a) * ∑ c ∈ C, (1 / k) * r c +
            (R * a - 1) * ∑ c ∈ C, (1 / k) * r c - (R * a - 1) * a * ∑ c ∈ C, (1 / k) := by
          simp only [mul_sum, ← sum_sub_distrib, ← sum_add_distrib]
          refine sum_congr rfl fun c _ => by ring
      _ = 2 * (1 - a) * (R * a - 1) := by rw [hmean, hq1]; ring
  set s := Real.sqrt R with hs
  have hs1 : 1 ≤ s := by rw [hs]; exact Real.one_le_sqrt.2 hR1
  have hsR : s ^ 2 = R := by rw [hs]; exact Real.sq_sqrt (by linarith)
  have hS0 : 0 ≤ (1 / k) * S := by positivity
  rw [le_div_iff₀ (by linarith)]
  rcases eq_or_lt_of_le hR1 with hReq | hRgt
  · -- `R = 1`: all ratios equal `1`
    have : ∀ c ∈ C, r c = 1 := fun c hc => by
      have h1 := hra c hc; have h2 := hrR c hc; rw [← hReq] at h2
      have : a = 1 := by rw [← hReq] at hRa1; linarith
      linarith
    have hS' : S = 0 := by
      rw [hS]; exact sum_eq_zero fun c hc => by rw [this c hc]; simp
    rw [hS']; rw [← hReq] at hsR
    have : s = 1 := by nlinarith
    rw [this]; norm_num
  · have hpos' : 0 < R * a - a := by nlinarith
    -- AM–GM step: `(1-a)(s²a-1) ≤ a(s-1)²`
    have hamgm : (1 - a) * (R * a - 1) ≤ a * (s - 1) ^ 2 := by
      rw [← hsR]; nlinarith [sq_nonneg (s * a - 1)]
    have key : (R * a - a) * ((1 / 2) * ((1 / k) * S) * (s + 1)) ≤ (R * a - a) * (s - 1) := by
      have h2 : (R * a - a) = a * (s - 1) * (s + 1) := by rw [← hsR]; ring
      nlinarith
    have := le_of_mul_le_mul_left key hpos'
    linarith

end Mismatch

section Penalty

variable {Θ : Type*}

/-- Step 3. If `|𝓛_task - 𝓛_bal| ≤ b` everywhere and `θ_i*` minimizes `𝓛_task`, then
`𝓛_bal(θ_i*) ≤ 𝓛_bal(θ_i^{bal,*}) + 2b` (here `b = B (√R-1)/(√R+1)`). -/
theorem bal_penalty_le {task bal : Θ → ℝ} {b : ℝ} (hb : ∀ θ, |task θ - bal θ| ≤ b)
    {θi θbal : Θ} (hopt : task θi ≤ task θbal) :
    bal θi ≤ bal θbal + 2 * b := by
  have h1 := (abs_le.1 (hb θi)).1
  have h2 := (abs_le.1 (hb θbal)).2
  linarith

/-- Step 4 (first half). For one exactly optimized CL stage, the regularizers are paid for by
the current-task learning pressure `Δ_i = [𝓛_task(θ_{i-1}) - 𝓛_task(θ_i*)]_+`. -/
theorem cl_penalties_le {task Dφ Dψ : Θ → ℝ} {α β : ℝ} {θprev θi : Θ}
    (hDφ0 : Dφ θprev = 0) (hDψ0 : Dψ θprev = 0)
    (hopt : task θi + α * Dφ θi + β * Dψ θi ≤ task θprev + α * Dφ θprev + β * Dψ θprev) :
    α * Dφ θi + β * Dψ θi ≤ max (task θprev - task θi) 0 := by
  rw [hDφ0, hDψ0] at hopt
  exact (by linarith : α * Dφ θi + β * Dψ θi ≤ task θprev - task θi).trans (le_max_left _ _)

/-- Step 4. Forgetting of the transition `i-1 → i`:
`𝓛^bal_{<i}(θ_i*) - 𝓛^bal_{<i}(θ_{i-1}) ≤ Δ_i · min(c_Φ/α_i, L^sm/(m_Ψ β_i))`. -/
theorem forgetting_le [NormedAddCommGroup Θ] {task Dφ Dψ old : Θ → ℝ}
    {α β cφ Lsm mψ : ℝ} {θprev θi : Θ} (hα : 0 < α) (hβ : 0 < β) (hmψ : 0 < mψ)
    (hLsm : 0 ≤ Lsm) (hcφ : 0 ≤ cφ)
    (hDφ0 : Dφ θprev = 0) (hDψ0 : Dψ θprev = 0) (hDφ : 0 ≤ Dφ θi) (hDψ : 0 ≤ Dψ θi)
    (hopt : task θi + α * Dφ θi + β * Dψ θi ≤ task θprev + α * Dφ θprev + β * Dψ θprev)
    (hcalib : old θi - old θprev ≤ cφ * Dφ θi)
    (hsmooth : old θi - old θprev ≤ Lsm / 2 * ‖θi - θprev‖ ^ 2)
    (hstrong : mψ / 2 * ‖θi - θprev‖ ^ 2 ≤ Dψ θi) :
    old θi - old θprev ≤ max (task θprev - task θi) 0 * min (cφ / α) (Lsm / (mψ * β)) := by
  set Δ := max (task θprev - task θi) 0
  have hpen := cl_penalties_le hDφ0 hDψ0 hopt
  have hφ : Dφ θi ≤ Δ / α := by rw [le_div_iff₀ hα]; nlinarith
  have hψ : Dψ θi ≤ Δ / β := by rw [le_div_iff₀ hβ]; nlinarith
  have hout : old θi - old θprev ≤ Δ * (cφ / α) := by
    calc old θi - old θprev ≤ cφ * Dφ θi := hcalib
      _ ≤ cφ * (Δ / α) := by gcongr
      _ = Δ * (cφ / α) := by ring
  have hpar : old θi - old θprev ≤ Δ * (Lsm / (mψ * β)) := by
    have hq : ‖θi - θprev‖ ^ 2 ≤ 2 / mψ * Dψ θi := by
      rw [div_mul_eq_mul_div, le_div_iff₀ hmψ]; nlinarith
    calc old θi - old θprev ≤ Lsm / 2 * ‖θi - θprev‖ ^ 2 := hsmooth
      _ ≤ Lsm / 2 * (2 / mψ * (Δ / β)) := by gcongr; exact hq.trans (by gcongr)
      _ = Δ * (Lsm / (mψ * β)) := by field_simp
  have hΔ : 0 ≤ Δ := le_max_right _ _
  rcases le_total (cφ / α) (Lsm / (mψ * β)) with h | h
  · rw [min_eq_left h]; exact hout
  · rw [min_eq_right h]; exact hpar

end Penalty

section Boundaries

/-- Step 5. For the exponential profile `w(u) = w₀ IF^{-u}`, the local imbalance factor of the
partition `[u_{i-1}, u_i]` is `R_i = w(u_{i-1})/w(u_i) = IF^{u_i - u_{i-1}}`. -/
theorem local_IF_exp {w0 IF u v : ℝ} (hw0 : 0 < w0) (hIF : 0 < IF) :
    (w0 * IF ^ (-u)) / (w0 * IF ^ (-v)) = IF ^ (v - u) := by
  rw [mul_div_mul_left _ _ hw0.ne', ← Real.rpow_sub hIF]
  congr 1; ring

/-- Pigeonhole: gaps summing to `1` contain one of length at least `1/N`. -/
theorem exists_gap_ge {N : ℕ} (hN : 0 < N) (g : Fin N → ℝ) (hsum : ∑ i, g i = 1) :
    ∃ i, 1 / (N : ℝ) ≤ g i := by
  by_contra h
  push Not at h
  have : ∑ i, g i < ∑ _i : Fin N, 1 / (N : ℝ) :=
    sum_lt_sum_of_nonempty ⟨⟨0, hN⟩, mem_univ _⟩ fun i _ => h i
  rw [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul] at this
  field_simp at this
  linarith

/-- Step 5 (optimal boundaries). For `IF ≥ 1` and any boundaries `0 = u_0 < ⋯ < u_N = 1`
(gaps `g_i = u_i - u_{i-1}` summing to `1`), some partition has `R_i = IF^{g_i} ≥ IF^{1/N}`;
equal gaps achieve `R_i = IF^{1/N}` for all `i`. Moreover, if every `R_i ≤ IF^{1/N}` (with
`IF > 1`), all gaps are equal to `1/N`: the equal-local-imbalance rule is the unique optimum. -/
theorem equal_boundaries_optimal {N : ℕ} (hN : 0 < N) {IF : ℝ} (hIF : 1 ≤ IF)
    (g : Fin N → ℝ) (hsum : ∑ i, g i = 1) :
    (∃ i, IF ^ (1 / (N : ℝ)) ≤ IF ^ g i) ∧
      (∀ i : Fin N, IF ^ (fun _ : Fin N => 1 / (N : ℝ)) i = IF ^ (1 / (N : ℝ))) ∧
      (1 < IF → (∀ i, IF ^ g i ≤ IF ^ (1 / (N : ℝ))) → ∀ i, g i = 1 / N) := by
  refine ⟨?_, fun _ => rfl, ?_⟩
  · obtain ⟨i, hi⟩ := exists_gap_ge hN g hsum
    exact ⟨i, Real.rpow_le_rpow_of_exponent_le hIF hi⟩
  · intro hIF1 hall
    have hle : ∀ i, g i ≤ 1 / N := fun i => (Real.rpow_le_rpow_left_iff hIF1).1 (hall i)
    intro i
    by_contra hne
    have hlt : g i < 1 / N := lt_of_le_of_ne (hle i) hne
    have : ∑ j, g j < ∑ _j : Fin N, 1 / (N : ℝ) :=
      sum_lt_sum (fun j _ => hle j) ⟨i, mem_univ _, hlt⟩
    rw [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul] at this
    have hN' : (N : ℝ) ≠ 0 := by exact_mod_cast hN.ne'
    field_simp at this
    linarith

/-- The bias function `b(R) = (√R - 1)/(√R + 1)` is monotone, so minimizing `max_i R_i` also
minimizes the worst-partition bias `max_i b(R_i)`. -/
theorem bias_mono {R R' : ℝ} (h : R ≤ R') :
    (Real.sqrt R - 1) / (Real.sqrt R + 1) ≤ (Real.sqrt R' - 1) / (Real.sqrt R' + 1) := by
  have h1 : Real.sqrt R ≤ Real.sqrt R' := Real.sqrt_le_sqrt h
  have h0 : 0 ≤ Real.sqrt R := Real.sqrt_nonneg R
  rw [div_le_div_iff₀ (by positivity) (by positivity)]
  nlinarith

/-- With equal boundaries, `√R_i = √(IF^{1/N}) = IF^{1/(2N)}` (the form used in Eq. (tradeoff_eq)). -/
theorem sqrt_equal_R {IF : ℝ} (hIF : 0 ≤ IF) (N : ℕ) :
    Real.sqrt (IF ^ (1 / (N : ℝ))) = IF ^ (1 / (2 * (N : ℝ))) := by
  rw [Real.sqrt_eq_rpow, ← Real.rpow_mul hIF]
  congr 1; ring

/-- More partitions never increase the bias term: `N ↦ (IF^{1/(2N)} - 1)/(IF^{1/(2N)} + 1)` is
antitone for `IF ≥ 1`. -/
theorem bias_antitone_N {IF : ℝ} (hIF : 1 ≤ IF) {N M : ℕ} (hN : 0 < N) (hNM : N ≤ M) :
    (IF ^ (1 / (2 * (M : ℝ))) - 1) / (IF ^ (1 / (2 * (M : ℝ))) + 1) ≤
      (IF ^ (1 / (2 * (N : ℝ))) - 1) / (IF ^ (1 / (2 * (N : ℝ))) + 1) := by
  have hexp : 1 / (2 * (M : ℝ)) ≤ 1 / (2 * (N : ℝ)) := by
    have : (0 : ℝ) < N := by exact_mod_cast hN
    gcongr
  have h := Real.rpow_le_rpow_of_exponent_le hIF hexp
  have h0 : 0 ≤ IF ^ (1 / (2 * (M : ℝ))) := Real.rpow_nonneg (by linarith) _
  rw [div_le_div_iff₀ (by positivity) (by positivity)]
  nlinarith

end Boundaries

section OptimalN

/-- The partition-selection score of Eq. (tradeoff_eq): `S_η(N) = b(N) + (N-1)η`. -/
noncomputable def score (b : ℕ → ℝ) (η : ℝ) (N : ℕ) : ℝ := b N + ((N : ℝ) - 1) * η

/-- **Proposition 3.10 (monotonicity).** For any candidate set and any bias profile `b`, if `N_i`
minimizes `S_{η_i}` and `N_j` minimizes `S_{η_j}` with `η_i < η_j`, then `N_i ≥ N_j`. -/
theorem optimal_N_antitone (b : ℕ → ℝ) (cand : Set ℕ) {ηi ηj : ℝ} (hη : ηi < ηj)
    {Ni Nj : ℕ} (hNi : Ni ∈ cand) (hNj : Nj ∈ cand)
    (hmini : ∀ N ∈ cand, score b ηi Ni ≤ score b ηi N)
    (hminj : ∀ N ∈ cand, score b ηj Nj ≤ score b ηj N) :
    Nj ≤ Ni := by
  have h1 := hmini Nj hNj
  have h2 := hminj Ni hNi
  unfold score at h1 h2
  have : ((Ni : ℝ) - Nj) * (ηi - ηj) ≤ 0 := by nlinarith
  have : (Nj : ℝ) ≤ Ni := by nlinarith
  exact_mod_cast this

/-- Existence of an optimal number of partitions `N* ≥ 1` when `η > 0` and the bias term is
bounded (`0 ≤ b(N) ≤ K`, e.g. `K = 2B`). (For `η = 0` the score can fail to attain its infimum.) -/
theorem exists_optimal_N (b : ℕ → ℝ) {η K : ℝ} (hη : 0 < η) (hb : ∀ N, 0 ≤ b N ∧ b N ≤ K) :
    ∃ Nstar, 1 ≤ Nstar ∧ ∀ N, 1 ≤ N → score b η Nstar ≤ score b η N := by
  set T := ⌈K / η⌉₊ + 1
  obtain ⟨Nstar, hmem, hmin⟩ := exists_min_image (Icc 1 (T + 1)) (score b η)
    ⟨1, mem_Icc.2 ⟨le_rfl, by omega⟩⟩
  refine ⟨Nstar, (mem_Icc.1 hmem).1, fun N hN => ?_⟩
  by_cases hNT : N ≤ T + 1
  · exact hmin N (mem_Icc.2 ⟨hN, hNT⟩)
  · have h1 := hmin 1 (mem_Icc.2 ⟨le_rfl, by omega⟩)
    have hK : K / η ≤ (T : ℝ) := by
      have := Nat.le_ceil (K / η); push_cast [T]; linarith
    have hNge : (T : ℝ) + 1 ≤ N := by exact_mod_cast (by omega : T + 1 ≤ N)
    have : K ≤ ((N : ℝ) - 1) * η := by
      rw [div_le_iff₀ hη] at hK; nlinarith
    unfold score at h1 ⊢
    have := (hb 1).2; have := (hb N).1
    push_cast at h1
    linarith

end OptimalN

end CLTR
