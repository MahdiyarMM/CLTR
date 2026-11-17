/-
# Remark 3.8: exponential long-tailed datasets

Sorted class frequencies follow `w(u) = w₀ IF^{-u}` on `u ∈ [0,1]`. For the Head prefix
`P_τ = [0, τ]` and Tail suffix `T_τ = (τ, 1]`:

* `mu_interval`: `μ([a,b]) = ∫_a^b w₀ IF^{-u} du = w₀/log IF · (IF^{-a} - IF^{-b})`.
* `eps_tau_eq`: `ε_τ = μ(T_τ)/μ([0,1]) = (IF^{-τ} - IF^{-1})/(1 - IF^{-1})`.
* `risk_decomposition`: `L_U = (1-ε_τ) L_{P_τ} + ε_τ L_{T_τ}` for the weighted risks with a shared
  regularizer (derived from additivity of the integral).
* `remark3_8_bound`: `dist(θ*_U, S_{P_τ}) ≤ C_τ √(2 M_τ (IF^{-τ} - IF^{-1})/(1 - IF^{-1}))`.
* `remark3_8`: `dist(θ*_U, S_{P_τ}) = O(IF^{-τ/2}) = O(IF_τ^{-1/2})` with `IF_τ = IF^τ`.
-/
import Mathlib
import Proofs.Nonconvex

open Set Filter Asymptotics Topology Metric Real intervalIntegral

namespace CLTR

/-- Mass of an interval under the exponential frequency profile. -/
theorem mu_interval {w0 IF : ℝ} (hIF : 0 < IF) (hIF1 : IF ≠ 1) (a b : ℝ) :
    ∫ u in a..b, w0 * IF ^ (-u) = w0 / log IF * (IF ^ (-a) - IF ^ (-b)) := by
  have hlog : log IF ≠ 0 := by
    intro h
    rcases Real.log_eq_zero.1 h with h | h | h <;> [linarith; exact hIF1 h; linarith]
  have hform : ∀ u : ℝ, IF ^ (-u) = exp (-log IF * u) := fun u => by
    rw [rpow_def_of_pos hIF]; ring_nf
  have hderiv : ∀ u ∈ uIcc a b, HasDerivAt (fun u => -(w0 / log IF) * exp (-log IF * u))
      (w0 * IF ^ (-u)) u := by
    intro u _
    have h := ((hasDerivAt_id u).const_mul (-log IF)).exp.const_mul (-(w0 / log IF))
    rw [hform]
    refine h.congr_deriv ?_
    simp only [id, mul_one]
    field_simp
  rw [integral_eq_sub_of_hasDerivAt hderiv]
  · rw [hform, hform]; ring
  · apply Continuous.intervalIntegrable
    simp_rw [hform]
    fun_prop

/-- Tail mass fraction of the exponential profile: `ε_τ = (IF^{-τ} - IF^{-1})/(1 - IF^{-1})`. -/
theorem eps_tau_eq {w0 IF τ : ℝ} (hw0 : 0 < w0) (hIF : 1 < IF) :
    (∫ u in τ..1, w0 * IF ^ (-u)) / (∫ u in (0 : ℝ)..1, w0 * IF ^ (-u)) =
      (IF ^ (-τ) - IF⁻¹) / (1 - IF⁻¹) := by
  have hIF0 : 0 < IF := by linarith
  have hlog : 0 < log IF := log_pos hIF
  rw [mu_interval hIF0 hIF.ne' , mu_interval hIF0 hIF.ne', neg_zero, rpow_zero,
    rpow_neg_one]
  have : 1 - IF⁻¹ ≠ 0 := by
    have : IF⁻¹ < 1 := inv_lt_one_of_one_lt₀ hIF
    linarith
  field_simp

/-- Weighted risk over `[a,b]`: `L_{[a,b]}(θ) = (1/μ[a,b]) ∫_a^b ℓ_u(θ) w(u) du + R(θ)`. -/
noncomputable def expRisk {Θ : Type*} (ℓ : ℝ → Θ → ℝ) (w : ℝ → ℝ) (R : Θ → ℝ) (a b : ℝ)
    (θ : Θ) : ℝ :=
  (∫ u in a..b, ℓ u θ * w u) / (∫ u in a..b, w u) + R θ

/-- Risk decomposition `L_U = L_{P_τ} + ε_τ (L_{T_τ} - L_{P_τ})` with `ε_τ = μ(T_τ)/μ(U)`. -/
theorem risk_decomposition {Θ : Type*} (ℓ : ℝ → Θ → ℝ) (w : ℝ → ℝ) (R : Θ → ℝ) (τ : ℝ) (θ : Θ)
    (hℓP : IntervalIntegrable (fun u => ℓ u θ * w u) MeasureTheory.volume 0 τ)
    (hℓT : IntervalIntegrable (fun u => ℓ u θ * w u) MeasureTheory.volume τ 1)
    (hwP : IntervalIntegrable w MeasureTheory.volume 0 τ)
    (hwT : IntervalIntegrable w MeasureTheory.volume τ 1)
    (hmP : 0 < ∫ u in (0 : ℝ)..τ, w u) (hmT : 0 < ∫ u in τ..1, w u) :
    expRisk ℓ w R 0 1 θ = expRisk ℓ w R 0 τ θ +
      ((∫ u in τ..1, w u) / (∫ u in (0 : ℝ)..1, w u)) *
        (expRisk ℓ w R τ 1 θ - expRisk ℓ w R 0 τ θ) := by
  unfold expRisk
  rw [← integral_add_adjacent_intervals hℓP hℓT, ← integral_add_adjacent_intervals hwP hwT]
  field_simp
  ring

/-- **Remark 3.8** (explicit bound). Under the Hölder-1/2 error bound for `L_{P_τ}` on `U` and the
bounded loss contrast `|L_{T_τ} - L_{P_τ}| ≤ M_τ` on `U`, the full-data optimum satisfies
`dist(θ*_U, S_{P_τ}) ≤ C_τ √(2 M_τ (IF^{-τ} - IF^{-1})/(1 - IF^{-1}))`. -/
theorem remark3_8_bound {Θ : Type*} [PseudoMetricSpace Θ] {LP LT LU : Θ → ℝ} {w0 IF τ C M : ℝ}
    {S U : Set Θ} (hw0 : 0 < w0) (hIF : 1 < IF) (hτ : τ ≤ 1) (hC : 0 ≤ C)
    (hdecomp : ∀ θ, LU θ = LP θ +
      ((∫ u in τ..1, w0 * IF ^ (-u)) / (∫ u in (0 : ℝ)..1, w0 * IF ^ (-u))) * (LT θ - LP θ))
    (hM : ∀ θ ∈ U, |LT θ - LP θ| ≤ M)
    (herr : ∀ θ ∈ U, ∀ θbar ∈ S, infDist θ S ≤ C * Real.sqrt (LP θ - LP θbar))
    (hSU : S ⊆ U) {θstar θbar : Θ} (hstarU : θstar ∈ U) (hbarS : θbar ∈ S)
    (hstar : ∀ θ, LU θstar ≤ LU θ) :
    infDist θstar S ≤ C * Real.sqrt (2 * M * ((IF ^ (-τ) - IF⁻¹) / (1 - IF⁻¹))) := by
  rw [eps_tau_eq hw0 hIF] at hdecomp
  have hIF0 : 0 < IF := by linarith
  have hε : 0 ≤ (IF ^ (-τ) - IF⁻¹) / (1 - IF⁻¹) := by
    apply div_nonneg
    · rw [sub_nonneg, ← rpow_neg_one]
      exact rpow_le_rpow_of_exponent_le hIF.le (by linarith)
    · have : IF⁻¹ < 1 := inv_lt_one_of_one_lt₀ hIF
      linarith
  exact dist_le_of_perturbation hε hC hdecomp hM herr hstarU hbarS (hSU hbarS) (hstar θbar)

/-- `ε_τ ≤ 2 IF^{-τ}` once `IF ≥ 2`. -/
theorem eps_tau_le {IF τ : ℝ} (hIF : 2 ≤ IF) :
    (IF ^ (-τ) - IF⁻¹) / (1 - IF⁻¹) ≤ 2 * IF ^ (-τ) := by
  have hIF0 : 0 < IF := by linarith
  have hinv : IF⁻¹ ≤ 1 / 2 := by rw [inv_le_comm₀ hIF0 (by norm_num)]; linarith
  have hpos : 0 < 1 - IF⁻¹ := by linarith
  have hinv0 : 0 < IF⁻¹ := inv_pos.2 hIF0
  have hr : 0 ≤ IF ^ (-τ) := rpow_nonneg hIF0.le _
  rw [div_le_iff₀ hpos]
  nlinarith

/-- **Remark 3.8** (asymptotic form). For a family of exponential long-tailed problems indexed by
`IF` with uniform constants `C_τ, M_τ`, `dist(θ*_U(IF), S_{P_τ}(IF)) = O(IF^{-τ/2})`. -/
theorem remark3_8 {Θ : Type*} [PseudoMetricSpace Θ] {LP LT LU : ℝ → Θ → ℝ} {w0 τ C M : ℝ}
    {S U : ℝ → Set Θ} (hw0 : 0 < w0) (hτ : τ ≤ 1) (hC : 0 ≤ C) (hM0 : 0 ≤ M)
    (hdecomp : ∀ IF, 1 < IF → ∀ θ, LU IF θ = LP IF θ +
      ((∫ u in τ..1, w0 * IF ^ (-u)) / (∫ u in (0 : ℝ)..1, w0 * IF ^ (-u))) *
        (LT IF θ - LP IF θ))
    (hM : ∀ IF, ∀ θ ∈ U IF, |LT IF θ - LP IF θ| ≤ M)
    (herr : ∀ IF, ∀ θ ∈ U IF, ∀ θbar ∈ S IF,
      infDist θ (S IF) ≤ C * Real.sqrt (LP IF θ - LP IF θbar))
    (hSU : ∀ IF, S IF ⊆ U IF) (hS : ∀ IF, (S IF).Nonempty) (θstar : ℝ → Θ)
    (hstarU : ∀ IF, θstar IF ∈ U IF) (hstar : ∀ IF, ∀ θ, LU IF (θstar IF) ≤ LU IF θ) :
    (fun IF => infDist (θstar IF) (S IF)) =O[atTop] (fun IF => IF ^ (-τ / 2)) := by
  refine IsBigO.of_bound (C * Real.sqrt (4 * M)) ?_
  filter_upwards [eventually_ge_atTop (2 : ℝ)] with IF hIF
  have hIF0 : 0 < IF := by linarith
  obtain ⟨θbar, hbar⟩ := hS IF
  rw [Real.norm_of_nonneg infDist_nonneg, Real.norm_of_nonneg (rpow_nonneg hIF0.le _)]
  calc infDist (θstar IF) (S IF)
      ≤ C * Real.sqrt (2 * M * ((IF ^ (-τ) - IF⁻¹) / (1 - IF⁻¹))) :=
        remark3_8_bound hw0 (by linarith) hτ hC (hdecomp IF (by linarith)) (hM IF) (herr IF)
          (hSU IF) (hstarU IF) hbar (hstar IF)
    _ ≤ C * Real.sqrt (2 * M * (2 * IF ^ (-τ))) := by gcongr; exact eps_tau_le hIF
    _ = C * Real.sqrt (4 * M) * IF ^ (-τ / 2) := by
        rw [mul_assoc C]
        congr 1
        rw [show 2 * M * (2 * IF ^ (-τ)) = 4 * M * IF ^ (-τ) by ring,
          Real.sqrt_mul (by positivity), Real.sqrt_eq_rpow (IF ^ (-τ)),
          ← rpow_mul hIF0.le]
        congr 2; ring

/-- The boundary imbalance factor `IF_τ = w(0)/w(τ) = IF^τ`, and `IF^{-τ/2} = IF_τ^{-1/2}`. -/
theorem boundary_IF {w0 IF τ : ℝ} (hw0 : 0 < w0) (hIF : 0 < IF) :
    w0 / (w0 * IF ^ (-τ)) = IF ^ τ ∧ IF ^ (-τ / 2) = (IF ^ τ) ^ (-(1 : ℝ) / 2) := by
  constructor
  · rw [rpow_neg hIF.le]; field_simp
  · rw [← rpow_mul hIF.le]; congr 1; ring

end CLTR
